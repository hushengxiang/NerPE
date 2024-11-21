# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# The part of continuous heatmap regression in NerPE:
# Written by Shengxiang Hu (hushengxiang@njust.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.evaluate import accuracy
from lib.core.inference import heatmap_to_coord, get_final_preds
from lib.utils.vis import save_debug_images
from lib.utils.transforms import flip_back


logger = logging.getLogger(__name__)


class CoordGenerator(nn.Module):
    def __init__(self, cfg):
        super(CoordGenerator).__init__()
        self.feat_size = np.array(cfg.MODEL.POST_FEAT_SIZE)
        global_info = self.make_coord_grid(self.feat_size)
        self.global_grid, self.glo_cell_size = global_info

        self.sqrt_num = cfg.MODEL.SQRT_NUM_SAMPLE_PER_CELL
        local_info = self.make_coord_grid(np.array([self.sqrt_num, self.sqrt_num]))
        self.local_grid, self.loc_cell_size = local_info

        self.up_factor = cfg.MODEL.UPSAMPLING_FACTOR
        self.up_grid = self.make_coord_grid(np.array([self.up_factor, self.up_factor]))[0]

    def make_coord_grid(self, grid_size):
        g_center = (grid_size - 1) / 2
        cell_size = 2 / max(grid_size)

        cell_x = (range(grid_size[0]) - g_center[0]) * cell_size
        cell_y = (range(grid_size[1]) - g_center[1]) * cell_size

        cell_x = torch.from_numpy(cell_x).float()
        cell_y = torch.from_numpy(cell_y).float()

        Y, X = torch.meshgrid(cell_y, cell_x, indexing='ij')
        coord_grid = torch.stack((X, Y), dim=0)

        return coord_grid.cuda(), cell_size

    def random_sampling(self, batch_size, is_subdiv):
        ''' the smaller the batch size, the greater the role of xxx uniform sampling '''
        global_grid = self.global_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        anchor = F.interpolate(global_grid, scale_factor=self.sqrt_num, mode='nearest')
        
        if is_subdiv:
            local_grid = self.local_grid.unsqueeze(0).repeat(batch_size, 1, self.feat_size[1], self.feat_size[0])
            offset = local_grid + (torch.rand_like(local_grid) * 2 - 1) * self.loc_cell_size / 2
            query_coord = anchor + offset * self.glo_cell_size / 2
        else:
            offset = torch.rand_like(anchor) * 2 - 1
            query_coord = anchor + offset * self.glo_cell_size / 2

        return global_grid, query_coord

    def gridpoint_sampling(self, batch_size):
        global_grid = self.global_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        anchor =  F.interpolate(global_grid, scale_factor=self.up_factor, mode='nearest')

        offset = self.up_grid.unsqueeze(0).repeat(batch_size, 1, self.feat_size[1], self.feat_size[0])
        query_coord = anchor + offset * self.glo_cell_size / 2

        return global_grid, query_coord


def get_gt_conf(coord, target, target_weight, target_type, scale_param):
    conf_weight = target_weight.unsqueeze(-1).unsqueeze(-1)
    error = coord.unsqueeze(1) - target.unsqueeze(-1).unsqueeze(-1)
    error_x_sqr = error[:, :, 0, :, :] ** 2
    error_y_sqr = error[:, :, 1, :, :] ** 2

    if target_type == 'gaussian':
        gt_conf = torch.exp(- (error_x_sqr + error_y_sqr) / (2 * scale_param ** 2)) * conf_weight
    elif target_type == 'laplacian':
        gt_conf = torch.exp(- torch.sqrt(error_x_sqr + error_y_sqr) / scale_param) * conf_weight
    else:
        raise NotImplementedError

    return gt_conf


def train(config, train_loader, model, criterion, optimizer, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    cg = CoordGenerator(config)

    target_type = config.MODEL.TARGET_TYPE
    scale_param = config.MODEL.SCALE_PARAMETER
    is_subdiv = config.MODEL.IS_SUB_DIVISION

    end = time.time()
    for i, inputs in enumerate(train_loader):
        input, target, norm_target, target_weight, meta = inputs
        data_time.update(time.time() - end)
        num_image = input.size(0)

        input = input.cuda(non_blocking=True)
        norm_target = norm_target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        feat_coord, query_coord = cg.random_sampling(num_image, is_subdiv)
        gt_conf = get_gt_conf(query_coord, norm_target, target_weight, target_type, scale_param)
        pred_conf = model(input, feat_coord, query_coord, is_train=True)

        loss = criterion(pred_conf, gt_conf, target_weight)
        losses.update(loss.item(), num_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=num_image/batch_time.val, data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, val_dataset, model, output_dir, writer_dict=None):
    batch_time = AverageMeter()
    acc = AverageMeter()

    model.eval()
    cg = CoordGenerator(config)

    up_factor = config.MODEL.UPSAMPLING_FACTOR
    img_size = config.MODEL.IMAGE_SIZE
    hm_size = np.array(config.MODEL.POST_FEAT_SIZE) * up_factor
    stride = img_size[0] / hm_size[0]

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))

    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, inputs in enumerate(val_loader):
            input, target, norm_target, target_weight, meta = inputs
            num_image = input.size(0)

            input = input.cuda(non_blocking=True)

            feat_coord, query_coord = cg.gridpoint_sampling(num_image)
            heatmap = model(input, feat_coord, query_coord, is_train=False)

            if config.TEST.FLIP_TEST:
                heatmap_flipped = model(input.flip(3), feat_coord, query_coord, is_train=False)
                heatmap_flipped = flip_back(heatmap_flipped, val_dataset.flip_pairs)
                heatmap = (heatmap + heatmap_flipped) * 0.5

            coord, maxvals = heatmap_to_coord(heatmap)
            coord = (coord + 0.5) * stride - 0.5

            _, avg_acc, cnt = accuracy(
                coord.cpu().numpy(), target.numpy(), target_weight, img_size)
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            final_preds = get_final_preds(coord.cpu().numpy(), c, s, img_size)

            all_preds[idx:idx + num_image, :, 0:2] = final_preds[:, :, 0:2]
            all_preds[idx:idx + num_image, :, 2:3] = maxvals.cpu().numpy()

            all_boxes[idx:idx + num_image, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_image, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_image, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_image, 5] = score
            image_path.extend(meta['image'])

            idx += num_image

            if i % config.TEST.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, coord, heatmap, prefix)


        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0