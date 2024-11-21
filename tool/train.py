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

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils.data
from tensorboardX import SummaryWriter

import _init_paths
from lib.config import cfg, update_config
from lib.models import get_pose_net
from lib.core.loss import ConfsMSELoss
from lib.core.function import train, validate
from lib.utils.utils import create_logger, save_checkpoint

import lib.dataset


def parse_args():
    parser = argparse.ArgumentParser(description='training of continuous heatmap regression')
    parser.add_argument('--cfg', required=True, type=str, help='experiment config file name')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="modify by the command-line")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = get_pose_net(cfg, is_train=True)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    criterion = ConfsMSELoss(cfg.LOSS.REG_INVIS_CONF_TO_ZERO).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = eval('lib.dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, transform=transform
    )
    valid_dataset = eval('lib.dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, transform=transform
    )    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    begin_epoch = 0
    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        best_perf = checkpoint['perf']
        begin_epoch = checkpoint['epoch']
        last_epoch = begin_epoch - 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint (epoch {})".format(begin_epoch))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.NUM_EPOCH):
        train(cfg, train_loader, model, criterion, optimizer, epoch, output_dir, writer_dict)

        lr_scheduler.step()
        if (epoch + 1) % cfg.TRAIN.VALIDATE_EVERY > 0:
            continue

        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, output_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, output_dir)

    final_model_state_file = os.path.join(output_dir, 'final_state.pth')
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()