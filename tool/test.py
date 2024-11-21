# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
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
import pprint

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils.data

import _init_paths
from lib.config import cfg, update_config
from lib.models import get_pose_net
from lib.core.function import validate
from lib.utils.utils import create_logger

import lib.dataset


def parse_args():
    parser = argparse.ArgumentParser(description='testing of continuous heatmap regression')
    parser.add_argument('--cfg', required=True, type=str, help='experiment config file name')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="modify by the command-line")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, output_dir, _ = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    model = get_pose_net(cfg, is_train=False)

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    valid_dataset = eval('lib.dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, transform=transform
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    logger.info('=> performing validation')
    validate(cfg, valid_loader, valid_dataset, model, output_dir)


if __name__ == '__main__':
    main()
