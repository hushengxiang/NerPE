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

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,1,2,3)
_C.WORKERS = 24
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# cudnn related settings
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# dataset related settings
_C.DATASET = CN()
_C.DATASET.DATASET = 'mpii'
_C.DATASET.ROOT = ''
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.COLOR_RGB = False
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8

# model related settings
_C.MODEL = CN()
_C.MODEL.NAME = 'NerPE'
_C.MODEL.BACKONE = ''
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.SCALE_PARAMETER = 0.06
_C.MODEL.IMAGE_SIZE = [256, 256]
_C.MODEL.PRE_FEAT_SIZE = [64, 64]
_C.MODEL.POST_FEAT_SIZE = [16, 16]
_C.MODEL.NUM_CHANNELS = 256
_C.MODEL.UPSAMPLE_TYPE = 'deconv'
_C.MODEL.POS_ENC_COMPONENT = 0
_C.MODEL.POS_LEARNABLE_PROJ = False
_C.MODEL.IS_SUB_DIVISION = False
_C.MODEL.HIDDEN_CHANNELS_LIST = [256, 256, 256]
_C.MODEL.SQRT_NUM_SAMPLE_PER_CELL = 8
_C.MODEL.UPSAMPLING_FACTOR = 4
_C.MODEL.EXTRA = CN(new_allowed=True)

# loss related settings
_C.LOSS = CN()
_C.LOSS.REG_INVIS_CONF_TO_ZERO = False

# train related settings
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_EPOCH = 140
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.VALIDATE_EVERY = 1
_C.TRAIN.PRINT_FREQ = 100

# test related settings
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.PRINT_FREQ = 100
_C.TEST.EXTRA = CN(new_allowed=True)

# debug related settings
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

