# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Shengxiang Hu (hushengxiang@njust.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from lib.utils.transforms import transform_preds


def heatmap_to_coord(heatmap):
    batch_size = heatmap.shape[0]
    num_joints = heatmap.shape[1]
    width = heatmap.shape[3]
    heatmap_reshaped = heatmap.reshape((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmap_reshaped, dim=2)

    coord = idx.unsqueeze(-1).repeat(1, 1, 2).to(torch.float32)
    coord[:, :, 0] = (coord[:, :, 0]) % width
    coord[:, :, 1] = torch.floor((coord[:, :, 1]) / width)

    return coord, maxvals.unsqueeze(-1)


def get_final_preds(pred_coord, center, scale, img_size):
    final_preds = pred_coord.copy()
    for i in range(pred_coord.shape[0]):
        final_preds[i] = transform_preds(
            pred_coord[i], center[i], scale[i], img_size
        )

    return final_preds
