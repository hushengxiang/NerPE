# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def calc_dists(pred, target, joints_vis, normalize):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((pred.shape[1], pred.shape[0]))
    for n in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            if joints_vis[n, c] > 0.0:
                normed_pred = pred[n, c, :] / normalize[n]
                normed_target = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_pred - normed_target)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(pred, target, joints_vis, img_size):
    w = img_size[0]
    h = img_size[1]
    norm = np.ones((pred.shape[0], 2)) * np.array([w, h]) / 10
    dists = calc_dists(pred, target, joints_vis, norm)

    num_joints = pred.shape[1]
    acc = np.zeros((num_joints + 1))
    avg_acc = 0
    cnt = 0

    for i in range(num_joints):
        acc[i + 1] = dist_acc(dists[i])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt

