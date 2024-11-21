from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

    
class ConfsMSELoss(nn.Module):
    def __init__(self, reg_invis_conf_to_zero):
        super(ConfsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.reg_invis_conf_to_zero = reg_invis_conf_to_zero 

    def forward(self, pred_conf, gt_conf, target_weight):
        conf_mask = target_weight.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        gt_conf = gt_conf.unsqueeze(1).repeat(1, pred_conf.size(1), 1, 1, 1)

        if not self.reg_invis_conf_to_zero:
            pred_conf = pred_conf * conf_mask

        return 0.5 * self.criterion(pred_conf, gt_conf)