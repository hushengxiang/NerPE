from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models import pose_hrnet, pose_resnet


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class FeatZoomModule(nn.Module):
    def __init__(self, zoom_factor, num_channels, upsample_type):
        super(FeatZoomModule, self).__init__()
        num_scale = round(math.log(zoom_factor, 2))
        if 2 ** num_scale != zoom_factor:
            raise NotImplementedError
        
        self.up_type = upsample_type
        self.layers = self.make_layers(num_scale, num_channels)

    def make_layers(self, num_scale, num_channels):
        layers = []
        if num_scale > 0:
            for _ in range(0, num_scale):
                if self.up_type == 'upconv':
                    layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    layers.append(nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False))
                elif self.up_type == 'deconv':
                    layers.append(nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 1, bias=False))
                else:
                    raise NotImplementedError
                
                layers.append(nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM))
                layers.append(nn.ReLU(inplace=True))

        elif num_scale < 0:
            for _ in range(num_scale, 0):
                layers.append(nn.Conv2d(num_channels, num_channels, 3, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM))
                layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(num_channels, num_channels, 1, 1, 0))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ImpNeuRepNodule(nn.Module):
    def __init__(self, in_nchs, out_nchs, num_component, is_pos_mlp, hidden_list):
        super(ImpNeuRepNodule, self).__init__()
        self.num_component = num_component
        pos_dim = num_component * 4 + 2
        self.pos_mlp = None

        if is_pos_mlp:
            self.pos_mlp = self.make_pos_mlp(pos_dim)
            self.layers = self.make_inr_layers(in_nchs + 32, out_nchs, hidden_list)
        else:
            self.layers = self.make_inr_layers(in_nchs + pos_dim, out_nchs, hidden_list)

    def make_pos_mlp(self, in_dim):
        layers = []
        layers.append(nn.Conv2d(in_dim, 32, 1, 1, 0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32, 32, 1, 1, 0))

        return nn.Sequential(*layers)

    def make_inr_layers(self, in_dim, out_dim, hidden_list):
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Conv2d(lastv, hidden, 1, 1, 0))
            layers.append(nn.ReLU(inplace=True))
            lastv = hidden
        layers.append(nn.Conv2d(lastv, out_dim, 1, 1, 0))

        return nn.Sequential(*layers)

    def positional_encoding(self, x):
        emb_component = [x]

        for i in range(self.num_component):
            emb_component.append(torch.sin(2 ** i * math.pi * x))
            emb_component.append(torch.cos(2 ** i * math.pi * x))

        return torch.cat(emb_component, dim=1)

    def forward(self, q_feat, rel_coord):
        q_pos = self.positional_encoding(rel_coord)
        if self.pos_mlp != None:
            q_pos = self.pos_mlp(q_pos)        
        out = self.layers(torch.cat((q_feat, q_pos), dim=1))

        return out


class ImplicitPoseEstNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ImplicitPoseEstNet, self).__init__()
        if cfg.MODEL.BACKONE == 'resnet':
            self.backbone = pose_resnet.build_backbone(cfg)
        elif cfg.MODEL.BACKONE == 'hrnet':
            self.backbone = pose_hrnet.build_backbone(cfg)
        else:
            raise NotImplementedError

        self.zoom_layers = FeatZoomModule(
            cfg.MODEL.POST_FEAT_SIZE[0] / cfg.MODEL.PRE_FEAT_SIZE[0],
            cfg.MODEL.NUM_CHANNELS, cfg.MODEL.UPSAMPLE_TYPE
            )

        self.inr_layers = ImpNeuRepNodule(
            cfg.MODEL.NUM_CHANNELS, 
            cfg.MODEL.NUM_JOINTS, 
            cfg.MODEL.POS_ENC_COMPONENT,
            cfg.MODEL.POS_LEARNABLE_PROJ,
            cfg.MODEL.HIDDEN_CHANNELS_LIST
            )
    
    def local_ensemble(self, feat, feat_coord, query_coord, is_train):
        temp_coord = query_coord.clone()

        # ?????????????????????????????????????
        h, w = feat.shape[-2:]
        temp_coord[:, 0, :, :] *= max(h, w) / w
        temp_coord[:, 1, :, :] *= max(h, w) / h

        vx_lst = vy_lst = [-1, 1]
        rx = 2 / w / 2
        ry = 2 / h / 2

        preds, areas = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                temp_query = temp_coord.clone().permute(0, 2, 3, 1)
                temp_query[:, :, :, 0] += vx * rx
                temp_query[:, :, :, 1] += vy * ry
                temp_query.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(feat, temp_query, mode='nearest', align_corners=False)
                center = F.grid_sample(feat_coord, temp_query, mode='nearest', align_corners=False)

                rel_coord = query_coord - center
                rel_coord *= max(feat.shape[-2:])

                preds.append(self.inr_layers(q_feat, rel_coord))
                areas.append(torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :]) + 1e-9)

        if is_train:
            return torch.stack(preds, dim=1)
        else:
            tot_area = torch.stack(areas).sum(dim=0)
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

            conf_grid = 0
            for pred, area in zip(preds, areas):
                conf_grid = conf_grid + pred * (area / tot_area).unsqueeze(1)
            return conf_grid

    def forward(self, input, feat_coord, query_coord, is_train):
        feature = self.zoom_layers(self.backbone(input))
        heatmap = self.local_ensemble(feature, feat_coord, query_coord, is_train)
        
        return heatmap

    def init_weights(self, pretrained=''):
        logger.info('=> init weights in backbone')
        self.backbone.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = ImplicitPoseEstNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model