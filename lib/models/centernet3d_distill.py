import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.fusion import Fusion
from lib.models.centernet3d import CenterNet3D

from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.losses.head_distill_loss import compute_head_distill_loss
from lib.losses.feature_distill_loss import compute_backbone_l1_loss
from lib.losses.feature_distill_loss import compute_backbone_resize_affinity_loss
from lib.losses.feature_distill_loss import compute_backbone_bkl_loss



class MonoDistill(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4, flag='training', model_type='distill', kd_type=['cross_kd']):
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.centernet_rgb = CenterNet3D(backbone=backbone, neck=neck, num_class=num_class, downsample=downsample, flag=flag, model_type=model_type)
        self.centernet_depth = CenterNet3D(backbone=backbone, neck=neck, num_class=num_class, downsample=downsample, flag=flag, model_type=model_type)

        for i in self.centernet_depth.parameters():
            i.requires_grad = False


        channels = self.centernet_rgb.backbone.channels #[16, 32, 64, 128, 256, 512]
        input_channels = channels[2:]
        out_channels = channels[2:]
        mid_channel = channels[-1]
        rgb_fs = nn.ModuleList()
        for idx, in_channel in enumerate(input_channels):
            rgb_fs.append(Fusion(in_channel, mid_channel, out_channels[idx], idx < len(input_channels)-1))
        self.rgb_fs = rgb_fs[::-1]


        self.adapt_list = ['adapt_layer8','adapt_layer16','adapt_layer32']
        for i, adapt_name in enumerate(self.adapt_list):
            fc = nn.Sequential(
                nn.Conv2d(channels[i+3], channels[i+3], kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+3], channels[i+3], kernel_size=1, padding=0, bias=True)
            )
            #self.fill_fc_weights(fc)
            self.__setattr__(adapt_name, fc)

        self.flag = flag
        self.kd_type = kd_type

    def align_scale(self, stu_feats, tea_feats):
        algn_feats = []
        for i in range(len(stu_feats)):
            stu_feat = stu_feats[i]
            tea_feat = tea_feats[i]
            N, C, H, W = stu_feat.size()
            # normalize student feature
            stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
            stu_mean = stu_feat.mean(dim=-1, keepdim=True)
            stu_std = stu_feat.std(dim=-1, keepdim=True)
            stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
            #
            tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
            tea_mean = tea_feat.mean(dim=-1, keepdim=True)
            tea_std = tea_feat.std(dim=-1, keepdim=True)
            stu_feat = stu_feat * tea_std + tea_mean
            algn_feats.append(stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3))
        return algn_feats

    def forward(self, input, target=None):
        if self.flag == 'training' and target != None:
            rgb = input['rgb']
            depth = input['depth']
            rgb_feat, rgb_neck, rgb_outputs = self.centernet_rgb(rgb)
            depth_feat, depth_neck, depth_outputs = self.centernet_depth(depth)
            if 'cross_kd' in self.kd_type:
                align_rgb_feat = self.align_scale([rgb_neck], [depth_neck])[0]
                cross_rgb_outputs = self.centernet_depth.forward_head(align_rgb_feat)
            
            if 'bkl_kd' in self.kd_type:
                align_depth_feat = self.align_scale([depth_neck], [rgb_neck])[0]
                cross_depth_outputs = self.centernet_rgb.forward_head(align_depth_feat)

            ### rgb feature fusion
            ### References: Distilling Knowledge via Knowledge Review, CVPR'21
            shapes = [rgb_feat_item.shape[2:] for rgb_feat_item in rgb_feat[::-1]]
            out_shapes = shapes
            x = rgb_feat[::-1]

            results = []
            out_features, res_features = self.rgb_fs[0](x[0], out_shape=out_shapes[0])
            results.append(out_features)
            for features, rgb_f, shape, out_shape in zip(x[1:], self.rgb_fs[1:], shapes[1:], out_shapes[1:]):
                out_features, res_features = rgb_f(features, res_features, shape, out_shape)
                results.insert(0, out_features)

            ### adapt layer
            distill_feature = []
            for i, adapt in enumerate(self.adapt_list):
                distill_feature.append(self.__getattr__(adapt)(results[i+1]))

            ### rgb_loss
            rgb_loss, rgb_stats_batch = compute_centernet3d_loss(rgb_outputs, target)

            distll_loss = {}
            ### distillation loss
            if 'head_kd' in self.kd_type:
                head_loss, _ = compute_head_distill_loss(rgb_outputs, depth_outputs, target)
                distll_loss['head_loss'] = head_loss
            if 'l1_kd' in self.kd_type:
                backbone_loss_l1 = compute_backbone_l1_loss(distill_feature, depth_feat[-3:], target)
                distll_loss['backbone_loss_l1'] = backbone_loss_l1
            if 'affinity_kd' in self.kd_type:
                backbone_loss_affinity = compute_backbone_resize_affinity_loss(distill_feature, depth_feat[-3:])
                distll_loss['backbone_loss_affinity'] = backbone_loss_affinity
            if 'cross_kd' in self.kd_type:
                cross_head_loss, _ = compute_head_distill_loss(cross_rgb_outputs, depth_outputs, target)
                distll_loss['cross_head_loss'] = cross_head_loss
            if 'bkl_kd' in self.kd_type:
                bkl_kd_loss= compute_backbone_bkl_loss(distill_feature, depth_feat[-3:])
                distll_loss['bkl_kd'] = bkl_kd_loss

            return rgb_loss, distll_loss, rgb_stats_batch

        else:
            rgb = input['rgb']
            rgb_feat, rgb_neck, rgb_outputs = self.centernet_rgb(rgb)

            return rgb_feat, rgb_neck, rgb_outputs

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



