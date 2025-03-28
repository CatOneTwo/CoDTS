# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# a class that integrate multiple simple fusion methods (Single Scale)
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch
import torch.nn as nn
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


import torch.nn.functional as F

from pdb import set_trace as pause

class PointPillarBaseline(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarBaseline, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'when2comm':
            self.fusion_net = When2commFusion(args['when2comm'])

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2 
            
        # hys add
        self.supervise_single = False
        if 'supervise_single' in args.keys(): # 单侧的监督
            self.supervise_single = args['supervise_single']
            
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features'] # (num_voxels, 32, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords'] # (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len'] # 


        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords, 
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        

        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict) # sptial_features [b, 64, 200, 504] voxel 尺寸
        
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])
        batch_dict = self.backbone(batch_dict)        
        
        spatial_features_2d = batch_dict['spatial_features_2d'] # [b, 384, 100, 252]

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d) # [b, 256, 100, 252] channel维度压缩

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        fused_feature = self.fusion_net(spatial_features_2d, record_len, t_matrix) # [b_ego, 256, 100, 252]

        psm = self.cls_head(fused_feature) # [b_ego, 2, 100, 252]
        rm = self.reg_head(fused_feature) # [b_ego, 14, 100, 252]

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)}) # [b_ego, 4, 100, 252]


        # 有单侧监督,投影到ego视角后的单侧预测
        if self.supervise_single:
            psm_single = self.cls_head(spatial_features_2d) # TODO 用于生成粗粒度的heatmap
            rm_single = self.reg_head(spatial_features_2d)
            if self.use_dir:
                dir_single = self.dir_head(spatial_features_2d)

            output_dict.update({'cls_preds_single': psm_single, 
                                'reg_preds_single': rm_single})
            if self.use_dir:
                output_dict.update({'dir_preds_single': dir_single})     

        return output_dict
    
    