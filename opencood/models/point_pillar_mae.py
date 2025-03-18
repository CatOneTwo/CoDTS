# -*- coding: utf-8 -*-
# Author: Yushan Han, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatterMAE
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone

from opencood.models.sub_modules.mask_autoencoder import LightDecoder

from pdb import set_trace as pause

class PointPillarMAE(nn.Module):
    def __init__(self, args):
        super(PointPillarMAE, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatterMAE(args['point_pillar_scatter'])
        
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger

        # Decoder
        self.decoder = LightDecoder(args['ssl_decoder'])
        self.out_channel = args['ssl_decoder']['output_dim']

        # reconstruction head
        self.occupancy_head = nn.Conv2d(self.out_channel, 1, kernel_size=1)
        self.density_head = nn.Conv2d(self.out_channel, 1, kernel_size=1)
        self.number_head = nn.Conv2d(self.out_channel, 1, kernel_size=1)
        
    def forward(self, data_dict):
        # 只生成 non empty voxels
        voxel_features = data_dict['processed_lidar']['voxel_features'] # [80608, 32, 4] (num_voxels, max_points_per_voxel, C)
        voxel_coords = data_dict['processed_lidar']['voxel_coords'] # [80608, 4] 每个pillar的位置 (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx] pillar的z_idx为0
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] # [80608] (num_voxels) 一个voxel最多32个points，non-empty voxel至少有一个point
        
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict) # pillar_features: [80608, 64] 每个非空pillar的特征 (num_voxels, new_C)
        
        # 生成GT(pillar内的点，pillar点密度(norm后的)，pillar occupancy（1,0）)
        batch_dict = self.scatter(batch_dict) # spatial_features: [8, 64, 200, 504] [8, 64, 160, 160] voxel的尺寸
        
        gt_pillar_occupancy = batch_dict['gt_pillar_occupancy']
        gt_pillar_points_density = batch_dict['gt_pillar_points_density']
        gt_pillar_points_number = batch_dict['gt_pillar_point_number']
        select_index = batch_dict['select_index'] # mask之后的pillar索引
        
        output_dict = {'gt_pillar_occupancy': gt_pillar_occupancy,
                       'gt_pillar_points_density': gt_pillar_points_density,
                       'gt_pillar_points_number': gt_pillar_points_number,
                       'select_index': select_index}

        batch_dict = self.backbone(batch_dict) # spatial_features_2d 尺寸 [bs, 384, 100, 252] [8, 384, 80, 80]

        spatial_features_2d = batch_dict['spatial_features_2d'] 

        # decoder [n, C1, H, W] -> [n, C2, 2*H, 2*W]
        spatial_features_2d = self.decoder(spatial_features_2d) # [8, 256, 200, 504] [8, 256, 80, 80]

        # reconstruction head
        pred_pillar_occupancy = self.occupancy_head(spatial_features_2d)
        pred_pillar_occupancy = torch.squeeze(pred_pillar_occupancy)

        pred_pillar_points_density = self.density_head(spatial_features_2d)
        pred_pillar_points_density = torch.squeeze(pred_pillar_points_density)

        pred_pillar_points_number = self.number_head(spatial_features_2d)
        pred_pillar_points_number = torch.squeeze(pred_pillar_points_number)

        output_dict.update({'pred_pillar_occupancy': pred_pillar_occupancy,
                            'pred_pillar_points_density': pred_pillar_points_density,
                            'pred_pillar_points_number': pred_pillar_points_number,
                            'encode_features': spatial_features_2d})
        
        return output_dict
    




