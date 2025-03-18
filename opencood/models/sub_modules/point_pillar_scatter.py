from signal import pause
import torch
import torch.nn as nn

from pdb import set_trace as pause

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # [704, 200, 1] 

        assert self.nz == 1

    def forward(self, batch_dict):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
            pillars = pillars.t() # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
            batch_spatial_features.append(spatial_feature) 


        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict



# pillar-mae
class PointPillarScatterMAE(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features'] # 64
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # [704, 200, 1] 
        self.max_points_per_voxel = model_cfg['max_points_per_voxel'] # 32

        assert self.nz == 1
        self.mask_ratio = model_cfg['mask_ratio'] # [0.9,0.5]

    def forward(self, batch_dict):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        # 同时完成bev-mae的功能
        # 1. 记录BEV维度上的GT pillar occpancy(哪个位置pillar非空)，以及pillar内点密度
        # 2. 在BEV的基础上实现mask 

        pillar_features, coords, voxel_num_points  = batch_dict['pillar_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        batch_pillar_occupancy = []
        batch_pillar_points_density = []
        batch_pillar_points_number = []
        
        if 'select_index' in batch_dict.keys():
            select_index = batch_dict['select_index'] # teacher网络直接读取mask后的pillar
        else:
            # 随机对pillar进行mask操作
            select_index = self.mask_pillars(coords) # student网络随机mask
            batch_dict['select_index'] = select_index 
        
        pillar_features_partial, pillar_coords_partial = pillar_features[select_index,:], coords[select_index,:] # mask之后的特征

        for batch_idx in range(batch_size):

            # MAE Ground Truth generation
            # empty or non-empty pillars
            pillar_occupancy = torch.zeros(
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            
            # point density in non-empty pillars
            pillar_points_density = torch.zeros(
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            
            # point number in non-empty pillars
            pillar_points_number = torch.zeros(
                self.nz * self.nx * self.ny,
                dtype=voxel_num_points.dtype,
                device=pillar_features.device)

            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,704], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
    
            # 1. MAE GT genetate --start--
            # 在索引位置标注non-empty pillar
            pillar_occupancy[indices]=1.0
            # 将pillar信息加入list
            batch_pillar_occupancy.append(pillar_occupancy)
            # 在索引位置标注点数
            points_num = voxel_num_points[batch_mask] # (batch_idx_voxel)
            pillar_points_number[indices] = points_num
            batch_pillar_points_number.append(pillar_points_number)
            # 在索引位置标注点密度
            points_density = points_num/self.max_points_per_voxel
            pillar_points_density[indices] = points_density
            batch_pillar_points_density.append(pillar_points_density)

            # 1. MAE GT genetate --end--

            # 2. MAE mask voxel --start--
            spatial_feature_partial = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            
            # batch_index的mask
            batch_partial_mask = pillar_coords_partial[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords_partial = pillar_coords_partial[batch_partial_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,704], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices_partial = this_coords_partial[:, 1] + this_coords_partial[:, 2] * self.nx + this_coords_partial[:, 3]
            # 转换数据类型
            indices_partial = indices_partial.type(torch.long)
            # 根据mask提取pillar_features
            pillars_partial = pillar_features_partial[batch_partial_mask, :] # (batch_idx_voxel,64)
            pillars_partial = pillars_partial.t() # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            spatial_feature_partial[:, indices_partial] = pillars_partial
            # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
            batch_spatial_features.append(spatial_feature_partial) 
            # 2. MAE mask voxel --end--


        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
        batch_dict['spatial_features'] = batch_spatial_features # [b, c, ny, nx]

        # MAE GT
        batch_pillar_occupancy = torch.stack(batch_pillar_occupancy, 0)
        batch_pillar_points_density = torch.stack(batch_pillar_points_density, 0)
        batch_pillar_points_number = torch.stack(batch_pillar_points_number, 0)
        
        batch_pillar_occupancy = batch_pillar_occupancy.view(batch_size, self.ny, self.nx)
        batch_pillar_points_density = batch_pillar_points_density.view(batch_size, self.ny, self.nx)
        batch_pillar_points_number = batch_pillar_points_number.view(batch_size, self.ny, self.nx)


        batch_dict['gt_pillar_occupancy'] = batch_pillar_occupancy # [b, ny, nx]
        batch_dict['gt_pillar_points_density'] = batch_pillar_points_density # [b, ny, nx]
        batch_dict['gt_pillar_point_number'] = batch_pillar_points_number # [b, ny, nx]
        
        return batch_dict
    
    def mask_pillars(self, voxel_coords):

        # batch_size: int
        # voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]

        mask_ratio = self.mask_ratio[0] # mask的比例 0.7
        mask = torch.rand(voxel_coords.size(0)) < mask_ratio
        unmasked = (~mask).nonzero().ravel()
        select_index = unmasked

        return select_index
    
    