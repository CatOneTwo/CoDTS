import os
from collections import OrderedDict
from signal import pause

import numpy as np
from torch.utils.data import Dataset

import random
import json
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix

from opencood.utils import box_utils,common_utils
from random import shuffle

from pdb import set_trace as pause

class DAIRV2XBaseDataset(Dataset):
    def __init__(self, root_dir, data_dir, save_dir):
    

        self.root_dir = data_dir
        self.split_info = read_json(root_dir) # train 数据集

        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

        self.ori_label_num = 0
        self.sparse_label_num = 0

        self.reinitialize()
    
    def reinitialize(self):
        
        pause()
        for idx in range(len(self.split_info)):
            veh_frame_id = self.split_info[idx]
            frame_info = self.co_data[veh_frame_id]
            system_error_offset = frame_info["system_error_offset"]

        
            # pose of agent 
            # 车辆端
            lidar_to_novatel = read_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
            novatel_to_world = read_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
            transformation_matrix_veh = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
            vehicle_lidar_pose = tfm_to_pose(transformation_matrix_veh) # 车端坐标系
            
            # 路端
            inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
            virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
            transformation_matrix_inf = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
            infra_lidar_pose = tfm_to_pose(transformation_matrix_inf) # 路端坐标系

            
            # lidar Label
            
            # vehicle_label_ori = read_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar_ori/{}.json'.format(veh_frame_id))) 
            # vehicle_label = read_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id))) # TODO 车辆点云范围内的目标，需要生成稀疏的标注文件(车端坐标系下)
            veh_label_path = os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id))
            veh_sparse_label_path = os.path.join(self.root_dir, 'vehicle-side/label/lidar_sparse/{}.json'.format(veh_frame_id))
            output_path = os.path.join(self.root_dir, 'vehicle-side/label/lidar_sparse')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            veh_label_world, veh_boxes = self.sparse_label_single(veh_label_path, veh_sparse_label_path, transformation_matrix_veh)

            # infra_label = read_json(os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id))) # TODO 路段点云范围内的目标，需要生成稀疏的标注文件，(路端坐标系下)
            infra_label_path = os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id))
            infra_sparse_label_path = os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar_sparse/{}.json'.format(inf_frame_id))
            output_path = os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar_sparse')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            inf_label_world, inf_boxes = self.sparse_label_single(infra_label_path, infra_sparse_label_path, transformation_matrix_inf)
            

            # collab_label_ori = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'].replace("label_world", "label_world_ori"))) # 原来的协同标注文件
            # collab_label = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'])) # TODO 车辆坐标下的协同标注，需要生成稀疏的标注，(世界坐标系下)
            
            collab_sparse_label_path = os.path.join(self.root_dir,frame_info['cooperative_label_path']).replace('label_world','label_world_sparse')
            output_path = os.path.join(self.root_dir, 'cooperative/label_world_sparse')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            co_label = self.sparse_label_collab(veh_label_world,veh_boxes, inf_label_world, inf_boxes, collab_sparse_label_path)
            print(len(co_label))

        print(len(self.split_info)) # 4811 训练集帧数
        print(self.ori_label_num) # 237725
        print(self.sparse_label_num) # 9622
        print(self.sparse_label_num/self.ori_label_num) # 0.04047533915238195


    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        pass
    

    def sparse_label_single(self, labels_path, labels_sparse_path, transformation_matrix):
        # 生成路端或车端的稀疏标签
        labels = read_json(labels_path)
        labels_sparse = []

        self.ori_label_num += len(labels)
        self.sparse_label_num += 1

        # 保证筛选的单标签在智能体的点云范围内，且类别是车辆
        lidar_range = [-100.8, -40, -3.5, 100.8, 40, 1.5]
        tmp_object_list = labels #世界坐标系下的目标
        output_dict = {}
        box_utils.load_single_objects_dairv2x(tmp_object_list,
                                        output_dict,
                                        lidar_range,
                                        'hwl')

        # 对于每一帧数据，随机保留一个label
        object_id_list = list(output_dict.keys())
        shuffle(object_id_list)
        

        if len(object_id_list) > 0: 
            object_ids_keep = object_id_list[0] # 筛选的车辆编号
            object_keep_label = labels[object_ids_keep]
               
        else:
            object_keep_label = labels[0]
        
        # 保存单侧的稀疏标签
        labels_sparse.append(object_keep_label)
        with open(labels_sparse_path, 'w') as f:
            json.dump(labels_sparse, f)

        # 转换至世界坐标系下
        box = object_keep_label
        x = box['3d_location']['x']
        y = box['3d_location']['y']
        z = box['3d_location']['z']
        h = box['3d_dimensions']['h']
        w = box['3d_dimensions']['w']
        l = box['3d_dimensions']['l']
        rotation = box['rotation']
        bbx_lidar = [x,y,z,h,w,l,rotation]
        bbx_lidar = np.array(bbx_lidar).reshape(1,-1) # [1, 7]
        bbx_lidar = box_utils.boxes_to_corners_3d(bbx_lidar,'hwl') #[1, 8, 3]
        bbx_lidar = box_utils.project_box3d(bbx_lidar, transformation_matrix) #[1, 8, 3]
        box_3d = box_utils.corner_to_center(bbx_lidar)
        label = {}
        label = object_keep_label
        label["3d_dimensions"] = {"h": float(box_3d[0,5]),"w": float(box_3d[0,4]),"l": float(box_3d[0,3])}
        label["3d_location"] = {"x": float(box_3d[0,0]),"y": float(box_3d[0,1]),"z": float(box_3d[0,2])}
        label['rotation'] = float(box_3d[0,6])
        for i in range(0,8):
            for j in range(0,3):
                bbx_lidar[0,i,j] = float(bbx_lidar[0,i,j])
        label["world_8_points"] = bbx_lidar[0].tolist()


        boxes = np.empty((1, 8, 3))
        boxes[0] = bbx_lidar[0]

        return label, boxes

    def sparse_label_collab(self, veh_label_world, veh_boxes, inf_label_world, inf_boxes, collab_label_path):
        
        co_label = []
        co_label.append(veh_label_world)

        inf_id_redundant = []
        inf_polygon_list = list(common_utils.convert_format(inf_boxes))
        veh_polygon_list = list(common_utils.convert_format(veh_boxes))
        iou_thresh = 0.05 
        # match prediction and gt bounding box
        for i in range(len(inf_polygon_list)):
            inf_polygon = inf_polygon_list[i]
            ious = common_utils.compute_iou(inf_polygon, veh_polygon_list)
            if (ious > iou_thresh).any():
                inf_id_redundant.append(i)
        # 与车辆端端标签不重合则加入协同标签
        if len(inf_id_redundant)==0:
            co_label.append(inf_label_world)
        
        with open(collab_label_path, 'w') as f:
            json.dump(co_label, f)
        
        return co_label

if __name__ == '__main__':
    data_dir= "dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
    root_dir= "dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/train.json"
    save_dir= "dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/train_plus.json"

    v2xsim_train = DAIRV2XBaseDataset(root_dir, data_dir, save_dir)    