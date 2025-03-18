# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>
import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
from torch.utils.data import Dataset
from opencood.utils.transformation_utils import tfm_to_pose


from opencood.utils import box_utils
from random import shuffle

from pdb import set_trace as pause

class V2XSIMBaseDataset(Dataset):
    """
        First version.
        Load V2X-sim 2.0 using yifan lu's pickle file. 
        Only support LiDAR data.
    """

    def __init__(self, root_dir, save_dir):

        self.root_dir = root_dir # 训练集路径

        print("Dataset dir:", root_dir)
        
        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        # TODO param: one as ego or all as ego?
        self.ego_mode = 'one'  # "all"

        self.reinitialize()

        with open(os.path.join(save_dir), 'wb') as f:
            pickle.dump(self.dataset_info_pkl, f)
        
        print("New Dataset dir:", save_dir)

    def reinitialize(self):
        self.scene_database = OrderedDict()
        if self.ego_mode == 'one':
            self.len_record = len(self.dataset_info_pkl)
        else:
            raise NotImplementedError(self.ego_mode)
        
        
        all_label_num = 0
        sparse_label_num = 0

        
        for i, scene_info in enumerate(self.dataset_info_pkl):
            self.scene_database.update({i: OrderedDict()})
            cav_num = scene_info['agent_num']
            assert cav_num > 0


            cav_ids = list(range(1, cav_num + 1))
            
            for j, cav_id in enumerate(cav_ids):
                
                # self.scene_database[i][cav_id]['params']['vehicles'] = scene_info[f'labels_{cav_id}']['gt_boxes_global'] # vehicle的边界框
                # self.scene_database[i][cav_id]['params']['object_ids'] = scene_info[f'labels_{cav_id}']['gt_object_ids'].tolist() # vehicle的编号
                

                vehicles = scene_info[f'labels_{cav_id}']['gt_boxes_global'] # vehicle的边界框
                object_ids = scene_info[f'labels_{cav_id}']['gt_object_ids'].tolist() # vehicle的编号
                lidar_pose = tfm_to_pose(scene_info[f"lidar_pose_{cav_id}"])  # [x, y, z, roll, pitch, yaw]
        
                
                # 在数据集中添加 single instance 标签
                vehicles_keep, object_ids_keep = self.sparse_label(vehicles, object_ids, lidar_pose)
                self.dataset_info_pkl[i][f'labels_{cav_id}']['gt_boxes_global_sparse'] = vehicles_keep
                self.dataset_info_pkl[i][f'labels_{cav_id}']['gt_object_ids_sparse'] = object_ids_keep

                all_label_num +=  len(object_ids)
                sparse_label_num += 1   

        print(self.len_record) # 8000 训练集帧数
        print(all_label_num) # 698991
        print(sparse_label_num) # 29300
        print(sparse_label_num/all_label_num) # 标注比例 0.0419
            

    def __len__(self) -> int:
        return self.len_record

    @abstractmethod
    def __getitem__(self, index):
        pass
    
    def sparse_label(self, vehicles, object_ids, lidar_pose):

        # 保证筛选的单标签在点云范围内
        lidar_range = [-32, -32, -3, 32, 32, 2]
        tmp_object_dict = {"gt_boxes": vehicles, "object_ids":object_ids}
        output_dict = {}
        box_utils.project_world_objects_v2x(tmp_object_dict,
                                        output_dict,
                                        lidar_pose,
                                        lidar_range,
                                        'hwl')

        # 对于每一帧数据，随机保留一个label
        object_id_list = list(output_dict.keys())
        shuffle(object_id_list)
        
        if len(object_id_list) > 0: 
            object_ids_keep = object_id_list[0] # 筛选的车辆编号
            ind = object_ids.index(object_ids_keep)
            object_bbx = vehicles[ind]    
        else:
            object_ids_keep = object_ids[0]
            object_bbx = vehicles[0]
        
        return np.array([object_bbx]), np.array([object_ids_keep])
        
        
        

    
if __name__ == '__main__':
    train_root = "dataset/v2xsim2_info/v2xsim_infos_train.pkl"
    save_root = "dataset/v2xsim2_info/v2xsim_infos_train_plus.pkl"
    v2xsim_train = V2XSIMBaseDataset(train_root, save_root)

