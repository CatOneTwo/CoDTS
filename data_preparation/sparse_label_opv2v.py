
import os
from torch.utils.data import Dataset

import yaml
from opencood.hypes_yaml.yaml_utils import load_yaml

from opencood.utils import box_utils
from random import shuffle

from pdb import set_trace as pause


class OPV2VBaseDataset(Dataset):
    def __init__(self, root_dir, save_dir):

 
        self.root_dir = root_dir 
        
        print("Dataset dir:", root_dir)
        
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_folders = scenario_folders

        self.frame=0
        self.all_label_num = 0
        self.sparse_label_num = 0

        self.reinitialize()


    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):

            # at least 1 cav should show up

            cav_list = sorted([x for x in os.listdir(scenario_folder)
                                if os.path.isdir(os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]] # 路端观测放置到列表最后


            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                output_path = cav_path.replace('train','train_sparse')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                yaml_files = sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    
                    params = load_yaml(yaml_file)

                   
                    lidar_pose = params['lidar_pose']
                    vehicles = params['vehicles']

                    object_ids_keep = self.sparse_label(vehicles, lidar_pose)

                    vehicles_sparse = {}
                    if object_ids_keep != -1:
                        vehicles_sparse[object_ids_keep] = vehicles[object_ids_keep]
                    params['vehicles'] = vehicles_sparse

                    save_name = yaml_file.replace('train','train_sparse')
                    with open(save_name, 'w') as outfile:
                        yaml.dump(params, outfile, default_flow_style=False)

                    self.all_label_num += len(vehicles.keys())
                    self.sparse_label_num += 1

                if j == 0:    
                    self.frame+=len(timestamps)
            print('scenario:',scenario_folder)

        print(self.frame) # 6765 帧
        print(self.all_label_num) # 358142 个全标注 
        print(self.sparse_label_num) # 21139 个稀疏标注
        print(self.sparse_label_num/self.all_label_num) # 0.05902407424987854    

    def sparse_label(self, vehicles, lidar_pose):
        
        # 保证筛选的单标签在点云范围内
        tmp_object_dict = vehicles
        output_dict = {}  
        filter_range = [-140.8, -40, -3, 140.8, 40, 1]  
        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        lidar_pose,
                                        filter_range,
                                        'hwl')
        # 对于每一帧数据，随机保留一个label
        object_id_list = list(output_dict.keys())
        shuffle(object_id_list)

        if len(object_id_list) > 0: # 雷达范围内有目标
            object_ids_keep = object_id_list[0] # 筛选的车辆编号
               
        else:
            vehicle_ids = list(vehicles.keys()) # 存在有的ego周围没有车辆
            object_ids_keep = vehicle_ids[0] if len(vehicle_ids) >0  else -1
        
        return object_ids_keep

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps
                                 

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass
  

  
if __name__ == '__main__':
    train_root = "dataset/OPV2V/train"
    save_root = "dataset/OPV2V/train_sparse"
    opv2v_train = OPV2VBaseDataset(train_root, save_root)  


    
    
