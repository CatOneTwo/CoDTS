# https://github.com/yifanlu0227/CoAlign/issues/2
# 生成V2X-Sim 2.0 train val test pickle文件的脚本
# 只需要lidar.zip
# 需要去掉v1.0mini内的lidarseg.json

from typing import OrderedDict
from nuscenes.nuscenes import NuScenes
from icecream import ic
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path
import pickle
import os 
from tqdm import tqdm
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, tfm_to_pose
from opencood.utils.box_utils import corner_to_center, mask_boxes_outside_range_numpy, create_bbx, get_points_in_rotated_box_3d

from pdb import set_trace as pause

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def make_split_2(nusc, train_n=80, val_n=10, test_n=10):
    """ make split of trainset, valset, testset
        for v2x-sim 2.0
    """
    scene_num = len(nusc.scene)
    assert (train_n + val_n + test_n) == scene_num
    
    # np.random.seed(234)
    # perm = np.random.permutation(scene_num)
    perm = list(range(100))

    train_split = perm[:train_n]
    val_split = perm[train_n:train_n+val_n]
    test_split = perm[train_n+val_n:train_n+val_n+test_n]

    ic(train_split)
    ic(val_split)
    ic(test_split)

    return train_split, val_split, test_split

def make_split_2_new(nusc, train_n=80, val_n=10, test_n=10):
    """ make split of trainset, valset, testset
        for v2x-sim 2.0
        refer to: https://github.com/coperception/coperception/issues/7
    """
    scene_num = len(nusc.scene)
    assert (train_n + val_n + test_n) == scene_num
    
    # np.random.seed(234)
    # perm = np.random.permutation(scene_num)
    perm = list(range(100))

    train_split = [82,25,95,0,2,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,26,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,64,66,67,69,70,71,72,73,74,75,77,80,81,83,85,86,87,88,89,90,93,94,98,99]
    val_split = [1,3,4,63,65,68,76,78,79,84]
    test_split = [5,8,19,27,28,29,91,92,96,97]

    ic(train_split)
    ic(val_split)
    ic(test_split)

    return train_split, val_split, test_split

def build_hash_map(nusc):

    instance_set = set()

    for sample in nusc.sample:
        anns = sample['anns']
        instance_token = [nusc.get("sample_annotation",anno_token)['instance_token'] for anno_token in anns]

        instance_set.update(instance_token)
    
    hash_map = dict()

    for idx, token in enumerate(instance_set):
        hash_map[token] = idx
    
    return hash_map


def fill_split_info(nusc, scene_ids, save_filename=None):
    """ use the v2x-sim dataset in nuscenes format, and aggregate the point cloud file path,
        ego pose, gt boxes information. For one split(train/val/test).

        only support single sweep now.

    Args:
        nusc: nuscenes dataset object
        scene_ids: list

    """
    split_infos = []
    if save_filename:
        print(save_filename)

    for scene_id in tqdm(scene_ids):
        scene = nusc.scene[scene_id]
        sample_token = scene['first_sample_token']
        while(sample_token != ''):
            sample = nusc.get('sample', sample_token)  # dict
            agent_num = eval(max([i[-1] for i in sample['data'].keys() if i.startswith("LIDAR_TOP")])) 
            # LIDAR_TOP_id_0 is not vehicle. It's roadside unit.

            info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'agent_num': agent_num, 
            }


            """
            Updated by Yifan Lu, 2022.9.21
            1. no shuffle agent's order.
            2. filter boxes without lidar point hit.
            3. every agent will find it's gt boxes.
            """

            # shuffle_order = np.random.permutation(agent_num) 
            shuffle_order = list(range(1, agent_num+1))

            for i in range(agent_num):
                idx_in_nuscenes = shuffle_order[i]
                idx_in_info = i+1

                lidar_sample_data =  nusc.get("sample_data", sample['data'][f'LIDAR_TOP_id_{idx_in_nuscenes}']) # dict

                info[f'lidar_path_{idx_in_info}'] = nusc.get_sample_data_path(lidar_sample_data['token'])

                # dict, include token, timestamp, rotation, translation. rotation is Quaternion
                ego_pose_record = nusc.get("ego_pose", lidar_sample_data['ego_pose_token'])
                q = Quaternion(ego_pose_record['rotation'])
                T_world_ego = q.transformation_matrix
                T_world_ego[:3,3] = ego_pose_record['translation']

                # info[f'ego_pose_{shuffle_idx}'] = T_world_ego
                
                # dict, include token, timestamp, rotation, translation. rotation is Quaternion
                cs_record = nusc.get("calibrated_sensor", lidar_sample_data['calibrated_sensor_token'])
                translation_ego_lidar = cs_record['translation']
                rotation_ego_lidar = cs_record['rotation']
                q = Quaternion(rotation_ego_lidar)
                T_ego_lidar = q.transformation_matrix
                T_ego_lidar[:3,3] = translation_ego_lidar

                T_world_lidar = np.dot(T_world_ego, T_ego_lidar)
                info[f'lidar_pose_{idx_in_info}'] = T_world_lidar


                # id 0 exists for sure, it will fetch all annotations in the sample
                # they are shared for all agents

                # boxes are in global coordinate
                boxes = nusc.get_boxes(sample['data']['LIDAR_TOP_id_1'])

                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
                rots = np.array([b.orientation.elements for b in boxes]).reshape(-1, 4)
                names = np.array([b.name for b in boxes]) # like "vehicle.audi.tt"
                tokens = [b.token for b in boxes]
                object_ids = [hash_map[nusc.get("sample_annotation", anno_token)['instance_token']] for anno_token in tokens]
                tokens = np.array(tokens)
                object_ids = np.array(object_ids)

                gt_boxes = np.concatenate([locs, dims, rots], axis=1)  # [N, 10]

                # filter1: vehicle type
                # filter2: size
                vehicle_mask1 = [name.startswith("vehicle") for name in names]
                vehicle_mask2 = (dims[:,1] > 1.5).tolist() # filter small vehicle, width smaller than 1.5m.  Are they really vehicles?
                vehicle_mask = np.array([i and j for (i, j) in zip(vehicle_mask1,vehicle_mask2)], dtype=bool)

                names = names[vehicle_mask]
                tokens = tokens[vehicle_mask]
                gt_boxes = gt_boxes[vehicle_mask]
                object_ids = object_ids[vehicle_mask]

                # filter3: box in range
                # filter4: box with lidar point hit.

                # load the corresponding data 
                nbr_dims = 4 # x,y,z,intensity
                scan = np.fromfile(info[f'lidar_path_{idx_in_info}'], dtype='float32')
                lidar_np = scan.reshape((-1, 5))[:, :nbr_dims]  # [N, 4], in ego coord.
                lidar_range = [-90, -90, -3, 90, 90, 2]
                box_mask = []

                for (i, object_content) in enumerate(gt_boxes):
                    x,y,z,dx,dy,dz,w,a,b,c = object_content

                    q = Quaternion([w,a,b,c])
                    T_world_object = q.transformation_matrix
                    T_world_object[:3,3] = object_content[:3]

                    object2lidar = np.linalg.solve(T_world_lidar, T_world_object) # T_lidar_object


                    # shape (3, 8). 
                    # or we can use the create_bbx funcion.
                    x_corners = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) # (8,)
                    y_corners = dy / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])
                    z_corners = dz / 2 * np.array([-1, -1, -1, -1,  1,  1,  1,  1])

                    bbx = np.vstack((x_corners, y_corners, z_corners)) # (3, 8)

                    # bounding box under ego coordinate shape (4, 8)
                    bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

                    # project the 8 corners to world coordinate
                    bbx_lidar = np.dot(object2lidar, bbx).T # (8, 4)
                    bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # (1, 8, 3)
                    bbx_lidar = corner_to_center(bbx_lidar, order='hwl')
                    bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                            lidar_range,
                                                            'hwl')

                    if bbx_lidar.shape[0] == 0:
                        box_mask.append(0)
                        continue

                    enlarge = 0.2 # 0.2 m
                    enlarge_extent = [dx/2 + enlarge, dy/2 + enlarge, dz/2 + enlarge]
                    enlarge_bbx = create_bbx(enlarge_extent).T
                    # bounding box under ego coordinate shape (4, 8)
                    enlarge_bbx = np.r_[enlarge_bbx, [np.ones(enlarge_bbx.shape[1])]]
                    enlarge_bbx_lidar = np.dot(object2lidar, enlarge_bbx).T  # (8, 4)
                    enlarge_bbx_lidar = enlarge_bbx_lidar[:, :3] # (8, 3)

                    points_in_bbx = get_points_in_rotated_box_3d(lidar_np[:,:3], enlarge_bbx_lidar)
                    if points_in_bbx.shape[0] == 0:
                        box_mask.append(0)
                    else:
                        box_mask.append(1)

                box_mask = np.array(box_mask, dtype=np.bool)

                info[f'labels_{idx_in_info}'] = OrderedDict()
                info[f'labels_{idx_in_info}']['gt_names'] = names[box_mask]
                info[f'labels_{idx_in_info}']['gt_boxes_token'] = tokens[box_mask]
                info[f'labels_{idx_in_info}']['gt_boxes_global'] = gt_boxes[box_mask]
                info[f'labels_{idx_in_info}']['gt_object_ids'] = object_ids[box_mask]

                info["sample_token"] = sample_token

                # also store the frame token

            split_infos.append(info)
                
            sample_token = sample['next']

    if save_filename is not None:
        with open(save_filename, 'wb') as f:
            pickle.dump(split_infos, f)
    else:
        return split_infos

def fill_infos(nusc, train_split, val_split, test_split):
    """ use the v2x-sim dataset in nuscenes format, and aggregate the point cloud file path,
    ego pose, gt boxes information.

    Args:
        data_path: str, root path for nusc dataset
        nusc: nuscenes dataset object
        *_split: list, contains index of scene
    """

    scene_num = len(nusc.scene)

    train_infos = fill_split_info(nusc, train_split)
    val_infos = fill_split_info(nusc, val_split)
    test_infos = fill_split_info(nusc, test_split)

    return train_infos, val_infos, test_infos


def create_pkl(nusc, save_dir):
    """ create info.pkl for train/val/test set

    Args: 
        nusc: nuscenes dataset object
        save_dir: directory for saving pkl.

    Returns: None
    """
    train_split, val_split, test_split = make_split_2_new(nusc)

    # if MULTI_PROCESSES:
    #     fill_infos_multiprocess(nusc, train_split, val_split, test_split, save_dir)
    #     return

    
    train_infos, val_infos, test_infos = fill_infos(nusc, train_split, val_split, test_split)

    print(f'train sample: {len(train_infos)}, val sample: {len(val_infos)}, test sample: {len(test_infos)}')

    with open(os.path.join(save_dir, f'v2xsim_infos_train.pkl'), 'wb') as f:
        pickle.dump(train_infos, f)
    with open(os.path.join(save_dir, f'v2xsim_infos_val.pkl'), 'wb') as f:
        pickle.dump(val_infos, f)
    with open(os.path.join(save_dir, f'v2xsim_infos_test.pkl'), 'wb') as f:
        pickle.dump(test_infos, f)



if __name__=="__main__":

    nusc = NuScenes(version='v1.0-mini', dataroot='/home/hys/datasets/V2X-Sim', verbose=True) # specify your own path.
    output_path = "/home/hys/datasets/V2X-Sim" # a folder path. specify your own path.

    global hash_map
    hash_map = build_hash_map(nusc)

    create_pkl(nusc, output_path)