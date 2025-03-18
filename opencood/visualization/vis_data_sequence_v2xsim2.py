# -*- coding: utf-8 -*-
# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>

import os
from torch.utils.data import DataLoader, Subset
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import simple_vis
from opencood.data_utils.datasets import build_dataset
import numpy as np
from pdb import set_trace as pause

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path, "../hypes_yaml/v2xsim/visualization.yaml"))
    # params = load_yaml(os.path.join(current_path, "../hypes_yaml/v2xsim/visualization_single.yaml"))
    # params = load_yaml(os.path.join(current_path, "../hypes_yaml/opv2v/visualization_opv2v.yaml"))
    # output_path = "data_vis/v2xsim"
    # output_path = "data_vis/opv2v"
    # output_path = "data_vis/v2xsim_sparse_all"
    output_path = "data_vis/v2xsim_sparse_co"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    opencda_dataset = build_dataset(params, visualize=True, train=True)
    len = len(opencda_dataset)
    sampled_indices = np.random.permutation(len)[200:300]
    subset = Subset(opencda_dataset, sampled_indices)
    # indices=[0,49,79,99,149,160,199,219,249,299,307,349,389,487,399,449,499,512,549,599,649,670,699,749,799,849,899]
    # subset = Subset(opencda_dataset, indices)
    
    # data_loader = DataLoader(subset, batch_size=1, num_workers=0,
    #                          collate_fn=opencda_dataset.collate_batch_test,
    #                          shuffle=False,
    #                          pin_memory=False)
    
    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=0,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)

    vis_gt_box = True # 可视化gt
    vis_pred_box = False # 可视化预测
    hypes = params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, batch_data in enumerate(data_loader):
        print(i)
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data)

        gt_dict = {}
        if vis_gt_box: # 是否可视化框
            gt_dict['gt_box_tensor'] = gt_box_tensor

        # vis_save_path = os.path.join(output_path, '3d_%05d.png' % i)
        # simple_vis.visualize(gt_dict,
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='3d',
        #                     left_hand=False)
            
        vis_save_path = os.path.join(output_path, 'bev_%05d.png' % i)
        simple_vis.visualize(gt_dict,
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=False)

    