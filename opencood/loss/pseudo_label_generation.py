from cmath import pi
import numpy as np
import torch
import torch.nn.functional as F
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils
from opencood.utils.common_utils import limit_period

from pdb import set_trace as pause


# SSC3OD: Sparsely Supervised Collaborative 3D Object Detection from LiDAR Point Clouds (2023 SMC)
def generate_pseudo_label_ego(target_dict, output_dict, anchors, limit_range, score_thresh=0.32):
        
    # 稀疏标签生成的label
    pos_equal_one = target_dict['pos_equal_one'] # [b, h, w, 2]
    positives = pos_equal_one > 0 # 正样本很少
    neg_equal_one = target_dict['neg_equal_one'] # [b, h, w, 2]
    negatives = neg_equal_one > 0 # 负样本很多
    reg_targets = target_dict['targets'] # [b, h, w, 14]

    # 协同端端预测
    reg_preds = output_dict['reg_preds'] # [b, 14, h, w]
    cls_preds = output_dict['cls_preds'] # [b, 2, h, w]

    B, _, H, W = cls_preds.shape

    # ego端预测
    foreground_maps_ego = cls_preds.sigmoid() # [b, 2, h,w]
    foreground_maps_ego *= (1-pos_equal_one).permute(0, 3, 1, 2) # 把已有的pos位置注释掉

    # 1. score-based filtering
    if score_thresh == 'hierarchy':
        score_thresh=0.3
        score_thresh2=0.25
        num_thre = 10
        foreground_pseudo_label = torch.where(foreground_maps_ego>score_thresh, 1, 0).type_as(pos_equal_one) # [b,2,h,w]
        for b in range(B):
            if foreground_pseudo_label[b].sum() < num_thre:
                foreground_pseudo_label[b] = torch.where(foreground_maps_ego[b]>score_thresh2,1,0)
    else:
        foreground_pseudo_label = torch.where(foreground_maps_ego>score_thresh, 1, 0).type_as(pos_equal_one) # [b,2,h,w]

    foreground_pseudo_label = foreground_pseudo_label.permute(0, 2, 3, 1) # [b,h,w,2]

    # 2. IoU-guided suppression
    # remove large bbx
    boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(reg_preds.permute(0, 2, 3, 1), anchors) # [b,h*w*2,7]
    boxes3d_pred = boxes3d_pred.view(-1,7) # [b*h*w*2,7]

    # adding dir classifier
    if 'dir_preds' in output_dict.keys():
        dir_offset = 0.7853
        num_bins = 2

        dm  = output_dict['dir_preds'] # [N, 4, H, W]
        dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
        
        period = (2 * np.pi / num_bins) # pi
        dir_rot = limit_period(boxes3d_pred[..., 6] - dir_offset, 0, period) # 限制在0到pi之间
        boxes3d_pred[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
        boxes3d_pred[..., 6] = limit_period(boxes3d_pred[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

    pred_box3d_tensor = box_utils.boxes_to_corners_3d(boxes3d_pred, order='hwl') # [b*h*w*2,8,3]
    keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor) # [b*h*w*2]
    keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor) # [b*h*w*2]
    keep_index = torch.logical_and(keep_index_1, keep_index_2) # [b*h*w*2]
    # 保持在lidar范围内
    _,keep_index3 = box_utils.mask_boxes_outside_range_numpy(pred_box3d_tensor.cpu().numpy(), limit_range, order=None,return_mask=True)
    keep_index = torch.logical_and(keep_index, torch.tensor(keep_index3).to(keep_index.device)) # [b*h*w*2]
    keep_index_ = (keep_index+0).view(foreground_pseudo_label.size()) # [b,h,w,2]
    foreground_pseudo_label *= keep_index_ # [b,h,w,2]


    # nms处理,去除重叠度大的预测框
    print('before nms',foreground_pseudo_label.sum().item())
    foreground = foreground_pseudo_label.reshape(B,-1) # [b, h*w*2]
    scores = cls_preds.reshape(B,-1) # [b, h*w*2]
    pred_box3d = pred_box3d_tensor.reshape(B,-1,8,3) # [b, h*w*2,8,3]

    for b in range(B):
        pseudo_index = foreground[b].nonzero()[:,0] # 已有的前景索引
        if pseudo_index.size(0) > 0:
            keep_list = box_utils.nms_rotated(pred_box3d[b,pseudo_index],scores[b,pseudo_index], 0.15).tolist() # 只返回索引
            ori_list = list(range(0, pseudo_index.size(0)))
            remove_list = list( set(ori_list).difference(set(keep_list)))
            remove_pseudo_index = pseudo_index[remove_list]
            foreground[b][remove_pseudo_index] = 0 

    foreground_pseudo_label = foreground.view(foreground_pseudo_label.size())       
    print('after nms',foreground_pseudo_label.sum().item())

    for b in range(B):
        map_ego = foreground_maps_ego[b] 
        print(b,(map_ego>score_thresh).sum().item(), pos_equal_one[b].sum().item(),foreground_pseudo_label[b].sum().item())

    pos_equal_one_pseudo = torch.where((pos_equal_one + foreground_pseudo_label)>=1, 1, 0) # 求并集
    positives_pseudo = pos_equal_one_pseudo > 0
    print('pos: ori:{}->pseudo:{}'.format(positives.sum().item(),positives_pseudo.sum().item()))

    neg_equal_one_pseudo = torch.where((neg_equal_one - foreground_pseudo_label)>=1, 1, 0) # 求交集
    negatives_pseudo = neg_equal_one_pseudo > 0
    print('neg: ori:{}->pseudo:{}'.format(negatives.sum().item(),negatives_pseudo.sum().item()))
    
    # 3. 生成新的targets
    pseudo_targets = reg_targets.detach().cpu() # [b,h,w,14] 原始的稀疏GT
    
    pseudo_gt = reg_preds.detach().cpu() # [b, 14, h, w]
    pseudo_ids = (pos_equal_one_pseudo - pos_equal_one).detach().cpu() # 挖掘出的伪正样本索引 # [b, h,w,2]
    pseudo_gt = pseudo_gt.permute(0,2,3,1) # [b,h,w,14]
    pseudo_gt = pseudo_gt.view(B,H,W,2,7) # [b,h,w,2,7]
    pseudo_ids = pseudo_ids.unsqueeze(-1) # [b,h,w,2,1]
    pseudo_gt = pseudo_ids * pseudo_gt # [[b,h,w,2,7]
    pseudo_targets += pseudo_gt.view(B,H,W,14) # 将伪正样本的回归值添加到原来的gt上    


    target_dict['pos_equal_one'] = pos_equal_one_pseudo.type_as(pos_equal_one)
    target_dict['neg_equal_one'] = neg_equal_one_pseudo.type_as(neg_equal_one)
    target_dict['targets'] = pseudo_targets.to(pos_equal_one.device)

    return target_dict


