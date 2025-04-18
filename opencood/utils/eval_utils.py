# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils

from pdb import set_trace as pause

def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1) # false positive
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1) # true positive

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]['score'] += det_score.tolist()
    
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt

def caluclate_tp_fp_tn_fn(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    
    matched_true_boxes = set()

    TP=0
    FP=0
    FN=0
    TN=0

    gt = gt_boxes.shape[0]

    pred = 0
    if det_boxes is not None:
        pred = det_boxes.shape[0]
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box, in confidence descending order
        
        if score_order_descend.shape[0]==0:
           print(score_order_descend)
        else:
            for i in range(score_order_descend.shape[0]):
                det_polygon = det_polygon_list[score_order_descend[i]]
                ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

                # assert ious.size != 0

                if ious.size == 0:
                    continue

                best_iou = np.max(ious) 
                best_true_idx = np.argmax(ious)

                if best_iou >= iou_thresh:
                    if best_true_idx not in matched_true_boxes:
                        TP += 1
                        matched_true_boxes.add(best_true_idx)
                    else:
                        TP += 1
                else:
                    FP += 1
                
            # Calculate FN as unmatched true boxes
            FN = gt - len(matched_true_boxes)

    # print(gt,TP,FN,'|',det_boxes.shape[0],FP,TN)
    # print('false pred', FP/det_boxes.shape[0], 'missing pred', FN/gt)
    # pause()
    result_stat[iou_thresh]['gt'] += gt
    result_stat[iou_thresh]['pred'] += pred
    result_stat[iou_thresh]['fp'] += FP
    result_stat[iou_thresh]['tp'] += TP
    result_stat[iou_thresh]['fn'] += FN



def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, infer_info=None):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    # dump_dict.update({'ap30': ap_30,
    #                   'ap_50': ap_50,
    #                   'ap_70': ap_70,
    #                   'mpre_50': mpre_50,
    #                   'mrec_50': mrec_50,
    #                   'mpre_70': mpre_70,
    #                   'mrec_70': mrec_70,
    #                   })
    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70
                      })
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    print('The Average Precision at IOU 0.3 is %.4f, '
          'The Average Precision at IOU 0.5 is %.4f, '
          'The Average Precision at IOU 0.7 is %.4f' % (ap_30, ap_50, ap_70))

    return ap_30, ap_50, ap_70