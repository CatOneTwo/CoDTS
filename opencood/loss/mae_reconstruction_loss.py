# -*- coding: utf-8 -*-
# Author: Yushan Han

import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as pause

class MaeReconstructionLoss(nn.Module):
    def __init__(self, args):
        super(MaeReconstructionLoss, self).__init__()

        if 'occupancy_weight' in args:
            self.occupancy_weight = args['occupancy_weight']
        else:
            self.occupancy_weight = None
        
        if 'density_weight' in args:
            self.density_weight = args['density_weight']
        else:
            self.density_weight = None
        
        if 'number_weight' in args:
            self.number_weight = args['number_weight']
        else:
            self.number_weight = None
        
        if 'kd' in args:
            self.kd = args['kd']
        else:
            self.kd = None

        self.loss_dict = {}

    def forward(self, output_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        
        total_loss = 0

        if self.occupancy_weight:
            
            pred_pillar_occupancy = output_dict['pred_pillar_occupancy']
            gt_pillar_occupancy = output_dict['gt_pillar_occupancy']
            batch_size = pred_pillar_occupancy.shape[0]
            loss_occupied_src = F.binary_cross_entropy_with_logits(pred_pillar_occupancy, gt_pillar_occupancy)
            occupied_loss = loss_occupied_src.sum() / batch_size
            occupied_loss *= self.occupancy_weight

            self.loss_dict.update({'occupied_loss': occupied_loss.item()}) 

            total_loss += occupied_loss 

        if self.density_weight:
            pred_pillar_points_density = output_dict['pred_pillar_points_density']
            gt_pillar_points_density = output_dict['gt_pillar_points_density']
            batch_size = pred_pillar_points_density.shape[0]
            loss_density_src = F.smooth_l1_loss(pred_pillar_points_density, gt_pillar_points_density)
            density_loss = loss_density_src.sum() / batch_size
            density_loss *= self.density_weight

            self.loss_dict.update({'density_loss': density_loss.item()}) 
            
            total_loss += density_loss

        if self.number_weight:
            pred_pillar_points_number = output_dict['pred_pillar_points_number']
            gt_pillar_points_number = output_dict['gt_pillar_points_number']
            batch_size = pred_pillar_points_number.shape[0]
            loss_number_src = F.smooth_l1_loss(pred_pillar_points_number, gt_pillar_points_number)
            number_loss = loss_number_src.sum() / batch_size
            number_loss *= self.number_weight

            self.loss_dict.update({'number_loss': number_loss.item()}) 
            
            total_loss += number_loss


        # 知识蒸馏
        if self.kd:
            student_feature = output_dict['encode_features']
            
            teacher_feature = output_dict['teacher_encode_features']

            kd_loss = 0
            
            # 特征蒸馏
            N, C, H, W = teacher_feature.shape
            student_feature = student_feature.permute(0,2,3,1).reshape(N*H*W, C)
            teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
            kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)
            kd_loss_feature = kl_loss_mean(
                    F.log_softmax(student_feature, dim=1), 
                    F.softmax(teacher_feature, dim=1)
                )
            kd_loss += kd_loss_feature
            
            if self.kd.get('decoder_kd', False):
                # 预测蒸馏
                teacher_pred_pillar_occupancy = output_dict['teacher_pred_pillar_occupancy']
                N, H, W = teacher_pred_pillar_occupancy.shape
                student_pred = pred_pillar_occupancy.view(N*H*W, -1)
                teacher_pred = teacher_pred_pillar_occupancy.view(N*H*W, -1)
                
                kd_loss_pred = kl_loss_mean(
                        F.log_softmax(student_pred, dim=1), 
                        F.softmax(teacher_pred, dim=1)
                        )
                
                kd_loss += kd_loss_pred
            
            kd_loss *= self.kd['weight']
            
            self.loss_dict.update({'kd_loss': kd_loss.item()}) 
            
            total_loss += kd_loss

        self.loss_dict.update({'total_loss': total_loss.item()})
        

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        # occupied_loss = self.loss_dict.get('occupied_loss', 0)

        # msg = "[epoch %d][%d/%d] || Loss: %.4f || Occ Loss: %.4f" % (
        #         epoch, batch_id + 1, batch_len,
        #           total_loss, occupied_loss)
        
        msg = "[epoch %d][%d/%d] || Loss: %.4f" % (epoch, batch_id + 1, batch_len, total_loss)
        
        if self.kd:
            kd_loss = self.loss_dict.get('kd_loss', 0)
            msg += " || KD Loss: %.4f" % kd_loss

        if self.occupancy_weight:
            occupied_loss = self.loss_dict.get('occupied_loss', 0)
            msg += " || Occ Loss: %.4f" % occupied_loss

        if self.density_weight:
            density_loss = self.loss_dict.get('density_loss', 0)
            msg += " || Density Loss: %.4f" % density_loss
        
        if self.number_weight:
            number_loss = self.loss_dict.get('number_loss', 0)
            msg += " || Number Loss: %.4f" % number_loss
            
        if pbar is None:
            print(msg)
        else:
            pbar.set_description(msg)

        if not writer is None:
            if self.occupancy_weight:
                writer.add_scalar('Occupied_loss', occupied_loss, epoch*batch_len + batch_id)
            if self.kd:
                writer.add_scalar('KD_loss', kd_loss, epoch*batch_len + batch_id)
            if self.density_weight:
                writer.add_scalar('Density_loss', density_loss, epoch*batch_len + batch_id)
            if self.number_weight:
                writer.add_scalar('Number_loss', number_loss, epoch*batch_len + batch_id)

