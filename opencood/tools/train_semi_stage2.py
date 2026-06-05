# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Mean Teacher

import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import glob

import tqdm

import random
import numpy as np
import copy

from opencood.loss.pseudo_label_generation import generate_pseudo_label_ego

from pdb import set_trace as pause

def seed_torch(seed=1029):
    print('seed fixed!')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='self supervised pretrained model')
    parser.add_argument('--semi_ratio', type=float, default=0.1,
                        help='labeld ratio of semi-supervised learning')
    parser.add_argument('--ema_decay',  type=float,default='0.999',
                        help='the parameters of teacher model update')                
    parser.add_argument('--log', type=str, default='',
                        help='log name suffix')
    opt = parser.parse_args()
    return opt


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

# def get_current_consistency_weight(epoch):
#     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def main():
    seed_torch()
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes['log_suffix']=opt.log
    
    # 根据semi_ratio选择相应的split文件
    semi_ratio = opt.semi_ratio
    hypes['root_dir'] = hypes['root_dir'].replace('train','train_labeled_{}'.format(semi_ratio)) 

    hypes['train_params']['epoches'] = 25
    hypes['optimizer']['lr'] = 0.0002

    print('Dataset Building')
    opencood_train_dataset_labeled = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader_labeled = DataLoader(opencood_train_dataset_labeled,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4, 
                              collate_fn=opencood_train_dataset_labeled.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)

    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4, 
                            collate_fn=opencood_train_dataset_labeled.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)
    
    # unlabeled
    hypes['root_dir'] = hypes['root_dir'].replace('labeled','unlabeled')
    opencood_train_dataset_unlabeled = build_dataset(hypes, visualize=False, train=True)
    train_loader_unlabeled = DataLoader(opencood_train_dataset_unlabeled,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4, 
                              collate_fn=opencood_train_dataset_unlabeled.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)


    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    lowest_teacher_val_loss = 1e10
    lowest_teacher_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

        # load pretrainde ssl model if it exists.
        if opt.pretrained_model:
            model = train_utils.load_pretrained_model(opt.pretrained_model, model)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)


    # EMA mean teacher
    ema_teacher_model = copy.deepcopy(model)
    print('ema_teacher_model is a deep copy of the model')
    for p in ema_teacher_model.parameters():
        # p.requires_grad(False)
        p.detach_()
    if torch.cuda.is_available():
        ema_teacher_model.to(device)
    ema_teacher_model.eval()

    global_step = 0


    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset_labeled, "supervise_single") else opencood_train_dataset_labeled.supervise_single
    # used to help schedule learning rate
    
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(train_loader_labeled), leave=True)
        for i, batch_data in enumerate(train_loader_labeled):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            
            if epoch > 15: # 15
                ########## 1. 加载未标记数据 ##########
                batch_data_unlabeled = next(iter(train_loader_unlabeled))

                batch_data_unlabeled['ego']['label_dict']['targets'] = torch.zeros_like(batch_data_unlabeled['ego']['label_dict']['targets'])
                batch_data_unlabeled['ego']['label_dict']['pos_equal_one'] = torch.zeros_like(batch_data_unlabeled['ego']['label_dict']['pos_equal_one'])
                batch_data_unlabeled['ego']['label_dict']['neg_equal_one'] = torch.zeros_like(batch_data_unlabeled['ego']['label_dict']['neg_equal_one'])

                ########## 2. 将标记和未标记数据拼接 ##########
                record_len = batch_data['ego']['record_len'] # [2,2,2,2]
                pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix'] # [4, 5, 5, 4, 4]
                voxel_features = batch_data['ego']['processed_lidar']['voxel_features'] # (num_voxels, 32, 4)
                voxel_coords = batch_data['ego']['processed_lidar']['voxel_coords'] # (num_voxels, 4) [batch_idx, z_idx, y_idx, x_idx]
                voxel_num_points = batch_data['ego']['processed_lidar']['voxel_num_points'] # (num_voxels)

                record_len_unlabeled = batch_data_unlabeled['ego']['record_len'] # [2,2,2,2]
                pairwise_t_matrix_unlabeled = batch_data_unlabeled['ego']['pairwise_t_matrix'] # [4, 5, 5, 4, 4]
                voxel_features_unlabeled = batch_data_unlabeled['ego']['processed_lidar']['voxel_features']
                voxel_coords_unlabeled = batch_data_unlabeled['ego']['processed_lidar']['voxel_coords']
                voxel_num_points_unlabeled = batch_data_unlabeled['ego']['processed_lidar']['voxel_num_points'] # (num_voxels)
                
                record_len_new = torch.cat((record_len,record_len_unlabeled),dim=0).to(device)
                pairwise_t_matrix_new = torch.cat((pairwise_t_matrix,pairwise_t_matrix_unlabeled),dim=0).to(device)
                voxel_features_new = torch.cat((voxel_features,voxel_features_unlabeled),dim=0).to(device)
                voxel_coords_unlabeled_= copy.deepcopy(voxel_coords_unlabeled)# 第一维维batch id
                voxel_coords_unlabeled_[:,0] += torch.sum(record_len)# 第一维维batch id
                voxel_coords_new = torch.cat((voxel_coords,voxel_coords_unlabeled_),dim=0).to(device)
                voxel_num_points_new = torch.cat((voxel_num_points,voxel_num_points_unlabeled),dim=0).to(device)
                
                batch_data['ego']['record_len'] = record_len_new
                batch_data['ego']['pairwise_t_matrix'] = pairwise_t_matrix_new
                batch_data['ego']['processed_lidar']['voxel_features'] = voxel_features_new
                batch_data['ego']['processed_lidar']['voxel_coords'] = voxel_coords_new
                batch_data['ego']['processed_lidar']['voxel_num_points'] = voxel_num_points_new
                ########## 2. 将标记和未标记数据拼接 ##########

                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                batch_data = train_utils.to_device(batch_data, device)
                batch_data['ego']['epoch'] = epoch
                ouput_dict = model(batch_data['ego'])
                
                ########## 3. 使用EMA teacher进行预测并生成伪标签  ##########
                batch_data_unlabeled = train_utils.to_device(batch_data_unlabeled, device)
                teacher_ouput_dict_unlabeled = ema_teacher_model(batch_data_unlabeled['ego'])
                
                cls_preds_concat = ouput_dict['cls_preds'] 
                reg_preds_concat = ouput_dict['reg_preds'] 
                dir_preds_concat = ouput_dict['dir_preds']

                ouput_dict['cls_preds'] = cls_preds_concat[:4]
                ouput_dict['reg_preds'] = reg_preds_concat[:4]
                ouput_dict['dir_preds'] = dir_preds_concat[:4]
            
                # unlabeled
                ouput_dict['teacher_cls_preds'] = teacher_ouput_dict_unlabeled['cls_preds']
                ouput_dict['student_cls_preds'] = cls_preds_concat[4:] 
            else:
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                batch_data = train_utils.to_device(batch_data, device)
                batch_data['ego']['epoch'] = epoch
                ouput_dict = model(batch_data['ego'])

            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            
            criterion.logging(epoch, i, len(train_loader_labeled), writer,pbar=pbar2)
            
            if supervise_single_flag:
                final_loss += 1* criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader_labeled), writer, suffix="_single", pbar=pbar2)

            pbar2.update(1)

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            ################# update_ema_variables correct ##################
            # Use the true average until the exponential average is more correct
            # https://zhuanlan.zhihu.com/p/511761402
            alpha = min(1 - 1 / (global_step + 1), opt.ema_decay) # ema-decay=0.999
            with torch.no_grad():
                model_state_dict = model.state_dict()
                ema_model_state_dict = ema_teacher_model.state_dict()
                for entry in ema_model_state_dict.keys():
                    ema_param = ema_model_state_dict[entry].clone().detach()
                    param = model_state_dict[entry].clone().detach()
                    new_param = (ema_param * alpha) + (param * (1. - alpha))
                    ema_model_state_dict[entry] = new_param
                ema_teacher_model.load_state_dict(ema_model_state_dict)
            global_step += 1
            ################# update_ema_variables correct ##################

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            teacher_valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    # final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'],test=True) # 测试对比损失才需要
                    valid_ave_loss.append(final_loss.item())

                    # validate ema teacher model 
                    ema_teacher_model.zero_grad()
                    ema_teacher_model.eval()
                    teacher_ouput_dict = ema_teacher_model(batch_data['ego'])

                    teacher_final_loss = criterion(teacher_ouput_dict, batch_data['ego']['label_dict'])
                    teacher_valid_ave_loss.append(teacher_final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f \n' % (epoch, valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            teacher_valid_ave_loss = statistics.mean(teacher_valid_ave_loss)
            print('At epoch %d, the teacher validation loss is %f \n' % (epoch, teacher_valid_ave_loss))
            writer.add_scalar('Teacher_Validate_Loss', teacher_valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

            if teacher_valid_ave_loss < lowest_teacher_val_loss:
                lowest_teacher_val_loss = teacher_valid_ave_loss
                torch.save(ema_teacher_model.state_dict(), os.path.join(saved_path, 'ema_net_epoch_bestval_at%d.pth' % (epoch + 1)))
                                    
                if lowest_teacher_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'ema_net_epoch_bestval_at%d.pth' % (lowest_teacher_val_epoch))):
                    os.remove(os.path.join(saved_path, 'ema_net_epoch_bestval_at%d.pth' % (lowest_teacher_val_epoch)))
                lowest_teacher_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

        opencood_train_dataset_labeled.reinitialize() # 每个epoch需要shuffle每个场景内的车辆顺序
        opencood_train_dataset_unlabeled.reinitialize() # 每个epoch需要shuffle每个场景内的车辆顺序

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True    
    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        fusion_method = opt.fusion_method
        if 'noise_setting' in hypes and hypes['noise_setting']['add_noise']:
            cmd = f"CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_w_noise.py --model_dir {saved_path} --fusion_method {fusion_method}"
        else:
            cmd = f"CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
            cmd_ema = f"CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference_ema.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)
        print(f"Running command: {cmd_ema}")
        os.system(cmd_ema)

if __name__ == '__main__':
    main()
