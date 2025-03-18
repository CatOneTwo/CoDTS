# CoDTS 版本
# dynamic teacher参考HSSDA使用dual threshold方法确定高阈值

import argparse
import os
import statistics
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
import tqdm

import random
import numpy as np
from opencood.utils import box_utils
import copy
import tqdm
from sklearn.cluster import KMeans
from pdb import set_trace as pause

# static teacher -> pseudo label, 同时生成ema_teacher

def seed_torch(seed=1029):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

seed_torch()

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='self supervised pretrained model') 
    parser.add_argument('--log', type=str, default='',
                        help='log name suffix')                   
    parser.add_argument('--teacher_model', type=str, default='',
                        help='self supervised pretrained model')
    parser.add_argument('--score_thre',  type=float,default='0.3',
                        help='score threshold')
    parser.add_argument('--score_thre_1',  type=float,default='0.25',
                        help='the second stage score threshold for static teacher')
    parser.add_argument('--pseudo_label_merge',  type=bool,default=True,
                        help='if merge the pseudo label and orignial label or not')
    parser.add_argument('--ema_decay',  type=float,default='0.999',
                        help='the parameters of teacher model update')
    parser.add_argument('--start_epoch',  type=int,default=10,
                        help='start score threshold')
    parser.add_argument('--stop_epoch',  type=int,default=20,
                        help='stop score threshold')
    parser.add_argument('--ema_usage_way',  type=str,default='epoch',
                        help='epoch, step or best??')
    parser.add_argument('--delta',  type=int,default=5,
                        help='Delay learning rate changes by the number of epochs')
    opt = parser.parse_args()
    return opt


def format_decimal(num):
    a, b = str(num).split('.')
    return float(a + '.' + b[:2])

def get_jnb_threshold(score_list):
    kclf = KMeans(n_clusters=2)
    data_kmeans = np.array(score_list)
    data_kmeans = data_kmeans.reshape(len(data_kmeans), -1)
    kclf.fit(data_kmeans)
    threshod = kclf.cluster_centers_.reshape(-1)
    res = np.sort(threshod)[::-1]
    res = [format_decimal(r) for r in res]
    return res  

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    hypes['log_suffix']=opt.log

    # static teacher变换阈值后，保证学习率从头再来（0.002）
    hypes['train_params']['epoches']+=opt.delta # 20 -> 25
    step_size_list_new = [i + opt.delta for i in hypes['lr_scheduler']['step_size']]
    hypes['lr_scheduler']['step_size']=step_size_list_new # [10,15] -> [15,20]

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    lowest_teacher_val_loss = 1e10
    lowest_teacher_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

        # load pretrainde ssl model if it exists.
        if opt.pretrained_model:
            print('pretrained_model_dir:',opt.pretrained_model)
            model = train_utils.load_pretrained_model(opt.pretrained_model, model)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) # hys 将普通BN替换成SyncBN
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # 加载生成pseudo_label的教师模型
    teacher_model = train_utils.create_model(hypes)
    print('static_teacher_model_dir:',opt.teacher_model)
    teacher_model = train_utils.load_pretrained_model(opt.teacher_model, teacher_model) 
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available():
        teacher_model.to(device)
    
    # EMA mean teacher
    # ema_teacher_model = copy.deepcopy(model)
    ema_teacher_model = copy.deepcopy(model_without_ddp)
    print('ema_teacher_model is a deep copy of the model')
    for p in ema_teacher_model.parameters():
        # p.requires_grad(False)
        p.detach_()
    if torch.cuda.is_available():
        ema_teacher_model.to(device)
    ema_teacher_model.eval()

    # 记录best ema teacher
    # if opt.ema_usage_way=='best':
    best_ema_teacher_model = copy.deepcopy(ema_teacher_model)
    for p in best_ema_teacher_model.parameters():
        p.detach_()
    if torch.cuda.is_available():
        best_ema_teacher_model.to(device)
    best_ema_teacher_model.eval()

    
    # 不需要并行
    # if opt.distributed:
    #     teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model).to(device) # hys 将普通BN替换成SyncBN
    #     teacher_model = \
    #         torch.nn.parallel.DistributedDataParallel(teacher_model,
    #                                                   device_ids=[opt.gpu],
    #                                                   find_unused_parameters=True)
    teacher_model.eval()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    global_step = 0
    auxiliary_teacher = False

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f \n' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)


        # 启动 auxiliary_teacher
        if epoch == opt.start_epoch:
            auxiliary_teacher = True
            if opt.ema_usage_way=='epoch':
                # 复制当前epoch的teacher
                # ema_teacher_model_copy = copy.deepcopy(ema_teacher_model)
                ema_teacher_model_copy = copy.deepcopy(best_ema_teacher_model)
                for p in ema_teacher_model_copy.parameters():
                    p.detach_()
                if torch.cuda.is_available():
                    ema_teacher_model_copy.to(device)
                ema_teacher_model_copy.eval()

        st_score_thre = opt.score_thre_1 if auxiliary_teacher else opt.score_thre
        print('auxiliary: {} score threshold for static teacher: {}'.format(auxiliary_teacher,st_score_thre))

        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch

            # hys add
            ################# 赋予每个scene伪gt ##################
            limit_range = hypes['preprocess']['cav_lidar_range'] # [-32, -32, -3, 32, 32, 2] # v2xsim
            ego_data = batch_data['ego']

            teacher_ouput_dict = teacher_model(ego_data)

            # generate corners_3d_pred for teacher model 3.5
            pseudo_pos_equal_one = []
            pseudo_neg_equal_one = []
            pseudo_target = []

            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float().to(device)
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float().to(device)
            batch_data['ego'].update({'transformation_matrix': transformation_matrix_torch,
                                      'transformation_matrix_clean': transformation_matrix_clean_torch})
            
            batch_cls_preds = teacher_ouput_dict['cls_preds'] # [16, 2, 80, 80]
            batch_reg_preds = teacher_ouput_dict['reg_preds'] # [16, 14, 80, 80]
            batch_dir_preds = teacher_ouput_dict['dir_preds'] # [16, 4, 80, 80]

            # 辅助教师的预测
            if auxiliary_teacher:
                if opt.ema_usage_way=='epoch':
                    teacher_ouput_dict_2 = ema_teacher_model_copy(ego_data) # 每一轮使用固定的ema teacher
                elif opt.ema_usage_way=='best':
                    teacher_ouput_dict_2 = best_ema_teacher_model(ego_data) # 每一轮使用最佳的ema teacher
                else:
                    teacher_ouput_dict_2 = ema_teacher_model(ego_data) # 随着当前ema teacher 变化
                batch_cls_preds_2 = teacher_ouput_dict_2['cls_preds'] # [16, 2, 80, 80]
                batch_reg_preds_2 = teacher_ouput_dict_2['reg_preds'] # [16, 14, 80, 80]
                batch_dir_preds_2 = teacher_ouput_dict_2['dir_preds'] # [16, 4, 80, 80]

                teacher_ouput_dict = ema_teacher_model(ego_data)
                ################# generate dual threshold for EMA Teacher ##################
                # 稀疏标签生成的label
                pos_equal_one = ego_data['label_dict']['pos_equal_one'].permute(0, 3, 1, 2) # [b, h, w, 2]->[b, 2, h, w]
                # 协同端端预测
                foreground_maps_ego = batch_cls_preds_2.sigmoid() # [b, 2, h,w]
                # 统计预测gt的分数
                gt_pred_score = foreground_maps_ego[pos_equal_one > 0] # 
                if len(gt_pred_score) == 1:
                    score_thre_2  = float(gt_pred_score[0])
                else:
                    score_thre_2, _ = get_jnb_threshold(gt_pred_score.tolist()) # high thre, low thre
                print(score_thre_2)

            bs = batch_cls_preds.size(0)
            pred_center_flag = True
            pseudo_label_merge = opt.pseudo_label_merge
            for item in range(bs):
                item_teacher_ouput_dict = {'ego': {}}
                item_teacher_ouput_dict['ego'].update({
                    'cls_preds': batch_cls_preds[item].unsqueeze(0), # [1, 2, 80, 80]
                    'reg_preds': batch_reg_preds[item].unsqueeze(0), # [1, 14, 80, 80]
                    'dir_preds': batch_dir_preds[item].unsqueeze(0),
                    'pred_center' : pred_center_flag}) # [1, 4, 80, 80]
                
                item_teacher_ouput_dict['ego'].update({
                    'score_threshold': st_score_thre, # opt.score_thre
                    'pred_index': auxiliary_teacher
                })

                if pred_center_flag:
                    # 直接用七维的预测值生成label
                    if auxiliary_teacher:
                        pred_center, scores, mask_index = opencood_train_dataset.post_processor.post_process(batch_data, item_teacher_ouput_dict) 
                    else:
                        pred_center, scores = opencood_train_dataset.post_processor.post_process(batch_data, item_teacher_ouput_dict)
                else:
                    # [pred_num, 8, 3], [pred_num]
                    pred_box3d_tensor, scores = opencood_train_dataset.post_processor.post_process(batch_data, item_teacher_ouput_dict) 
                    pred_center = box_utils.corner_to_center_torch(pred_box3d_tensor) # [pred_num,7]

                
                # 辅助教师的预测
                if auxiliary_teacher:
                    # 1. 生成前景预测
                    item_teacher_ouput_dict_2 = {'ego': {}}
                    item_teacher_ouput_dict_2['ego'].update({
                        'cls_preds': batch_cls_preds_2[item].unsqueeze(0), # [1, 2, 80, 80]
                        'reg_preds': batch_reg_preds_2[item].unsqueeze(0), # [1, 14, 80, 80]
                        'dir_preds': batch_dir_preds_2[item].unsqueeze(0),
                        'pred_center' : pred_center_flag}) # [1, 4, 80, 80]
                    
                    item_teacher_ouput_dict_2['ego'].update({
                        'score_threshold': score_thre_2,
                        'pred_index': True
                    })
                    pred_center_2, scores_2, mask_index_2 = opencood_train_dataset.post_processor.post_process(batch_data, item_teacher_ouput_dict_2)
                    
                    if pred_center_2 is not None:
                        if pred_center is None:
                            pred_center = pred_center_2
                        else:
                            # 2. 只保留teacher_2额外预测的值
                            mask_index_list = mask_index.tolist()
                            mask_index_list_2 = mask_index_2.tolist()
                            # TODO 一个位置只保留一个anchor(一共有两个anchor)
                            mask_index_list = [int(x/2) for x in  mask_index_list]
                            mask_index_list_2 = [int(x/2) for x in  mask_index_list_2]
                            # intersection = list(set(mask_index_list).intersection(mask_index_list_2)) # 交集
                            remain_result = [x for x in mask_index_list_2 if x not in mask_index_list] # teacher_2额外预测的
                            remain_index = [mask_index_list_2.index(item) for item in remain_result]
                            pred_center_2 = pred_center_2[remain_index]
                            scores_2 = scores_2[remain_index]
                            # 3. 和teacher的pred_center合并
                            pred_center = torch.cat([pred_center, pred_center_2],dim=0)

                if pred_center is None:
                    psgt_pos_equal_one = np.zeros_like(ego_data['label_dict']['pos_equal_one'][item].cpu().numpy())
                    psgt_neg_equal_one = np.zeros_like(ego_data['label_dict']['neg_equal_one'][item].cpu().numpy())
                    psgt_targets = np.zeros_like(ego_data['label_dict']['targets'][item].cpu().numpy())
                else:
                    pseudo_object_bbx_center = np.zeros((200, 7))
                    pseudo_mask = np.zeros(200)                
                    pseudo_object_bbx_center[:pred_center.size(0), :] = pred_center.cpu()
                    pseudo_mask[:pred_center.size(0)] = 1
                    item_pseudo_label_dict = opencood_train_dataset.post_processor.generate_label(
                        gt_box_center=pseudo_object_bbx_center, 
                        anchors=ego_data['anchor_box'].cpu().numpy(), 
                        mask=pseudo_mask)

                    # pseudo label
                    psgt_pos_equal_one = item_pseudo_label_dict['pos_equal_one']
                    psgt_neg_equal_one = item_pseudo_label_dict['neg_equal_one']
                    psgt_targets = item_pseudo_label_dict['targets']

                # 与真实的稀疏标签合并
                if pseudo_label_merge:
                    ssgt_pos_equal_one = ego_data['label_dict']['pos_equal_one'][item].cpu().numpy() # (80,80,2)
                    ssgt_neg_equal_one = ego_data['label_dict']['neg_equal_one'][item].cpu().numpy() # (80,80,2)
                    ssgt_targets = ego_data['label_dict']['targets'][item].cpu().numpy() # (80,80,14)
                    
                    h, w, anchor_num = ssgt_pos_equal_one.shape
            
                    psgt_targets_merge = ssgt_targets + psgt_targets # 合并回归标签 (80,80,14)
                    psgt_pos_equal_one_merge = ssgt_pos_equal_one + psgt_pos_equal_one # 合并分类标签 (80,80,2)
                    equal_index = (psgt_pos_equal_one_merge==2) # 伪标签和稀疏标签的共同之处 (80,80,2)
                    if equal_index.sum()>0:
                        # 若伪标签和稀疏标签产生交集，回归值以稀疏标签的为准
                        psgt_pos_equal_one_merge[equal_index]=1
                        psgt_targets_merge_1 = psgt_targets_merge.reshape(h,w,anchor_num,7) # (80,80,2,7)
                        ssgt_targets_1 = ssgt_targets.reshape(h,w,anchor_num,7) # (80,80,2,7)
                        psgt_targets_merge_1[equal_index]=ssgt_targets_1[equal_index] # (80,80,2,7)
                        psgt_targets_merge = psgt_targets_merge_1.reshape(h,w,anchor_num*7) # (80,80,14)
                    psgt_neg_equal_one_merge = np.logical_and(ssgt_neg_equal_one, psgt_neg_equal_one) # 交集
                    
                    pseudo_pos_equal_one.append(psgt_pos_equal_one_merge)
                    pseudo_neg_equal_one.append(psgt_neg_equal_one_merge)
                    pseudo_target.append(psgt_targets_merge)    
                    
                else:
                    pseudo_pos_equal_one.append(psgt_pos_equal_one)
                    pseudo_neg_equal_one.append(psgt_neg_equal_one)
                    pseudo_target.append(psgt_targets)
            
            pseudo_pos_equal_one = torch.from_numpy(np.array(pseudo_pos_equal_one)).to(device)
            pseudo_neg_equal_one = torch.from_numpy(np.array(pseudo_neg_equal_one)).to(device)
            pseudo_target = torch.from_numpy(np.array(pseudo_target)).to(device)


            batch_data['ego']['label_dict']['pos_equal_one'] = pseudo_pos_equal_one
            batch_data['ego']['label_dict']['neg_equal_one'] = pseudo_neg_equal_one
            batch_data['ego']['label_dict']['targets'] = pseudo_target
            
            # assert (batch_data['ego']['label_dict']['pos_equal_one']!=pseudo_pos_equal_one).sum().item()==0
            # assert (batch_data['ego']['label_dict']['neg_equal_one']!=pseudo_neg_equal_one).sum().item()==0
            # assert (batch_data['ego']['label_dict']['targets']!=pseudo_target).sum().item()==0
            ################# 赋予每个scene伪gt ##################

            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            # criterion.logging(epoch, i, len(train_loader), writer)
            criterion.logging(epoch, i, len(train_loader), writer,pbar=pbar2)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                # criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single", pbar=pbar2)


            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()


            torch.cuda.empty_cache()
            
            ################# update_ema_variables correct ##################
            # Use the true average until the exponential average is more correct
            # https://zhuanlan.zhihu.com/p/511761402
            alpha = min(1 - 1 / (global_step + 1), opt.ema_decay) # ema-decay=0.999
            with torch.no_grad():
                # model_state_dict = model.state_dict()
                model_state_dict = model_without_ddp.state_dict()
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

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

                    # validate ema teacher model 
                    ema_teacher_model.zero_grad()
                    ema_teacher_model.eval()
                    teacher_ouput_dict = ema_teacher_model(batch_data['ego'])

                    teacher_final_loss = criterion(teacher_ouput_dict, batch_data['ego']['label_dict'])
                    teacher_valid_ave_loss.append(teacher_final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f \n' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            teacher_valid_ave_loss = statistics.mean(teacher_valid_ave_loss)
            print('At epoch %d, the teacher validation loss is %f \n' % (epoch, teacher_valid_ave_loss))
            writer.add_scalar('Teacher_Validate_Loss', teacher_valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1
            
            if teacher_valid_ave_loss < lowest_teacher_val_loss:
                lowest_teacher_val_loss = teacher_valid_ave_loss
                torch.save(ema_teacher_model.state_dict(), os.path.join(saved_path, 'ema_net_epoch_bestval_at%d.pth' % (epoch + 1)))
                
                # if opt.ema_usage_way=='best':
                if epoch < opt.start_epoch:
                    best_ema_teacher_model = copy.deepcopy(ema_teacher_model)
                    for p in best_ema_teacher_model.parameters():
                        p.detach_()
                    if torch.cuda.is_available():
                        best_ema_teacher_model.to(device)
                    best_ema_teacher_model.eval()
                    
                if lowest_teacher_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'ema_net_epoch_bestval_at%d.pth' % (lowest_teacher_val_epoch))):
                    os.remove(os.path.join(saved_path, 'ema_net_epoch_bestval_at%d.pth' % (lowest_teacher_val_epoch)))
                lowest_teacher_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            torch.save(ema_teacher_model.state_dict(), os.path.join(saved_path, 'ema_net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            # import numpy as np
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
                cmd = f"CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
                cmd_ema = f"CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference_ema.py --model_dir {saved_path} --fusion_method {fusion_method}"
            print(f"Running command: {cmd}")
            os.system(cmd)
            print(f"Running command: {cmd_ema}")
            os.system(cmd_ema)


if __name__ == '__main__':
    main()
