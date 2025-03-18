##################################################################################################################
## Generate pseudo labels with a static teacher
## SSC3OD: Sparsely Supervised Collaborative 3D Object Detection from LiDAR Point Clouds (SMC 2023)
##################################################################################################################

# DATASET=dairv2x
# DATASET=opv2v
DATASET=v2xsim
# DATASET=v2vreal

SETTING=lidar_only_weakly_supervised

FUSION_METHOD=intermediate # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]

# METHOD=att
# METHOD=fcooper
METHOD=discograph
# METHOD=v2xvit
# METHOD=v2vnet

YAML_FILE=opencood/hypes_yaml/$DATASET/$SETTING/pointpillar_${METHOD}.yaml

# 单卡finetune，v2xsim和dair_v2x

SCORE=0.3
PRETRAINED_MODEL=single_mae_0.7 # 第一步
TEACHER_MODEL=${METHOD}_ws_single_mae_0.7 # 第一步

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_ws.py \
-y $YAML_FILE \
--fusion_method $FUSION_METHOD \
--pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
--score_thre $SCORE \
--teacher_model opencood/logs/${DATASET}_point_pillar_lidar_${TEACHER_MODEL} \
--log finetune_${SCORE}


# 多卡finetune，opv2v, v2vreal

# SCORE=0.3
# PRETRAINED_MODEL=single_mae_0.7 
# TEACHER_MODEL=${METHOD}_ws_single_mae_0.7 
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch  --nproc_per_node=2 --use_env --master_port=2231 opencood/tools/train_ws_ddp.py  \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
# --score_thre $SCORE \
# --teacher_model opencood/logs/${DATASET}_point_pillar_lidar_${TEACHER_MODEL} \
# --log finetune_${SCORE}









