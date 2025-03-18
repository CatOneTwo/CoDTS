##################################################################################################################
## Generate pseudo labels with a dynamic teacher
## HINTED: Hard Instance Enhanced Detector with Mixed-Density Feature Fusion for Sparsely-Supervised 3D Object Detection (CVPR 2024)
##################################################################################################################

# DATASET=dairv2x
DATASET=opv2v
# DATASET=v2xsim
# DATASET=v2vreal

SETTING=lidar_only_weakly_supervised

FUSION_METHOD=intermediate # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]

# METHOD=att
# METHOD=fcooper
METHOD=discograph
# METHOD=v2xvit
# METHOD=v2vnet

YAML_FILE=opencood/hypes_yaml/$DATASET/$SETTING/pointpillar_${METHOD}.yaml

# single GPU: finetune，v2xsim, dair_v2x

TEACHER_MODEL=${METHOD}_ws_single_mae_0.7 # 第一步
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_ws_hinted.py \
-y $YAML_FILE \
--fusion_method $FUSION_METHOD \
--pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${TEACHER_MODEL} \
--log hinted


# multiple GPUs: opv2v, V2VReal
# CUDA_VISIBLE_DEVICES=0,7 python -m torch.distributed.launch  --nproc_per_node=2 --use_env --master_port=2231 opencood/tools/train_ws_hinted_ddp.py  \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${TEACHER_MODEL} \
# --log hinted

