##################################################################################################################
## Train intermediate collaborative detector from scratch with full or sparse labels 
##################################################################################################################

# DATASET=dairv2x
# DATASET=v2xsim
# DATASET=opv2v
DATASET=v2vreal

# SETTING=lidar_only_perfect # full label
SETTING=lidar_only_weakly_supervised # sparse label

# METHOD=pointpillar_single
# METHOD=pointpillar_early

# METHOD=pointpillar_fcooper
# METHOD=pointpillar_att
METHOD=pointpillar_discograph
# METHOD=pointpillar_v2vnet
# METHOD=pointpillar_v2xvit

YAML_FILE=opencood/hypes_yaml/$DATASET/$SETTING/$METHOD.yaml

# FUSION_METHOD=late # 所有协作车辆的后融合检测结果，使用场景所有车的gt box。 [late fusion dataset支持]
# FUSION_METHOD=early # 所有协作车辆的前融合检测结果，使用场景所有车的gt box。 [early fusion dataset支持]
FUSION_METHOD=intermediate # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]


# (single GPU version) for dairv2x and v2xsim
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
-y $YAML_FILE \
--fusion_method $FUSION_METHOD 
--log 

# (multiple GPUs version) for opv2v and v2vreal
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env --master_port=2233 opencood/tools/train_ddp.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD 

