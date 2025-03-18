##################################################################################################################
## Train encoder with self-supervised learning method Pillar-MAE proposed in SSC3OD (SMC 2023)
##################################################################################################################

# step 1: Train encoder with self-supervised learning method Pillar-MAE

# 1. --------------dataset--------------
# DATASET=dairv2x
# DATASET=opv2v
DATASET=v2xsim
# DATASET=v2vreal

# 2. --------------experiment setting--------------
SETTING=lidar_only_weakly_supervised

# 3. --------------MAE method--------------
METHOD=ssl_pointpillar_single_mae
FUSION_METHOD=late 

# single GPU
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_mae.py \
-y opencood/hypes_yaml/$DATASET/$SETTING/$METHOD.yaml \
--fusion_method $FUSION_METHOD \
--log 0.7 # mask ration in pillar-mae is 0.7



