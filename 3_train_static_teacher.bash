##################################################################################################################
## Load pre-trained encoder, and train static teacher with sparse labels (SMC 2023)
##################################################################################################################

# step 2: train static teacher with sparse labels

# 1. --------------dataset--------------
# DATASET=dairv2x
# DATASET=opv2v
DATASET=v2xsim
# DATASET=v2vreal

# 2. --------------experiment setting--------------
SETTING=lidar_only_weakly_supervised

# 3. --------------backbone & fusion method--------------
# METHOD=pointpillar_single
# METHOD=pointpillar_early

# METHOD=pointpillar_fcooper
# METHOD=pointpillar_att
METHOD=pointpillar_discograph
# METHOD=pointpillar_v2vnet
# METHOD=pointpillar_v2xvit

# 4.--------------fusion scheme--------------
# FUSION_METHOD=late 
# FUSION_METHOD=early 
FUSION_METHOD=intermediate 

# 5. --------------ssl pretrained model--------------
PRETRAINED_MODEL=single_mae_0.7

# single gpu: v2xsim, dairv2x
CUDA_VISIBLE_DEVICES=3 python opencood/tools/train.py \
-y opencood/hypes_yaml/$DATASET/$SETTING/$METHOD.yaml \
--fusion_method $FUSION_METHOD \
--pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
--log ${PRETRAINED_MODEL}

# multiple gpus: opv2v, v2vreal
# for METHOD in {pointpillar_fcooper,pointpillar_att,pointpillar_discograph}
#   do

#     CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch  --nproc_per_node=2 --use_env --master_port=2238 opencood/tools/train_ddp.py \
#     -y opencood/hypes_yaml/$DATASET/$SETTING/$METHOD.yaml \
#     --fusion_method $FUSION_METHOD \
#     --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
#     --log $PRETRAINED_MODEL
# done
