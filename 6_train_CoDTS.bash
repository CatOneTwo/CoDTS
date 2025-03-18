##################################################################################################################
## Generate pseudo labels with a static teacher and dynamic teacher
## CoDTS: Enhancing Sparsely Supervised Collaborative Perception with a Dual Teacher-Student Framework (AAAI 2025)
##################################################################################################################


# DATASET=dairv2x
DATASET=opv2v
# DATASET=v2xsim
# DATASET=v2vreal

SETTING=lidar_only_weakly_supervised

FUSION_METHOD=intermediate # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]

# METHOD=att
METHOD=fcooper
# METHOD=discograph
# METHOD=v2xvit
# METHOD=v2vnet


YAML_FILE=opencood/hypes_yaml/$DATASET/$SETTING/pointpillar_${METHOD}.yaml

SCORE=0.15 # low threshold of static teacher in the warm-up stage
SCORE1=0.25 # high threshold of static teacher in the refinement stage

DELAY=5
PRETRAINED_MODEL=single_mae_0.7 # 第一步
TEACHER_MODEL=${METHOD}_ws_single_mae_0.7 # 第一步

# single gpu: v2xsim, dairv2x

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_ws_codts.py \
-y $YAML_FILE \
--fusion_method $FUSION_METHOD \
--pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
--teacher_model opencood/logs/${DATASET}_point_pillar_lidar_${TEACHER_MODEL} \
--score_thre $SCORE \
--score_thre_1 $SCORE1 \
--delta $DELAY \
--log st_${SCORE}to${SCORE1}



# multiple gpus: OPV2V, v2vreal
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env --master_port=2235 opencood/tools/train_ws_codts_ddp.py  \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
# --score_thre $SCORE \
# --score_thre_1 $SCORE1 \
# --teacher_model opencood/logs/${DATASET}_point_pillar_lidar_${TEACHER_MODEL} \
# --delta $DELAY \
# --log st_${SCORE}to${SCORE1}









