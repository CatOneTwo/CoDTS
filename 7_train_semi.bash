
DATASET=dairv2x
# DATASET=opv2v
# DATASET=v2xsim


# SETTING=lidar_only_perfect
# SETTING=lidar_only_weakly_supervised
SETTING=lidar_only_semi_supervised

# METHOD=pointpillar_single
# METHOD=pointpillar_early

# METHOD=fcooper
# METHOD=att
METHOD=discograph
# METHOD=v2vnet
# METHOD=v2xvit

for METHOD in fcooper att discograph
do

YAML_FILE=opencood/hypes_yaml/$DATASET/$SETTING/pointpillar_$METHOD.yaml

# FUSION_METHOD=late # 所有协作车辆的后融合检测结果，使用场景所有车的gt box。 [late fusion dataset支持]
# FUSION_METHOD=early # 所有协作车辆的前融合检测结果，使用场景所有车的gt box。 [early fusion dataset支持]
FUSION_METHOD=intermediate # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]

# SEMI_RATIO=0.01
# SEMI_RATIO=0.02
# SEMI_RATIO=0.05
# SEMI_RATIO=0.1
SEMI_RATIO=0.2

# 使用labeled set进行预训练
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_semi_stage1.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --semi_ratio $SEMI_RATIO \
# --log stage1_$SEMI_RATIO


PRETRAINED_MODEL=${METHOD}_semi_stage1_${SEMI_RATIO} # 第一步

# Mean-Teacher
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_semi_stage2.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --semi_ratio $SEMI_RATIO \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
# --log stage2_$SEMI_RATIO

# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_semi_stage2.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --semi_ratio $SEMI_RATIO \
# --log stage2_${SEMI_RATIO}_nopretrain


# 3DIou-Match
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_semi_stage3.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --semi_ratio $SEMI_RATIO \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
# --log stage3_$SEMI_RATIO

# CoDTS
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_semi_stage4.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --semi_ratio $SEMI_RATIO \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
# --log stage4_${SEMI_RATIO}_thre_0.2_0.5_0.5

# HSSDA
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_semi_stage5.py \
# -y $YAML_FILE \
# --fusion_method $FUSION_METHOD \
# --semi_ratio $SEMI_RATIO \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
# --log stage5_$SEMI_RATIO

# HINTED
CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_semi_stage6.py \
-y $YAML_FILE \
--fusion_method $FUSION_METHOD \
--semi_ratio $SEMI_RATIO \
--pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_${PRETRAINED_MODEL} \
--log stage6_$SEMI_RATIO
done