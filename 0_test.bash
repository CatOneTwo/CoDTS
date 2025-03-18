# inference

MODEL_DIR=opencood/logs/xxx

FUSION_METHOD=intermediate # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py \
--model_dir $MODEL_DIR \
--fusion_method $FUSION_METHOD \
--save_vis_interval 40 


