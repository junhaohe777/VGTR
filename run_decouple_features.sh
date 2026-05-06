#!/bin/bash

# 核心改进：解决Memory导致的特征坍缩问题
# 
# 问题诊断：
# - fine_text_features 被 Memory 拉向稳定的语义原型
# - 后期训练 batch-level 特征分布过于集中
# - SDM 的 KL 散度梯度变小，排序区分度饱和
#
# 解决方案：
# - SDM/ITC: 使用原始 t_feats（判别能力，多样性强）
# - ID/MLM: 使用 fine_text_features（语义理解，记忆增强）
# - 解耦判别和语义，各司其职
#
# 理论依据：
# 1. SDM 是对比学习，需要 batch 内多样化的负样本
# 2. Memory 增强适合语义理解任务（分类、MLM）
# 3. 原始 CLIP 特征已经很强，判别任务不需要增强
#
# 预期效果：
# - 解决所有改进失败的根本原因
# - 74.2~74.5% (+0.22~0.52%)

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name iira_decouple_features_id \
--img_aug \
--batch_size 128 \
--MLM \
--loss_names 'sdm+id+mlm' \
--dataset_name 'CUHK-PEDES' \
--num_epoch 60 \
--memory_size 32 \
--num_slots 4 \
--memory_alpha 0.01 \
--memory_momentum 0.9 \
--memory_warmup_epochs 5 \
--learnable_alpha true
