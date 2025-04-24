#!/bin/bash

LOG_FILE="/volume1/home/pxie/topic_model/solution4/output/independent_training_log_20250423_005356.log"
echo "开始独立训练多个数据集" | tee -a "$LOG_FILE"
echo "====================================" | tee -a "$LOG_FILE"

TOTAL_START_TIME=$(date +%s)

# 训练第一个数据集
"/volume1/home/pxie/topic_model/solution4/output/run_independent_with_timer.sh" "/volume1/home/pxie/data/PBMC.h5ad" "PBMC" "/volume1/home/pxie/data/embeddings/fused_gene_embedding.pkl" "/volume1/home/pxie/topic_model/solution4/results" "/volume1/home/pxie/topic_model/solution4/results/checkpoint" "$LOG_FILE"

# 训练第二个数据集
"/volume1/home/pxie/topic_model/solution4/output/run_independent_with_timer.sh" "/volume1/home/pxie/data/combined_1.h5ad" "combined_1" "/volume1/home/pxie/data/embeddings/fused_gene_embedding.pkl" "/volume1/home/pxie/topic_model/solution4/results" "/volume1/home/pxie/topic_model/solution4/results/checkpoint" "$LOG_FILE" 

# 训练第三个数据集
"/volume1/home/pxie/topic_model/solution4/output/run_independent_with_timer.sh" "/volume1/home/pxie/data/combined_2.h5ad" "combined_2" "/volume1/home/pxie/data/embeddings/fused_gene_embedding.pkl" "/volume1/home/pxie/topic_model/solution4/results" "/volume1/home/pxie/topic_model/solution4/results/checkpoint" "$LOG_FILE" 

# 计算所有数据集的总训练时间
TOTAL_END_TIME=$(date +%s)
TOTAL_SECONDS=$((TOTAL_END_TIME - TOTAL_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

echo "====================================" | tee -a "$LOG_FILE"
echo "所有数据集总训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$LOG_FILE"
echo "====================================" | tee -a "$LOG_FILE"
