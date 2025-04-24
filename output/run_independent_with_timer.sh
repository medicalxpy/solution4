#!/bin/bash

# 获取脚本参数
ADATA_PATH="$1"
DATA_NAME="$2"
GENE_EMBEDDING_PATH="$3"
OUTPUT_DIR="$4"
CHECKPOINT_DIR="$5"
LOG_FILE="$6"

# 记录开始时间
START_TIME=$(date +%s)
echo "开始训练数据集 $DATA_NAME 时间: $(date)" | tee -a "$LOG_FILE"

# 执行训练命令
python train_multi_datasets.py \
    --adata_paths "$ADATA_PATH" \
    --data_names "$DATA_NAME" \
    --gene_embedding_path "$GENE_EMBEDDING_PATH" \
    --model_name \"independent_full\" \
    --num_topics 50 \
    --batch_size 1024 \
    --lr 2e-3 \
    --num_epochs 100 \
    --patience 10 \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "cuda:0" \
    --cell_embed_epochs 300 \
    --cell_embed_lr 1e-4 2>&1 | tee -a "$LOG_FILE"

# 记录结束时间
END_TIME=$(date +%s)
echo "结束训练数据集 $DATA_NAME 时间: $(date)" | tee -a "$LOG_FILE"

# 计算总运行时间
TOTAL_SECONDS=$((END_TIME - START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

# 输出总运行时间
echo "====================================" | tee -a "$LOG_FILE"
echo "数据集 $DATA_NAME 总训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$LOG_FILE"
echo "====================================" | tee -a "$LOG_FILE"
