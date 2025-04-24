#!/bin/bash
# 独立训练多个数据集脚本 - 每个数据集单独训练，不使用增量训练逻辑

# 设置数据路径
ADATA_PATH1="/volume1/home/pxie/data/PBMC.h5ad"
ADATA_PATH2="/volume1/home/pxie/data/combined_1.h5ad"
ADATA_PATH3="/volume1/home/pxie/data/combined_2.h5ad"
GENE_EMBEDDING_PATH="/volume1/home/pxie/data/embeddings/fused_genePT_geneformerv2.pkl"

# 设置输出目录
OUTPUT_DIR="/volume1/home/pxie/topic_model/solution4/results"
CHECKPOINT_DIR="/volume1/home/pxie/topic_model/solution4/results/checkpoint"
LOG_DIR="/volume1/home/pxie/topic_model/solution4/output"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# 获取当前时间作为日志文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/independent_training_log_$TIMESTAMP.log"

# 创建包含计时功能的训练脚本
cat > "$LOG_DIR/run_independent_with_timer.sh" << 'EOF'
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
    --model_name \independent_full\ \
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
EOF

# 设置执行权限
chmod +x "$LOG_DIR/run_independent_with_timer.sh"

# 创建主训练脚本
cat > "$LOG_DIR/train_all_independent.sh" << EOF
#!/bin/bash

LOG_FILE="$LOG_FILE"
echo "开始独立训练多个数据集" | tee -a "\$LOG_FILE"
echo "====================================" | tee -a "\$LOG_FILE"

TOTAL_START_TIME=\$(date +%s)

# 训练第一个数据集
"$LOG_DIR/run_independent_with_timer.sh" "$ADATA_PATH1" "PBMC" "$GENE_EMBEDDING_PATH" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "\$LOG_FILE"

# 训练第二个数据集
"$LOG_DIR/run_independent_with_timer.sh" "$ADATA_PATH2" "combined_1" "$GENE_EMBEDDING_PATH" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "\$LOG_FILE" 

# 训练第三个数据集
"$LOG_DIR/run_independent_with_timer.sh" "$ADATA_PATH3" "combined_2" "$GENE_EMBEDDING_PATH" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "\$LOG_FILE" 

# 计算所有数据集的总训练时间
TOTAL_END_TIME=\$(date +%s)
TOTAL_SECONDS=\$((TOTAL_END_TIME - TOTAL_START_TIME))
HOURS=\$((TOTAL_SECONDS / 3600))
MINUTES=\$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=\$((TOTAL_SECONDS % 60))

echo "====================================" | tee -a "\$LOG_FILE"
echo "所有数据集总训练时间: \${HOURS}小时 \${MINUTES}分钟 \${SECONDS}秒" | tee -a "\$LOG_FILE"
echo "====================================" | tee -a "\$LOG_FILE"
EOF

# 设置执行权限
chmod +x "$LOG_DIR/train_all_independent.sh"

# 使用nohup在后台运行训练脚本
nohup "$LOG_DIR/train_all_independent.sh" > /dev/null 2>&1 &

# 记录进程ID
PID=$!

# 输出进程ID，方便后续查看或终止进程
echo "独立训练已在后台启动，进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 命令查看训练进度"
