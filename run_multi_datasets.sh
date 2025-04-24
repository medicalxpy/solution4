#!/bin/bash
# 多数据集训练脚本 - 使用新的嵌入和数据集

# 设置数据路径
ADATA_PATH1="/volume1/home/pxie/data/PBMC.h5ad"
ADATA_PATH2="/volume1/home/pxie/data/Cortex.h5ad"
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
LOG_FILE="$LOG_DIR/train_log_$TIMESTAMP.log"

# 创建包含计时功能的训练脚本
cat > "$LOG_DIR/run_with_timer.sh" << 'EOF'
#!/bin/bash

# 获取脚本参数
PYTHON_CMD="$1"
LOG_FILE="$2"

# 记录开始时间
START_TIME=$(date +%s)
echo "开始训练时间: $(date)" | tee -a "$LOG_FILE"

# 执行训练命令
eval "$PYTHON_CMD" 2>&1 | tee -a "$LOG_FILE"

# 记录结束时间
END_TIME=$(date +%s)
echo "结束训练时间: $(date)" | tee -a "$LOG_FILE"

# 计算总运行时间
TOTAL_SECONDS=$((END_TIME - START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

# 输出总运行时间
echo "====================================" | tee -a "$LOG_FILE"
echo "总训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$LOG_FILE"
echo "====================================" | tee -a "$LOG_FILE"
EOF

# 设置执行权限
chmod +x "$LOG_DIR/run_with_timer.sh"

# 准备训练命令
TRAIN_CMD="python train_multi_datasets.py \
    --adata_paths $ADATA_PATH1 $ADATA_PATH2  \
    --data_names \"PBMC\" \"Cortex\"  \
    --gene_embedding_path $GENE_EMBEDDING_PATH \
    --model_name \"fused_genePT_geneformerv2\" \
    --num_topics 50 \
    --batch_size 1024 \
    --lr 2e-3 \
    --num_epochs 100 \
    --patience 10 \
    --output_dir $OUTPUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --device \"cuda:0\" \
    --cell_embed_epochs 300 \
    --cell_embed_lr 1e-4"

# 使用nohup在后台运行训练脚本
nohup "$LOG_DIR/run_with_timer.sh" "$TRAIN_CMD" "$LOG_FILE" > /dev/null 2>&1 &

# 记录进程ID
PID=$!

# 输出进程ID，方便后续查看或终止进程
echo "训练已在后台启动，进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 命令查看训练进度"
