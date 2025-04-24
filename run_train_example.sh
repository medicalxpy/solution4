#!/bin/bash
# 示例脚本：如何使用新的训练脚本

# 设置数据路径
ADATA_PATH1="/volume1/home/pxie/data/PBMC.h5ad"
ADATA_PATH2="/volume1/home/pxie/data/Cortex.h5ad"
GENE_EMBEDDING_PATH="/volume1/home/pxie/data/embeddings/GenePT.pkl"

# 设置输出目录
OUTPUT_DIR="/volume1/home/pxie/topic_model/solution4/results"
CHECKPOINT_DIR="/volume1/home/pxie/topic_model/solution4/results/checkpoint"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

# 运行训练脚本
python train_2.py \
    --adata_paths $ADATA_PATH1 $ADATA_PATH2 \
    --data_names "PBMC" "Cortex" \
    --gene_embedding_path $GENE_EMBEDDING_PATH \
    --model_name "gpt_base" \
    --num_topics 50 \
    --batch_size 64 \
    --lr 2e-3 \
    --num_epochs 300 \
    --patience 10 \
    --output_dir $OUTPUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --device "cuda:0" \
    --cell_embed_epochs 300 \
    --cell_embed_lr 1e-4
