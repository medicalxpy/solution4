"""
多数据集主题模型训练脚本

该脚本支持在多个h5ad数据集上训练主题模型，并按照规范化格式保存结果。
训练过程中，模型会从一个数据集学习到的先验知识转移到下一个数据集。

使用方法:
    python train_multi_datasets.py --adata_paths /path/to/data1.h5ad /path/to/data2.h5ad ...
                     --gene_embedding_path /path/to/gene_embedding.pkl
                     --model_name model_name
                     --num_topics 50
                     --batch_size 64
                     --lr 2e-3
                     --num_epochs 300
                     --patience 10
                     --output_dir ./results/
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import random
import time
from tqdm import tqdm
from pathlib import Path

from model import CTM
from dataset import CTMDataset
from utils.cell_embedding import generate_cell_embedding as generate_cell_embedding_new

def set_seed(seed):
    """
    设置随机种子，确保实验可重复性

    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多数据集主题模型训练")

    # 数据相关参数
    parser.add_argument("--adata_paths", nargs="+", required=True,
                        help="h5ad文件路径列表，可以指定多个文件")
    parser.add_argument("--gene_embedding_path", type=str, required=True,
                        help="基因嵌入文件路径")
    parser.add_argument("--data_names", nargs="+", default=None,
                        help="数据集名称列表，如果不指定，将使用文件名")

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="ctm_model",
                        help="模型名称，用于保存结果")
    parser.add_argument("--num_topics", type=int, default=50,
                        help="主题数量")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="训练轮数")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值")

    # 输出相关参数
    parser.add_argument("--output_dir", type=str, default="./results/",
                        help="输出目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./results/checkpoint/",
                        help="模型检查点保存目录")

    # 其他参数
    parser.add_argument("--device", type=str, default=None,
                        help="使用的设备，例如 'cuda:0'，如果不指定则自动选择")
    parser.add_argument("--cell_embed_epochs", type=int, default=300,
                        help="细胞嵌入训练轮数")
    parser.add_argument("--cell_embed_lr", type=float, default=1e-4,
                        help="细胞嵌入学习率")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，用于确保实验可重复性")

    args = parser.parse_args()

    # 如果没有指定数据名称，则使用文件名
    if args.data_names is None:
        args.data_names = [Path(path).stem for path in args.adata_paths]

    # 确保数据名称和文件路径数量一致
    if len(args.data_names) != len(args.adata_paths):
        raise ValueError("数据名称列表长度必须与数据文件路径列表长度相同")

    # 如果没有指定设备，则自动选择
    if args.device is None:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args

def prepare_dataset(adata_path, gene_embedding_path, device, cell_embed_epochs=300, cell_embed_lr=1e-4):
    """
    准备数据集

    参数:
        adata_path: h5ad文件路径
        gene_embedding_path: 基因嵌入文件路径
        device: 使用的设备
        cell_embed_epochs: 细胞嵌入训练轮数
        cell_embed_lr: 细胞嵌入学习率

    返回:
        train_dataset: 训练数据集
        gene_counts_common: 基因计数
        cell_embeddings: 细胞嵌入
    """
    print(f"正在处理数据集: {adata_path}")

    # 生成细胞嵌入
    print("生成细胞嵌入...")
    cell_embeddings, gene_counts_common, _ = generate_cell_embedding_new(
        adata_path=adata_path,
        gene_embedding_path=gene_embedding_path
    )

    # 准备训练数据集
    num_genes = gene_counts_common.n_vars
    id2token = {i: list(gene_counts_common.var_names)[i] for i in range(num_genes)}
    cell_embeddings_numpy = cell_embeddings.detach().numpy()

    train_dataset = CTMDataset(
        X_contextual=cell_embeddings_numpy,
        X_bow=gene_counts_common.X.toarray(),
        idx2token=id2token
    )

    print(f"数据集准备完成，基因数: {num_genes}, 细胞数: {len(cell_embeddings)}")

    return train_dataset, gene_counts_common, cell_embeddings

def train_model(train_dataset, cell_embeddings, num_topics, batch_size, lr, num_epochs,
                patience, device, prior_mean=None, prior_variance=None, seed=42):
    """
    训练主题模型

    参数:
        train_dataset: 训练数据集
        cell_embeddings: 细胞嵌入
        num_topics: 主题数量
        batch_size: 批次大小
        lr: 学习率
        num_epochs: 训练轮数
        patience: 早停耐心值
        device: 使用的设备
        prior_mean: 先验均值
        prior_variance: 先验方差
        seed: 随机种子

    返回:
        ctm: 训练好的模型
        prior_mean: 更新后的先验均值
        prior_variance: 更新后的先验方差
    """
    print("开始训练主题模型...")

    # 创建模型
    ctm = CTM(
        contextual_size=cell_embeddings.shape[1],
        bow_size=train_dataset.X_bow.shape[1],
        n_components=num_topics,
        batch_size=batch_size,
        device=device,
        lr=lr,
        num_epochs=num_epochs,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        seed=seed
    )

    # 训练模型
    prior_mean, prior_variance = ctm.fit(
        train_dataset,
        return_mean=False,
        patience=patience,
        seed=seed
    )

    print("模型训练完成")

    return ctm, prior_mean, prior_variance

def save_results(ctm, train_dataset, gene_counts_common, model_name, train_order, data_name,
                num_topics, output_dir, checkpoint_dir):
    """
    保存训练结果

    参数:
        ctm: 训练好的模型
        train_dataset: 训练数据集
        gene_counts_common: 基因计数
        model_name: 模型名称
        train_order: 训练顺序
        data_name: 数据集名称
        num_topics: 主题数量
        output_dir: 输出目录
        checkpoint_dir: 检查点目录
    """
    print(f"保存模型和结果: {model_name}_{train_order}_{data_name}")

    # 保存模型检查点
    checkpoint_name = f"{model_name}_{train_order}_{data_name}"
    ctm.save(model_dir=checkpoint_dir, part_name=checkpoint_name)

    # 获取主题分布
    topics_per_cell = ctm.get_thetas(train_dataset)
    df_topics_per_cell = pd.DataFrame(topics_per_cell)

    # 保存细胞-主题矩阵
    cell_output_file = os.path.join(output_dir, f"{model_name}_{train_order}_{data_name}_t{num_topics}_c.csv")
    df_topics_per_cell.to_csv(cell_output_file, index=False)
    print(f"细胞-主题矩阵已保存至: {cell_output_file}")

    # 获取主题-基因矩阵
    topics_per_gene = ctm.get_topic_word_matrix()
    df_topics_per_gene = pd.DataFrame(topics_per_gene)
    df_gene_topic = df_topics_per_gene.T
    df_gene_topic.index = gene_counts_common.var_names

    # 保存基因-主题矩阵
    gene_output_file = os.path.join(output_dir, f"{model_name}_{train_order}_{data_name}_t{num_topics}_g.csv")
    df_gene_topic.to_csv(gene_output_file)
    print(f"基因-主题矩阵已保存至: {gene_output_file}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print("=" * 50)
    print("多数据集主题模型训练")
    print("=" * 50)
    print(f"模型名称: {args.model_name}")
    print(f"主题数量: {args.num_topics}")
    print(f"数据集数量: {len(args.adata_paths)}")
    print(f"使用设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print("=" * 50)

    # 初始化先验参数
    prior_mean = 0.0
    prior_variance = None

    # 存储所有数据集的训练数据和结果
    all_datasets = []
    all_gene_counts = []
    all_cell_embeddings = []

    # 准备所有数据集
    print("准备所有数据集...")
    for adata_path in args.adata_paths:
        train_dataset, gene_counts_common, cell_embeddings = prepare_dataset(
            adata_path=adata_path,
            gene_embedding_path=args.gene_embedding_path,
            device=args.device,
            cell_embed_epochs=args.cell_embed_epochs,
            cell_embed_lr=args.cell_embed_lr
        )
        all_datasets.append(train_dataset)
        all_gene_counts.append(gene_counts_common)
        all_cell_embeddings.append(cell_embeddings)

    # 依次训练每个数据集
    for i, (train_dataset, gene_counts_common, cell_embeddings, data_name) in enumerate(
        zip(all_datasets, all_gene_counts, all_cell_embeddings, args.data_names)
    ):
        train_order = i + 1
        print(f"\n开始训练数据集 {train_order}/{len(args.adata_paths)}: {data_name}")

        # 训练模型
        ctm, prior_mean, prior_variance = train_model(
            train_dataset=train_dataset,
            cell_embeddings=cell_embeddings,
            num_topics=args.num_topics,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            patience=args.patience,
            device=args.device,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            seed=args.seed
        )

        # 保存结果
        save_results(
            ctm=ctm,
            train_dataset=train_dataset,
            gene_counts_common=gene_counts_common,
            model_name=args.model_name,
            train_order=train_order,
            data_name=data_name,
            num_topics=args.num_topics,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir
        )

    print("\n所有数据集训练完成!")

if __name__ == "__main__":
    main()
