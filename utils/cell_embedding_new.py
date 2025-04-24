"""
细胞嵌入生成模块 - 改进版

根据基因嵌入和基因表达数据生成细胞嵌入
改进：
1. 当嵌入数据中有而表达数据中没有的基因时，在表达数据中添加全0列
2. 当表达数据中有而嵌入数据中没有的基因时，删除这些基因
"""

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import os
import pickle
from scipy.sparse import issparse, csr_matrix
from utils.data_preprocess import preprocess_h5ad

def generate_cell_embedding(adata_path, gene_embedding_path, output_path=None):
    """
    根据基因嵌入和基因表达数据生成细胞嵌入

    参数:
        adata_path: AnnData对象或h5ad文件路径
        gene_embedding_path: 基因嵌入文件路径
        output_path: 输出路径，为None则不保存

    返回:
        cell_embeddings: 细胞嵌入张量
        gene_counts_common: 筛选后的AnnData对象
        common_genes: 共有基因列表
    """
    # 1. 加载数据
    adata = preprocess_h5ad(adata_path)
    if gene_embedding_path.endswith('.csv'):
        gene_embedding = pd.read_csv(gene_embedding_path, index_col=0)
    else:
        with open(gene_embedding_path, 'rb') as f:
            gene_embedding = pickle.load(f)

    # 2. 获取基因列表
    adata_genes = adata.var_names
    embedding_genes = gene_embedding.index

    # 3. 统计基因数量
    print(f"AnnData中的基因数: {len(adata_genes)}")
    print(f"基因嵌入中的基因数: {len(embedding_genes)}")

    # 4. 找出嵌入中有而表达数据中没有的基因
    genes_to_add = embedding_genes.difference(adata_genes)
    print(f"嵌入中有而表达数据中没有的基因数: {len(genes_to_add)}")

    # 5. 找出表达数据中有而嵌入中没有的基因
    genes_to_remove = adata_genes.difference(embedding_genes)
    print(f"表达数据中有而嵌入中没有的基因数: {len(genes_to_remove)}")

    # 6. 找出共有的基因
    common_genes = embedding_genes.intersection(adata_genes)
    print(f"共有基因数: {len(common_genes)}")

    if len(common_genes) == 0:
        raise ValueError("AnnData和基因嵌入之间没有共有基因！请检查基因ID格式是否匹配。")

    # 7. 处理表达数据
    # 7.1 先保留共有基因
    gene_counts_common = adata[:, common_genes].copy()

    # 7.2 为嵌入中有而表达数据中没有的基因添加全0列
    if len(genes_to_add) > 0:
        print("为嵌入中有而表达数据中没有的基因添加全0列...")

        # 获取细胞数量
        n_cells = gene_counts_common.n_obs

        # 创建全0矩阵
        if issparse(gene_counts_common.X):
            zeros_matrix = csr_matrix((n_cells, len(genes_to_add)))
        else:
            zeros_matrix = np.zeros((n_cells, len(genes_to_add)))

        # 创建新的var DataFrame
        new_var = pd.DataFrame(index=genes_to_add)

        # 如果原始var中有其他列，为新var添加相同的列
        for col in gene_counts_common.var.columns:
            if col == 'feature_name':
                # 对于feature_name列，使用基因ID作为特征名
                new_var[col] = genes_to_add.values
            else:
                # 对于其他列，填充NaN或适当的默认值
                new_var[col] = np.nan

        # 创建新的AnnData对象
        adata_zeros = sc.AnnData(X=zeros_matrix, var=new_var)
        adata_zeros.obs_names = gene_counts_common.obs_names

        # 合并两个AnnData对象
        gene_counts_common = sc.concat([gene_counts_common, adata_zeros], axis=1)

    # 8. 确保基因顺序与嵌入顺序一致
    gene_counts_common = gene_counts_common[:, embedding_genes]

    # 9. 将表达矩阵转换为张量
    if issparse(gene_counts_common.X):
        expression_matrix = torch.tensor(gene_counts_common.X.toarray(), dtype=torch.float32)
    else:
        expression_matrix = torch.tensor(gene_counts_common.X, dtype=torch.float32)

    # 10. 将基因嵌入转换为张量
    gene_embedding_tensor = torch.tensor(gene_embedding.values, dtype=torch.float32)

    # 11. 计算细胞嵌入
    cell_embeddings = torch.matmul(expression_matrix, gene_embedding_tensor)

    # 12. 保存细胞嵌入（如果需要）
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 将细胞嵌入转换为DataFrame
        cell_embeddings_df = pd.DataFrame(
            cell_embeddings.detach().numpy(),
            index=gene_counts_common.obs_names
        )

        # 保存为CSV文件
        cell_embeddings_df.to_csv(output_path)
        print(f"细胞嵌入已保存至: {output_path}")

    return cell_embeddings, gene_counts_common, embedding_genes


# 移除将嵌入添加到adata的函数


def example_workflow(adata_path, gene_embedding_path, output_path=None):
    """
    示例工作流：从文件加载数据，生成细胞嵌入，保存结果

    参数:
        adata_path: AnnData文件路径
        gene_embedding_path: 基因嵌入文件路径
        output_path: 输出路径

    返回:
        cell_embeddings: 细胞嵌入
        gene_counts_common: 处理后的AnnData对象
    """
    # 生成细胞嵌入
    cell_embeddings, gene_counts_common, _ = generate_cell_embedding(
        adata_path=adata_path,
        gene_embedding_path=gene_embedding_path,
        output_path=output_path
    )

    return cell_embeddings, gene_counts_common
