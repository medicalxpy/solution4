"""
细胞嵌入生成模块

根据基因嵌入和基因表达数据生成细胞嵌入
"""

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import os
import pickle
from scipy.sparse import issparse
from utils.data_preprocess import preprocess_h5ad

def generate_cell_embedding(adata_path, gene_embedding_path, output_path=None):
    """
    根据基因嵌入和基因表达数据生成细胞嵌入
    
    参数:
        adata: AnnData对象或h5ad文件路径
        gene_embedding: DataFrame格式的基因嵌入，索引为基因ID
        output_path: 输出路径，为None则不保存
        
    返回:
        cell_embeddings: 细胞嵌入张量
        gene_counts_common: 筛选后的AnnData对象
        common_genes: 共有基因列表
    """
    adata = preprocess_h5ad(adata_path)
    if gene_embedding_path.endswith('.csv'):
        gene_embedding = pd.read_csv(gene_embedding_path, index_col=0)
    else:
        with open(gene_embedding_path, 'rb') as f:
            gene_embedding = pickle.load(f)
    
    
    # 2. 找出adata和基因嵌入中共有的基因
    adata_genes = adata.var_names
    embedding_genes = gene_embedding.index
    
    common_genes = embedding_genes.intersection(adata_genes)
    
    print(f"AnnData中的基因数: {len(adata_genes)}")
    print(f"基因嵌入中的基因数: {len(embedding_genes)}")
    print(f"共有基因数: {len(common_genes)}")
    
    if len(common_genes) == 0:
        raise ValueError("AnnData和基因嵌入之间没有共有基因！请检查基因ID格式是否匹配。")
    
    # 3. 筛选共有基因
    gene_counts_common = adata[:, common_genes]
    filtered_gene_embedding = gene_embedding.loc[common_genes]
    
    # 4. 将表达矩阵转换为张量
    if issparse(gene_counts_common.X):
        expression_matrix = torch.tensor(gene_counts_common.X.toarray(), dtype=torch.float32)
    else:
        expression_matrix = torch.tensor(gene_counts_common.X, dtype=torch.float32)
    
    # 5. 将基因嵌入转换为张量
    gene_embedding_tensor = torch.tensor(filtered_gene_embedding.values, dtype=torch.float32)
    
    # 6. 计算细胞嵌入
    cell_embeddings = torch.matmul(expression_matrix, gene_embedding_tensor)
    
    # 7. 保存细胞嵌入（如果需要）
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
    
    return cell_embeddings, gene_counts_common, common_genes

