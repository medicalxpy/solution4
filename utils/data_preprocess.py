import argparse
import scanpy as sc
import pandas as pd
import re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

def check_if_all_strings(column):
    """
    检查给定列中前10个元素是否全部都是字符串。
    """
    if isinstance(column.dtype, pd.CategoricalDtype):
        column = column.astype(str)
    return column.head(10).apply(lambda x: isinstance(x, str)).all()

def remove_version_number(ensemble_id):
    """
    移除 Ensemble 基因 ID 中的版本号（形如 “.数字”）。
    例如：ENSG000001234.5 -> ENSG000001234
    """
    return re.sub(r'\.\d+$', '', ensemble_id)

def preprocess_h5ad(file_input_path,
                    cell_ge_expr_threshold=1000,
                    gene_num_threshold=100,
                    normalization=True,
                    log1p=True,
                    cell_type_threshold=100,
                    cell_type_label='cell_type'):
    """
    读取并预处理 h5ad 文件的 AnnData 对象。
    
    参数：
    - file_input_path: h5ad 文件的路径
    - cell_ge_expr_threshold: 细胞基因表达阈值（此处暂未使用）
    - gene_num_threshold: 每个细胞的最少基因数过滤阈值
    - normalization: 是否进行归一化
    - log1p: 是否进行 log1p 变换
    - cell_type_threshold: cell_type 的过滤阈值（过滤数量过少的细胞类型）
    - cell_type_label: 在 obs 中对应细胞类型的列名

    返回：
    - 预处理后的 AnnData 对象
    """
    # 1) 读取 h5ad 文件
    adata = sc.read_h5ad(file_input_path)

    # 2) 过滤细胞：去除基因数少于 gene_num_threshold 的细胞
    sc.pp.filter_cells(adata, min_genes=gene_num_threshold)

    # 3) 若 obs 中存在 cell_type_label 列，则过滤数量过少的细胞类型
    if cell_type_label in adata.obs:
        cell_types = adata.obs[cell_type_label]
        type_num = cell_types.value_counts()
        # 仅保留大于等于阈值的细胞类型
        adata = adata[adata.obs[cell_type_label].isin(type_num[type_num >= cell_type_threshold].index)]
    else:
        print(f"Warning: Column '{cell_type_label}' not found in obs. Skipping cell type filtering.")

    gene_map_df = pd.read_csv('/volume1/home/pxie/data/human_gene_map.csv')
    # 只保留需要的两列
    gene_map_df = gene_map_df[['hgnc_symbol', 'ensembl_gene_id']]

    symbol_to_ensg = dict(zip(gene_map_df['hgnc_symbol'], gene_map_df['ensembl_gene_id']))
    ensg_to_symbol = dict(zip(gene_map_df['ensembl_gene_id'], gene_map_df['hgnc_symbol']))

    cleaned_var_names = [remove_version_number(x) for x in adata.var_names]
    adata.var_names = cleaned_var_names  # 更新到 var_names

    if all([name.startswith('ENSG') for name in adata.var_names]):
        feature_names = []
        valid_indices = []
        for i, ensg_id in enumerate(adata.var_names):
            if ensg_id in ensg_to_symbol and pd.notnull(ensg_to_symbol[ensg_id]):
                feature_names.append(ensg_to_symbol[ensg_id])
                valid_indices.append(i)
            else:
                feature_names.append("unknown")
                valid_indices.append(i)
        adata.var["feature_name"] = feature_names
        adata.var["ensembl_id"] = adata.var_names.astype(str)

    else:
        mapped_indices = []
        mapped_ensg_list = []
        for i, old_name in enumerate(adata.var_names):
            if old_name in symbol_to_ensg:
                ensg_id = symbol_to_ensg[old_name]
                # 去除版本号
                ensg_id = remove_version_number(ensg_id)
                mapped_indices.append(i)
                mapped_ensg_list.append(ensg_id)
            else:
                # 不在映射中就丢弃
                pass

        adata = adata[:, mapped_indices].copy()

        adata.var_names = mapped_ensg_list
        adata.var["ensembl_id"] = adata.var_names.astype(str)
        original_features = np.array(cleaned_var_names)  # 筛选前的所有基因
        adata.var["feature_name"] = [original_features[idx] for idx in mapped_indices]


    # 5) 如果需要，做归一化 & log1p
    if normalization:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log1p:
        sc.pp.log1p(adata)

    # 6) 识别高变基因（HVG）
    sc.pp.highly_variable_genes(adata, flavor='seurat')

    # 7) 对非高变基因位置全置零（仅保留高变基因的表达）
    if 'highly_variable' in adata.var:
        non_hvg_indices = ~adata.var['highly_variable'].values
        mask = np.ones(adata.shape[1], dtype=bool)
        mask[non_hvg_indices] = False

        if issparse(adata.X):
            # 稀疏矩阵做法
            mask_sparse = csr_matrix(mask.astype(int))
            adata.X = adata.X.multiply(mask_sparse)
        else:
            # 稠密矩阵转换为稀疏矩阵再做乘法
            adata.X = csr_matrix(adata.X)
            mask_sparse = csr_matrix(mask.astype(int))
            adata.X = adata.X.multiply(mask_sparse)
    else:
        print("Warning: Column 'highly_variable' not found in var. Skipping HVG masking.")   

        
    adata.var = adata.var[['feature_name', 'ensembl_id']]
    adata.var_names = adata.var_names.astype(str)
    adata.var['feature_name'] = adata.var['feature_name'].astype(str)
    adata.var['ensembl_id'] = adata.var['ensembl_id'].astype(str)
    return adata