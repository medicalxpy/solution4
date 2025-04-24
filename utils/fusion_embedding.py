"""
基因嵌入融合模块

提供基因嵌入A和B的融合功能，以A为基础进行融合
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

class CustomFusionDataset(Dataset):
    """用于融合模型训练的数据集"""
    
    def __init__(self, embed_a, embed_b):
        """
        初始化融合数据集
        
        参数:
            embed_a: 基因嵌入A
            embed_b: 基因嵌入B
        """
        self.embed_a = embed_a
        self.embed_b = embed_b

    def __len__(self):
        """返回数据集大小"""
        return len(self.embed_a)

    def __getitem__(self, idx):
        """获取数据集项"""
        return self.embed_a[idx], self.embed_b[idx]


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模型"""
    
    def __init__(self, dim_a, dim_b, learning_rate=0.0001, device='cpu'):
        """
        初始化交叉注意力融合模型
        
        参数:
            dim_a: 嵌入A的维度
            dim_b: 嵌入B的维度
            learning_rate: 学习率
            device: 计算设备
        """
        super(CrossAttentionFusion, self).__init__()
        self.device = device
        self.query_layer = nn.Sequential(
            nn.Linear(dim_b, dim_a),
            nn.BatchNorm1d(dim_a),
            nn.LeakyReLU()
        ).to(self.device)
        self.key_layer = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.BatchNorm1d(dim_a),
            nn.LeakyReLU()
        ).to(self.device)
        self.value_layer = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.BatchNorm1d(dim_a),
            nn.LeakyReLU()
        ).to(self.device)

        self.embed_b_recon_func = nn.Linear(dim_a, dim_b).to(self.device)
        self.embed_a_recon_func = nn.Linear(dim_a, dim_a).to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.embed_a_cache = None
        self.embed_b_cache = None
        
    def get_fused_embed(self, embed_a, embed_b):
        """
        获取融合嵌入
        
        参数:
            embed_a: 嵌入A
            embed_b: 嵌入B
            
        返回:
            fused_embed: 融合后的嵌入
        """
        query = self.query_layer(embed_b)  # (num_genes, dim_a)
        key = self.key_layer(embed_a)      # (num_genes, dim_a)
        value = self.value_layer(embed_a)  # (num_genes, dim_a)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        fused_embed = torch.matmul(attention_weights, value)  # (num_genes, dim_a)
        
        return fused_embed

    def forward(self, embed_a, embed_b):
        """
        前向传播
        
        参数:
            embed_a: 嵌入A
            embed_b: 嵌入B
            
        返回:
            recon_loss: 重建损失
        """
        self.embed_a_cache = embed_a
        self.embed_b_cache = embed_b
        fused_embed = self.get_fused_embed(embed_a, embed_b)
        recon_embed_b = self.embed_b_recon_func(fused_embed)
        recon_embed_a = self.embed_a_recon_func(fused_embed)
        fusion_recon_loss_a = torch.mean((recon_embed_a-self.embed_a_cache)**2)
        fusion_recon_loss_b = torch.mean((recon_embed_b-self.embed_b_cache)**2)   
        recon_loss = fusion_recon_loss_a + fusion_recon_loss_b     
        return recon_loss
    
    def fit(self, dataset, batch_size=1024, epochs=10, show_progress=True):
        """
        训练模型
        
        参数:
            dataset: 数据集
            batch_size: 批大小
            epochs: 训练轮数
            show_progress: 是否显示进度条
        """
        self.train()  # 将模型设置为训练模式
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0.0

            # 针对 dataloader 的进度条
            batch_iter = dataloader
            if show_progress:
                batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
            
            for i, (embed_a_batch, embed_b_batch) in enumerate(batch_iter):
                embed_a_batch = embed_a_batch.to(self.device)
                embed_b_batch = embed_b_batch.to(self.device)

                self.optim.zero_grad()
                recon_loss = self.forward(embed_a_batch, embed_b_batch)
                recon_loss.backward()
                self.optim.step()

                running_loss += recon_loss.item()

                if show_progress:
                    batch_iter.set_postfix({"Batch Loss": f"{recon_loss.item():.4f}"})

            # 计算当前 epoch 的平均损失
            avg_loss = running_loss / len(dataloader)

            # 外层进度条，显示 epoch 级别的进度
            if show_progress:
                if 'epoch_pbar' not in locals():
                    epoch_pbar = tqdm(total=epochs, desc="Training Progress", unit="epoch")
                epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
                epoch_pbar.update(1)

        # 训练结束后关闭进度条
        if show_progress and 'epoch_pbar' in locals():
            epoch_pbar.close()


def fuse_gene_embeddings(embedding_a, embedding_b, device='cpu', batch_size=1024, 
                         epochs=300, learning_rate=0.0001, show_progress=True, 
                         save_path=None):
    """
    融合两个基因嵌入，以embedding_a为基础进行融合
    
    参数:
        embedding_a: DataFrame，基因嵌入A，索引为基因ID
        embedding_b: DataFrame，基因嵌入B，索引为基因ID
        device: 计算设备
        batch_size: 批大小
        epochs: 训练轮数
        learning_rate: 学习率
        show_progress: 是否显示进度条
        save_path: 保存融合后嵌入的路径，为None则不保存
        
    返回:
        fused_embedding: DataFrame，融合后的基因嵌入，与embedding_a具有相同的索引
    """
    # 1. 找出两个嵌入共有的基因
    common_genes = embedding_a.index.intersection(embedding_b.index)
    print(f"Embedding A中的基因数: {len(embedding_a)}")
    print(f"Embedding B中的基因数: {len(embedding_b)}")
    print(f"共有基因数: {len(common_genes)}")
    
    if len(common_genes) == 0:
        raise ValueError("两个嵌入之间没有共有基因！请检查基因ID格式是否匹配。")
    
    # 2. 准备共有基因的嵌入数据
    common_embed_a = embedding_a.loc[common_genes]
    common_embed_b = embedding_b.loc[common_genes]
    
    # 3. 转换为张量
    common_embed_a_tensor = torch.tensor(common_embed_a.values.astype(np.float32), dtype=torch.float32)
    common_embed_b_tensor = torch.tensor(common_embed_b.values.astype(np.float32), dtype=torch.float32)
    
    # 4. 创建数据集和模型
    fusion_dataset = CustomFusionDataset(common_embed_a_tensor, common_embed_b_tensor)
    fusion_model = CrossAttentionFusion(
        dim_a=common_embed_a.shape[1],
        dim_b=common_embed_b.shape[1],
        learning_rate=learning_rate,
        device=device
    )
    
    # 5. 训练模型
    fusion_model.fit(
        dataset=fusion_dataset,
        batch_size=batch_size,
        epochs=epochs,
        show_progress=show_progress
    )
    
    # 6. 获取融合后的嵌入
    fusion_model.to('cpu')  # 移至CPU进行推理
    fused_common_embed = fusion_model.get_fused_embed(common_embed_a_tensor, common_embed_b_tensor)
    fused_common_embed_np = fused_common_embed.detach().numpy()
    
    # 7. 创建融合后的完整嵌入（包括A中有但B中没有的基因）
    fused_embedding = embedding_a.copy()
    fused_embedding.loc[common_genes] = fused_common_embed_np
    
    # 8. 保存融合后的嵌入（如果需要）
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # 直接保存融合后的嵌入为pickle文件
        with open(save_path, 'wb') as f:
            pickle.dump(fused_embedding, f)
        print(f"融合后的基因嵌入已保存至: {save_path}")
    
    return fused_embedding


def load_embeddings(embedding_a_path, embedding_b_path=None, gene_map_path=None):
    """
    加载基因嵌入，并将索引标准化为ENSG格式
    
    参数:
        embedding_a_path: 基因嵌入A的文件路径
        embedding_b_path: 基因嵌入B的文件路径，为None则只加载A
        gene_map_path: 基因映射文件路径，默认为None则使用默认路径
        
    返回:
        embedding_a: DataFrame，基因嵌入A，索引为ENSG
        embedding_b: DataFrame或None，基因嵌入B，索引为ENSG
    """
    # 默认基因映射文件路径
    gene_map_path = gene_map_path or '/volume1/home/pxie/data/human_gene_map.csv'
    
    # 加载基因映射
    gene_map_df = pd.read_csv(gene_map_path)
    gene_map_df = gene_map_df[['hgnc_symbol', 'ensembl_gene_id']]
    gene_map_df.columns = ['HGNC.symbol', 'Gene.stable.ID']  # 重命名列
    
    # 创建从gene到ensemble ID的映射
    gene_to_ensemble = dict(zip(gene_map_df['HGNC.symbol'], gene_map_df['Gene.stable.ID']))
    
    # 加载嵌入A (GenePT)
    with open(embedding_a_path, 'rb') as f:
        embedding_a = pickle.load(f)
        
    # 处理GenePT格式
    if embedding_a_path.endswith('GenePT.pkl'):
        embedding_a = embedding_a.T
    
    # 确保embedding_a是DataFrame格式
    if not isinstance(embedding_a, pd.DataFrame):
        embedding_a = pd.DataFrame(embedding_a)
    
    # 如果索引不是ENSG格式，转换为ENSG格式
    if not all(str(idx).startswith('ENSG') for idx in embedding_a.index):
        # 找出在映射中存在的基因
        valid_genes = embedding_a.index.intersection(gene_to_ensemble.keys())
        filtered_embedding_a = embedding_a.loc[valid_genes]
        
        # 将索引替换为ensemble_id
        ensemble_ids = [gene_to_ensemble[gene] for gene in filtered_embedding_a.index]
        embedding_a = pd.DataFrame(filtered_embedding_a.values, index=ensemble_ids)
    
    # 如果提供了嵌入B的路径，加载嵌入B (Gformer)
    if embedding_b_path is not None:
        with open(embedding_b_path, 'rb') as f:
            embedding_b = pickle.load(f)
        
        # 处理Gformer格式
        if embedding_b_path.endswith('gene.pkl'):
            embedding_b.set_index(embedding_b.columns[0], inplace=True)
            embedding_b = embedding_b.groupby(embedding_b.index).mean()
        
        # 确保embedding_b是DataFrame格式
        if not isinstance(embedding_b, pd.DataFrame):
            embedding_b = pd.DataFrame(embedding_b)
        
        # 如果索引不是ENSG格式，转换为ENSG格式
        if not all(str(idx).startswith('ENSG') for idx in embedding_b.index):
            # 找出在映射中存在的基因
            valid_genes = embedding_b.index.intersection(gene_to_ensemble.keys())
            filtered_embedding_b = embedding_b.loc[valid_genes]
            
            # 将索引替换为ensemble_id
            ensemble_ids = [gene_to_ensemble[gene] for gene in filtered_embedding_b.index]
            embedding_b = pd.DataFrame(filtered_embedding_b.values, index=ensemble_ids)
        

        
        return embedding_a, embedding_b
    
    return embedding_a


def load_fused_embedding(fused_embedding_path):
    """
    加载融合后的基因嵌入
    
    参数:
        fused_embedding_path: 融合基因嵌入的文件路径
        
    返回:
        fused_embedding: DataFrame，融合后的基因嵌入
    """
    with open(fused_embedding_path, 'rb') as f:
        fused_embedding = pickle.load(f)
    
    return fused_embedding