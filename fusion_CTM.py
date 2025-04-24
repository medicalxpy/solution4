import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocess import preprocess_h5ad
import scanpy as sc
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
device = torch.device("cuda:1")

def data_prepare(adata_path):
    adata = preprocess_h5ad(adata_path)
    print('成功读取h5ad数据')
    
    with open('/volume1/home/pxie/data/embeddings/GenePT.pkl', 'rb') as f:
        gene_embeddings_GPT = pickle.load(f)
        gene_embeddings_GPT=gene_embeddings_GPT.T
    with open('/volume1/home/pxie/data/embeddings/gene.pkl', 'rb') as f:
        gene_embeddings_Gformer = pickle.load(f)
        gene_embeddings_Gformer.set_index(gene_embeddings_Gformer.columns[0], inplace=True)
        gene_embeddings_Gformer = gene_embeddings_Gformer.groupby(gene_embeddings_Gformer.index).mean()

    # 创建gene_map_df
    if 'feature_name' in adata.var.columns:
        # 使用原有逻辑
        gene_map_df = pd.DataFrame()
        gene_map_df['HGNC.symbol'] = adata.var['feature_name']
        gene_map_df['Gene.stable.ID'] = adata.var.index
    else:
        # 读取外部映射文件
        gene_map_df = pd.read_csv('/volume1/home/pxie/data/human_gene_map.csv')
        gene_map_df = gene_map_df[['hgnc_symbol', 'ensembl_gene_id']]
        gene_map_df.columns = ['HGNC.symbol', 'Gene.stable.ID']  # 重命名列以匹配原有逻辑

    # # 保留 gene_embeddings_GPT 中存在的基因
    valid_genes = gene_embeddings_GPT.index.intersection(gene_map_df['HGNC.symbol'])
    filtered_gene_embeddings = gene_embeddings_GPT.loc[valid_genes]
    # # 重新筛选 gene_map_df 中存在的基因
    filtered_gene_map_df = gene_map_df[gene_map_df['HGNC.symbol'].isin(valid_genes)]
    # # 创建从 gene 到 ensemble ID 的映射
    gene_to_ensemble = dict(zip(filtered_gene_map_df['HGNC.symbol'], filtered_gene_map_df['Gene.stable.ID']))
    # # 将 gene_embeddings_GPT 的行索引替换为 ensemble_id
    ensemble_ids = [gene_to_ensemble[gene] for gene in filtered_gene_embeddings.index]
    gene_embeddings_with_ensemble = pd.DataFrame(filtered_gene_embeddings.values, index=ensemble_ids)
    common_ensemble_ids = gene_embeddings_with_ensemble.index.intersection(gene_embeddings_Gformer.index)

    gene_id_embeddings = gene_embeddings_with_ensemble.loc[common_ensemble_ids] 
    gene_embeddings = gene_embeddings_Gformer.loc[common_ensemble_ids]  

    # expression 的 embedding
    adata_var_index = adata.var.index
    gene_embeddings_index = gene_embeddings.index
    common_genes = gene_embeddings_index.intersection(adata_var_index)
    gene_counts_common = adata[:, common_genes]
    gene_id_embeddings = gene_embeddings.loc[common_genes] # 
    gene_embeddings = gene_embeddings_Gformer.loc[common_genes] 

    gene_embeddings_array = gene_embeddings.values.astype(np.float32)
    gene_id_embeddings_array = gene_id_embeddings.values.astype(np.float32)

    
    gene_embeddings_tensor = torch.tensor(gene_embeddings_array, dtype=torch.float32)
    gene_id_embeddings_tensor = torch.tensor(gene_id_embeddings_array, dtype=torch.float32)

    bow = gene_counts_common.X.toarray()
    bow=torch.tensor(bow,dtype=torch.float32)
    return gene_id_embeddings_tensor, gene_embeddings_tensor ,gene_counts_common,bow

class CustomFusionDataset(Dataset):
    def __init__(self, cell_embed, gene_embed):
        self.cell_embed = cell_embed
        self.gene_embed = gene_embed

    def __len__(self):
        return len(self.cell_embed)

    def __getitem__(self, idx):
        return self.cell_embed[idx], self.gene_embed[idx]
    
class CrossAttentionFusion(nn.Module):
    def __init__(self,embedding_dim_cell,embedding_dim_gene,learning_rate = 0.0001,device='cpu'):
        super(CrossAttentionFusion, self).__init__()
        self.device = device
        self.query_layer = nn.Sequential(
            nn.Linear(embedding_dim_gene, embedding_dim_cell),
            nn.BatchNorm1d(embedding_dim_cell),
            nn.LeakyReLU()
        ).to(self.device)
        self.key_layer = nn.Sequential(
            nn.Linear(embedding_dim_cell, embedding_dim_cell),
            nn.BatchNorm1d(embedding_dim_cell),
            nn.LeakyReLU()
        ).to(self.device )
        self.value_layer = nn.Sequential(
            nn.Linear(embedding_dim_cell, embedding_dim_cell),
            nn.BatchNorm1d(embedding_dim_cell),
            nn.LeakyReLU()
        ).to(self.device )

        self.gene_embed_recon_func = nn.Linear(embedding_dim_cell,embedding_dim_gene).to(self.device)
        self.cell_embed_recon_func = nn.Linear(embedding_dim_cell,embedding_dim_cell).to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.cell_rho = None
        self.rho = None
        
    def get_fused_embed(self, cell_embedding, gene_embedding):
        query = self.query_layer(gene_embedding)  # (num_genes, embedding_dim_cell)
        key = self.key_layer(cell_embedding)  # (num_cells, embedding_dim_cell)
        value = self.value_layer(cell_embedding)  # (num_cells, embedding_dim_cell)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        fused_embed = torch.matmul(attention_weights, value)  # (num_genes, embedding_dim_cell)
        
        return fused_embed

    def forward(self,cell_embed,gene_embed):
        self.cell_rho = cell_embed
        self.rho = gene_embed
        fused_embed = self.get_fused_embed(cell_embed,gene_embed)
        recon_gene_embed = self.gene_embed_recon_func(fused_embed)
        recon_cell_embed = self.cell_embed_recon_func(fused_embed)
        fusion_recon_loss_cell = torch.mean((recon_cell_embed-self.cell_rho)**2)
        fusion_recon_loss_gene = torch.mean((recon_gene_embed-self.rho)**2)   
        recon_loss = fusion_recon_loss_cell + fusion_recon_loss_gene     
        return recon_loss
    
    def fit(self, dataset, batch_size=1024, epochs=10, show_progress=False):
        self.train()  # 将模型设置为训练模式
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0.0

            # 针对 dataloader 的进度条
            batch_iter = dataloader
            if show_progress:
                batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
            
            for i, (cell_batch, gene_batch) in enumerate(batch_iter):
                cell_batch, gene_batch = cell_batch.to(self.device), gene_batch.to(self.device)

                self.optim.zero_grad()
                recon_loss = self.forward(cell_batch, gene_batch)
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


def get_cellemb(adata_path,device,batch_size=1024,epochs = 300,learning_rate = 0.0001):
    gene_id_embeddings_tensor,gene_embeddings_tensor,gene_counts_common,bow = data_prepare(adata_path)
    fusion_dataset = CustomFusionDataset(gene_id_embeddings_tensor,gene_embeddings_tensor)
    fusion_model = CrossAttentionFusion(gene_id_embeddings_tensor.shape[1],gene_embeddings_tensor.shape[1],learning_rate=learning_rate,device=device)
    fusion_model.fit(fusion_dataset,batch_size=batch_size,epochs=epochs)
    fusion_model.to('cpu')
    gene_fused_embed = fusion_model.get_fused_embed(gene_id_embeddings_tensor,gene_embeddings_tensor)
    cell_embeddings = torch.matmul(bow, gene_fused_embed) # cancer dataset bow:183326*42055
    return cell_embeddings,gene_counts_common