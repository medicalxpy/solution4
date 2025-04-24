from model import CTM
from dataset import CTMDataset
from torch.utils.data import Subset
import numpy as np
import torch
from fusion_CTM import get_cellemb
import pickle
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.cell_embedding import generate_cell_embedding


adata_path1="/volume1/home/pxie/data/PBMC.h5ad"
adata_path2="/volume1/home/pxie/data/Cortex.h5ad"
gene_embedding_path="/volume1/home/pxie/data/embeddings/fused_gene_embedding.pkl"
model_name="gpt_base_fsuion"
cell_embeddings1, gene_counts_common1,_=generate_cell_embedding(
    adata_path=adata_path1,
    gene_embedding_path=gene_embedding_path)
cell_embeddings2, gene_counts_common2,_=generate_cell_embedding(
    adata_path=adata_path2,
    gene_embedding_path=gene_embedding_path
)
num_genes1 = gene_counts_common1.n_vars
num_genes2 = gene_counts_common2.n_vars
# 生成示例词汇表逆映射
id2token1 = {i: list(gene_counts_common1.var_names)[i] for i in range(num_genes1)}
cell_embeddings_numpy1 = cell_embeddings1.detach().numpy()
train_dataset1 = CTMDataset(X_contextual = cell_embeddings_numpy1,X_bow = gene_counts_common1.X.toarray(),idx2token=id2token1)
id2token2 = {i: list(gene_counts_common2.var_names)[i] for i in range(num_genes2)}
cell_embeddings_numpy2 = cell_embeddings2.detach().numpy()
train_dataset2 = CTMDataset(X_contextual = cell_embeddings_numpy2,X_bow = gene_counts_common2.X.toarray(),idx2token=id2token2)

lr=2e-3
num_topics = 50
num_epochs= 300
ctm = CTM(contextual_size=cell_embeddings1.shape[1],
            bow_size=num_genes1,
            n_components=num_topics,
            batch_size=64,
            device=device,
            lr=lr,
            num_epochs=num_epochs)
prior_mean,prior_variance=ctm.fit(train_dataset1,return_mean=False,patience=10)
ctm.save(model_dir=f'./results/checkpoint/',part_name=f"{model_name}_2_part1")
df_topics_per_cell1=pd.DataFrame()
gene_names_list=[]
topics_per_cell1 = ctm.get_thetas(train_dataset1)
df_topics_per_cell1= pd.DataFrame(topics_per_cell1)
output_file = f"/volume1/home/pxie/topic_model/solution4/results/{model_name}_2_PBMC_t{num_topics}_c.csv"
df_topics_per_cell1.to_csv(output_file, index=False)



ctm = CTM(contextual_size=cell_embeddings2.shape[1], 
          bow_size=num_genes2, 
          n_components=num_topics,
          batch_size=64,
          device=device,
          lr=lr,
          num_epochs=num_epochs,
          prior_mean=prior_mean,
          prior_variance=prior_variance)

prior_mean,prior_variance=ctm.fit(train_dataset2,return_mean=False,patience=10)
ctm.save(model_dir=f'./results/checkpoint/',part_name=f"{model_name}_2_part2")
df_topics_per_cell2=pd.DataFrame()
gene_names_list=[]
topics_per_cell2 = ctm.get_thetas(train_dataset2)
df_topics_per_cell2= pd.DataFrame(topics_per_cell2)
output_file = f"/volume1/home/pxie/topic_model/solution4/results/{model_name}_2_Cortex_t{num_topics}_c.csv"
df_topics_per_cell2.to_csv(output_file, index=False)