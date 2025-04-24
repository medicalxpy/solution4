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


adata_path="/volume1/home/pxie/data/combined_data.h5ad"
gene_embedding_path="/volume1/home/pxie/data/embeddings/fused_gene_embedding.pkl"
model_name="gpt_base_fsuion"
cell_embeddings, gene_counts_common,_=generate_cell_embedding(
    adata_path=adata_path,
    gene_embedding_path=gene_embedding_path)

num_genes = gene_counts_common.n_vars
# 生成示例词汇表逆映射
id2token = {i: list(gene_counts_common.var_names)[i] for i in range(num_genes)}
cell_embeddings_numpy = cell_embeddings.detach().numpy()
train_dataset = CTMDataset(X_contextual = cell_embeddings_numpy,X_bow = gene_counts_common.X.toarray(),idx2token=id2token)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_topics = 50
ctm = CTM(contextual_size=cell_embeddings.shape[1], 
          bow_size=num_genes, 
          n_components=num_topics,
          batch_size=64,
          device=device,
          lr=1e-4,
          num_epochs=100)

prior_mean=ctm.fit(train_dataset,patience=10)
ctm.save(model_dir=f'./results/checkpoint/',part_name=f"{model_name}_1")
df_topics_per_cell1=pd.DataFrame()
gene_names_list=[]
topics_per_cell1 = ctm.get_thetas(train_dataset)
df_topics_per_cell1= pd.DataFrame(topics_per_cell1)
output_file = f"/volume1/home/pxie/topic_model/solution4/results/{model_name}_1_t{num_topics}_c.csv"
df_topics_per_cell1.to_csv(output_file, index=False)

from model import CTM

ctm = CTM(contextual_size=cell_embeddings.shape[1], 
          bow_size=num_genes, 
          n_components=num_topics,
          batch_size=64,
          device=device,
          lr=1e-4,
          num_epochs=100)

prior_mean=ctm.fit(train_dataset,patience=10)
ctm.save(model_dir=f'./results/checkpoint/',part_name=f"{model_name}_2")
df_topics_per_cell2=pd.DataFrame()
gene_names_list=[]
topics_per_cell2 = ctm.get_thetas(train_dataset)
df_topics_per_cell2= pd.DataFrame(topics_per_cell2)
output_file = f"/volume1/home/pxie/topic_model/solution4/results/{model_name}_2_t{num_topics}_c.csv"
df_topics_per_cell2.to_csv(output_file, index=False)