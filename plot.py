import os
import pandas as pd
import numpy as np
from data_preprocess import preprocess_h5ad
import scanpy as sc
import matplotlib.pyplot as plt
import glob

def get_data_path(csv_filename):
    """
    根据csv文件名判断对应的数据集路径
    """
    if "PBMC" in csv_filename:
        return "/volume1/home/pxie/data/PBMC.h5ad"
    elif "Cortex" in csv_filename:
        return "/volume1/home/pxie/data/Cortex.h5ad"
    else:
        return "/volume1/home/pxie/data/combined_data.h5ad"

def process_and_plot(csv_path, output_dir="figures"):
    """
    处理单个csv文件并生成对应的热图
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）用于保存图像
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 加载数据
    print(f"Loading CSV: {csv_path}")
    cell_x_topic_matrix = pd.read_csv(csv_path)
    
    # 确保数据为数值型且无NaN
    cell_x_topic_matrix = cell_x_topic_matrix.fillna(0)
    
    print(f"Loading h5ad: {get_data_path(csv_path)}")
    data_path = get_data_path(csv_path)
    adata = preprocess_h5ad(data_path)
    
    # 创建新的AnnData对象用于绘图
    print("Creating AnnData object")
    adata1 = sc.AnnData(
        X=cell_x_topic_matrix.values.astype(np.float32),
        obs=pd.DataFrame(index=cell_x_topic_matrix.index)
    )
    
    # 确保cell_type数据正确对齐
    adata1.obs['cell_type'] = pd.Categorical(adata.obs['cell_type'])
    
    # 设置主题名称
    topic_names = [f'Topic_{i+1}' for i in range(cell_x_topic_matrix.shape[1])]
    adata1.var_names = topic_names
    adata1.var_names_make_unique()
    
    print("Generating heatmap")
    # 生成热图
    plt.figure(figsize=(30, 10))
    try:
        sc.pl.heatmap(adata1, 
                      var_names=topic_names,
                      groupby="cell_type",
                      swap_axes=True,
                      cmap='RdBu_r',
                      standard_scale='var',
                      figsize=(30, 10),
                      show=False)  # 设置show=False以便保存
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{filename}_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved heatmap: {output_path}")
        
    except Exception as e:
        plt.close()
        print(f"Error details:")
        print(f"Data shape: {adata1.X.shape}")
        print(f"Data type: {adata1.X.dtype}")
        print(f"Contains NaN: {np.isnan(adata1.X).any()}")
        print(f"Value range: [{np.min(adata1.X)}, {np.max(adata1.X)}]")
        print(f"Number of cell types: {len(adata1.obs['cell_type'].unique())}")
        raise

def main():
    # 设置目录路径
    model="solution4"
    results_dir = f"/volume1/home/pxie/topic_model/{model}/results"
    output_dir = f"/volume1/home/pxie/topic_model/{model}/results/figures"
    
    # 获取所有csv文件
    csv_files = glob.glob(os.path.join(results_dir, "*_t50_c.csv"))
    
    # 处理每个文件
    for csv_path in csv_files:
        print(f"\nProcessing file: {csv_path}")
        try:
            process_and_plot(csv_path, output_dir)
        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
