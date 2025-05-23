import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

import torch
import numpy as np
from model.PM25_GNN import GraphGNN,PM25_GNN
import matplotlib.pyplot as plt
from dataset import HazeData
from graph import Graph

def analyze_intermediate_outputs():
    """分析模型中的中间变量
    ξᵢᵗ: 节点表示 [X̂ᵢᵗ⁻¹, Pᵢᵗ]
    ζᵢᵗ: GNN输出的空间相关性
    xᵢᵗ: GRU输入 [ξᵢᵗ, ζᵢᵗ]
    """
    # 1. 初始化组件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = Graph(region='jjj')
    dataset = HazeData(graph, hist_len=1, pred_len=1, dataset_num=3, flag='Test', region='jjj')
    
    # 2. 初始化模型
    batch_size = 1
    in_dim = dataset.feature.shape[-1] + 1
    city_num = 13
    hist_len = 1
    pred_len = 1
    
    # 初始化完整模型
    model = PM25_GNN(
        hist_len=hist_len,
        pred_len=pred_len,
        in_dim=in_dim,
        city_num=city_num,
        batch_size=batch_size,
        device=device,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        wind_mean=dataset.wind_mean,
        wind_std=dataset.wind_std
    ).to(device)
    
    # 3. 准备数据
    pm25, feature, time_arr = dataset[0]
    pm25 = torch.from_numpy(pm25).float()
    feature = torch.from_numpy(feature).float()
    pm25 = pm25.unsqueeze(0)
    feature = feature.unsqueeze(0)
    
    # 4. 获取中间变量
    with torch.no_grad():
        # 获取ξᵢᵗ (节点表示)
        xi_t = torch.cat((pm25[:, 0], feature[:, 0]), dim=-1).to(device)
        
        # 获取ζᵢᵗ (GNN输出的空间相关性)
        zeta_t = model.graph_gnn(xi_t)
        
        # 获取xᵢᵗ (GRU输入)
        x_t = torch.cat([zeta_t, xi_t], dim=-1)
    
    # 5. 分析结果 (以北京为例)
    city_names = ["北京", "天津", "石家庄", "唐山", "秦皇岛", "邯郸", "保定", 
                 "张家口", "承德", "廊坊", "沧州", "衡水", "邢台"]
    
    print("\n=== 北京节点的中间变量分析 ===")
    
    # 分析ξᵢᵗ (节点表示)
    beijing_xi = xi_t[0, 0].cpu().numpy()
    print("\nξᵢᵗ (节点表示):")
    print(f"维度: {beijing_xi.shape}")
    print(f"PM2.5历史值: {beijing_xi[0]:.4f}")
    print("气象特征:")
    for i in range(1, len(beijing_xi)):
        print(f"特征 {i}: {beijing_xi[i]:.4f}")
    
    # 分析ζᵢᵗ (空间相关性)
    beijing_zeta = zeta_t[0, 0].cpu().numpy()
    print("\nζᵢᵗ (空间相关性):")
    print(f"维度: {beijing_zeta.shape}")
    print("与各城市的关联强度:")
    for i, city in enumerate(city_names):
        print(f"与{city}的关联: {beijing_zeta[i]:.4f}")
    
    # 分析xᵢᵗ (GRU输入)
    beijing_x = x_t[0, 0].cpu().numpy()
    print("\nxᵢᵗ (GRU输入):")
    print(f"维度: {beijing_x.shape}")
    print(f"空间相关性特征数: {len(beijing_zeta)}")
    print(f"节点表示特征数: {len(beijing_xi)}")
    

if __name__ == "__main__":
    analyze_intermediate_outputs() 