import os
import sys
# 添加项目根目录到系统路径
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

import torch
import numpy as np
from PM25_GNN import GraphGNN
import matplotlib.pyplot as plt
from dataset import HazeData
from graph import Graph

def analyze_beijing_gnn_output():
    """分析北京节点的GNN输出"""
    # 1. 初始化组件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = Graph(region='jjj')
    dataset = HazeData(graph, hist_len=1, pred_len=1, dataset_num=3, flag='Test', region='jjj')
    
    # 2. 初始化GNN
    batch_size = 1
    in_dim = dataset.feature.shape[-1] + 1
    out_dim = 13
    
    gnn = GraphGNN(
        device=device,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        in_dim=in_dim,
        out_dim=out_dim,
        wind_mean=dataset.wind_mean,
        wind_std=dataset.wind_std
    ).to(device)
    
    # 3. 准备数据
    pm25, feature, time_arr = dataset[0]
    pm25 = torch.from_numpy(pm25).float()
    feature = torch.from_numpy(feature).float()
    pm25 = pm25.unsqueeze(0)
    feature = feature.unsqueeze(0)
    
    x = torch.cat((pm25[:, 0], feature[:, 0]), dim=-1).to(device)
    
    # 4. 获取GNN输出
    with torch.no_grad():
        gnn_output = gnn(x)
    
    # 5. 分析北京的输出（北京是索引0）
    print(gnn_output)
    print(gnn_output[0, 0])
    beijing_output = gnn_output[0, 0].cpu().numpy()
    print(beijing_output)
    print("\n=== 北京节点GNN输出分析 ===")
    print(f"输出特征维度: {beijing_output.shape}")
    print(f"平均值: {beijing_output.mean():.4f}")
    print(f"标准差: {beijing_output.std():.4f}")
    print(f"最大值: {beijing_output.max():.4f}")
    print(f"最小值: {beijing_output.min():.4f}")
    print("\n特征值详情:")
    for i, value in enumerate(beijing_output):
        print(f"特征 {i+1}: {value:.4f}")
    
    # 6. 分析与北京相连的边
    edge_src, edge_target = gnn.edge_index
    beijing_edges_out = (edge_src == 0).nonzero().squeeze()  # 从北京出发的边
    beijing_edges_in = (edge_target == 0).nonzero().squeeze()  # 指向北京的边
    
    # 获取边权重
    node_src = x[:, edge_src]
    src_wind = node_src[:,:,-2:] * gnn.wind_std[None,None,:] + gnn.wind_mean[None,None,:]
    src_wind_speed = src_wind[:, :, 0]
    src_wind_direc = src_wind[:,:,1]
    
    city_dist = gnn.edge_attr_[:,:,0]
    city_direc = gnn.edge_attr_[:,:,1]
    theta = torch.abs(city_direc - src_wind_direc)
    edge_weight = torch.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)
    
    # 打印相连城市的信息
    city_names = ["北京", "天津", "石家庄", "唐山", "秦皇岛", "邯郸", "保定", 
                 "张家口", "承德", "廊坊", "沧州", "衡水", "邢台"]
    
    print("\n=== 北京的边连接信息 ===")
    print("出边（从北京到其他城市）:")
    for idx in beijing_edges_out:
        target_city = edge_target[idx].item()
        weight = edge_weight[0, idx].item()
        print(f"到 {city_names[target_city]}: 权重 = {weight:.4f}")
    
    print("\n入边（从其他城市到北京）:")
    for idx in beijing_edges_in:
        source_city = edge_src[idx].item()
        weight = edge_weight[0, idx].item()
        print(f"从 {city_names[source_city]}: 权重 = {weight:.4f}")
    
if __name__ == "__main__":
    analyze_beijing_gnn_output()

print(1)