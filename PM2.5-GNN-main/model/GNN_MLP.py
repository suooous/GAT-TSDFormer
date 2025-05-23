import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add
from torch.nn import functional as F
from torch.nn import Parameter
from model.PM25_GNN import GraphGNN  # 复用原有的GraphGNN模块

class PM25_GNN_MLP(nn.Module):
    """PM2.5预测模型，结合GNN和MLP进行时空预测
    
    将原模型中的GRU替换为MLP:
    xᵢᵗ = [ξᵢᵗ, ζᵢᵗ]  # MLP输入
    hᵢᵗ = MLP(xᵢᵗ)    # 时空特征
    X̂ᵢᵗ = Ω(hᵢᵗ)     # 最终预测
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(PM25_GNN_MLP, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim # 输入特征维度，为13
        self.hid_dim = 64  # 隐藏层维度
        self.out_dim = 1   # 输出维度（PM2.5预测值）
        self.gnn_out = 13  # GNN输出维度

        # 模型组件
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)  # 输入特征转换层，好像没有使用
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)  # 图神经网络
        
        # MLP
        self.mlp = Sequential(
            Linear(self.in_dim + self.gnn_out, self.hid_dim),  # 输入层
            Sigmoid(),
            Linear(self.hid_dim, self.hid_dim),  # 隐藏层
            Sigmoid(),
            Linear(self.hid_dim, self.hid_dim),  # 隐藏层
            Sigmoid()
        )
        
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)  # 输出层

    def forward(self, pm25_hist, feature):
        """前向传播
        
        Args:
            pm25_hist: 历史PM2.5数据 [batch_size, hist_len, num_nodes, 1]
            feature: 输入特征 [batch_size, hist_len + pred_len, num_nodes, feature_dim]
            
        Returns:
            pm25_pred: 预测的PM2.5浓度 [batch_size, pred_len, num_nodes, 1]
        """
        pm25_pred = []
        xn = pm25_hist[:, -1]  # 获取最后一个历史时间步的PM2.5值
        
        # 逐步预测
        for i in range(self.pred_len):
            # 合并当前PM2.5值和特征，在最终一个特征维度合并
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            # 通过GNN获取空间相关性 ζᵢᵗ
            xn_gnn = x
            xn_gnn = xn_gnn.contiguous() # 确保数据在内存中是连续的
            xn_gnn = self.graph_gnn(xn_gnn)
            
            # 合并节点表示和空间相关性 xᵢᵗ = [ξᵢᵗ, ζᵢᵗ]
            x = torch.cat([xn_gnn, x], dim=-1)
            
            # 通过MLP处理时空特征
            x = x.view(self.batch_size * self.city_num, -1)  # 展平
            x = self.mlp(x)
            xn = x.view(self.batch_size, self.city_num, self.hid_dim)
            
            # 生成预测值 X̂ᵢᵗ = Ω(hᵢᵗ)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)
        
        # 堆叠所有时间步的预测结果
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred 