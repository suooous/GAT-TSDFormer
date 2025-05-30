import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add  # PyG库中的散点聚合函数
from torch.nn import functional as F
from torch.nn import Parameter


class GraphGNN(nn.Module):
    """知识增强的图神经网络模块 (Knowledge-enhanced GNN)
    
    实现了论文中的公式(7):
    ξᵢᵗ = [X̂ᵢᵗ⁻¹, Pᵢᵗ]  # 节点表示
    eᵢ→ⱼᵗ = Ψ([ξⱼᵗ, ξᵢᵗ, Qᵢ→ⱼᵗ])  # 边表示
    ζᵢᵗ = Φ(Σⱼ∈N(i)(eⱼ→ᵢᵗ - eᵢ→ⱼᵗ))  # 空间相关性
    """
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        """
        Args:
            device: 计算设备（CPU/GPU）
            edge_index: 边的连接关系 [2, num_edges]
            edge_attr: 边的属性（距离、方向等）[num_edges, 2]
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            wind_mean: 风速风向的均值
            wind_std: 风速风向的标准差
        """
        super(GraphGNN, self).__init__()
        self.device = device
        # 边的连接关系和属性
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        # 边属性标准化
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        # 可学习的权重参数
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        # 风速风向的统计值
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        
        # 边特征处理MLP
        e_h = 32  # 隐藏层维度
        e_out = 30  # 输出维度
        n_out = out_dim
        self.edge_mlp = Sequential(
            Linear(in_dim * 2 + 2 + 1, e_h),  # 输入：源节点特征 + 目标节点特征 + 边属性 + 边权重
            Sigmoid(),
            Linear(e_h, e_out),
            Sigmoid(),
        )
        # 节点特征处理MLP
        self.node_mlp = Sequential(
            Linear(e_out, n_out),
            Sigmoid(),
        )

    def forward(self, x):
        """前向传播，实现论文中的消息传递机制
        
        Args:
            x: 输入特征 [batch_size, num_nodes, feature_dim]，对应论文中的 ξᵢᵗ
            
        Returns:
            out: 空间相关性特征 [batch_size, num_nodes, out_dim]，对应论文中的 ζᵢᵗ
        """
        # 确保所有张量在正确的设备上
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)

        # 获取边的源节点和目标节点
        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]  # 源节点特征
        node_target = x[:, edge_target]  # 目标节点特征

        # 处理风速风向特征
        src_wind = node_src[:,:,-2:] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        src_wind_speed = src_wind[:, :, 0]  # 风速
        src_wind_direc = src_wind[:,:,1]  # 风向
        
        # 处理边属性
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:,:,0]  # 城市间距离
        city_direc = self.edge_attr_[:,:,1]  # 城市间方向

        # 计算考虑风向的边权重
        theta = torch.abs(city_direc - src_wind_direc)  # 风向与城市连线方向的夹角
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)  # 边权重计算
        edge_weight = edge_weight.to(self.device)
        
        # 准备边特征
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:,:,None]], dim=-1)

        # 边特征处理
        out = self.edge_mlp(out)
        # 消息聚合：收集传入和传出的消息
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))  # 入度消息
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # 出度消息

        # 合并消息并进行节点特征转换
        out = out_add + out_sub
        out = self.node_mlp(out)

        return out


class MTGNN(nn.Module):
    """PM2.5预测模型，结合GNN和GRU进行时空预测
    
    实现了论文中的公式(8)和(9):
    xᵢᵗ = [ξᵢᵗ, ζᵢᵗ]  # GRU输入
    hᵢᵗ = GRU(xᵢᵗ, hᵢᵗ⁻¹)  # 时空特征
    X̂ᵢᵗ = Ω(hᵢᵗ)  # 最终预测
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        """
        Args:
            hist_len: 历史序列长度
            pred_len: 预测序列长度
            in_dim: 输入特征维度
            city_num: 城市数量
            batch_size: 批次大小
            device: 计算设备
            edge_index: 边的连接关系
            edge_attr: 边的属性
            wind_mean: 风速风向均值
            wind_std: 风速风向标准差
        """
        super(MTGNN, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64  # GRU隐藏层维度
        self.out_dim = 1   # 输出维度（PM2.5预测值）
        self.gnn_out = 13  # GNN输出维度

        # 模型组件
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)  # 输入特征转换层
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)  # 图神经网络
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)  # GRU单元
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
        # 初始化隐藏状态
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        # 获取最后一个历史时间步的PM2.5值作为初始输入
        xn = pm25_hist[:, -1]
        
        # 逐步预测
        for i in range(self.pred_len):
            # 合并当前PM2.5值和特征
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            # 通过GNN获取空间相关性 ζᵢᵗ
            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)
            
            # 合并节点表示和空间相关性 xᵢᵗ = [ξᵢᵗ, ζᵢᵗ]
            x = torch.cat([xn_gnn, x], dim=-1)

            # 通过GRU处理时序依赖 hᵢᵗ = GRU(xᵢᵗ, hᵢᵗ⁻¹)
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            
            # 生成预测值 X̂ᵢᵗ = Ω(hᵢᵗ)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        # 堆叠所有时间步的预测结果
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred
