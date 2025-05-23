import torch
from torch import nn
from model.PM25_GNN import GraphGNN

class PM25_GNN_Direct(nn.Module):
    """PM2.5预测模型，直接使用GNN输出进行预测"""

    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(PM25_GNN_Direct, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim  # 输入特征维度
        self.gnn_out = 13     # GNN输出维度
        self.out_dim = 1      # 输出维度（PM2.5预测值）

        # 图神经网络
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        
        # 输出层
        self.fc_out = nn.Linear(self.gnn_out, self.out_dim)

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
            # 合并当前PM2.5值和特征
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            # 通过GNN获取空间相关性
            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()  # 确保数据在内存中是连续的
            xn_gnn = self.graph_gnn(xn_gnn)
            
            # 直接使用GNN输出进行预测
            xn = self.fc_out(xn_gnn)
            pm25_pred.append(xn)
        
        # 堆叠所有时间步的预测结果
        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred 