import torch
from torch import nn
import torch.nn.functional as F
from model.PM25_GNN import GraphGNN
import math

epsilon = 0.0001  # 数值稳定性常数

class TimeSeriesModule(nn.Module):
    """使用SSCNN方式的时间序列建模模块"""
    def __init__(self, input_dim, hidden_dim, num_nodes, hist_len, pred_len, cycle_len, short_period_len):
        super(TimeSeriesModule, self).__init__()

        # 这个cycle_len 是周期的长度，影响后面的季节性建模的周期划分，hist_len和pred_len的是cycle_len的整数倍
        
        # input_dim: 输入特征维度，为26,self.gnn_out+self.in_dim=26
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # 隐藏层维度，为64    
        self.num_nodes = num_nodes  # 节点数量，为13 
        self.hist_len = hist_len  # 历史序列长度，
        self.pred_len = pred_len  # 预测长度
        self.cycle_len = cycle_len  # 周期长度，用于季节性的分割
        self.short_period_len = short_period_len  # 短期依赖长度，一天的数据点数
        

        # 输入投影层
        self.input_projection = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        # 注意力参数
        self.I_st = nn.Parameter(torch.zeros(1, 1, 1, self.short_period_len))  # 短期注意力
        self.E_st = nn.Parameter(torch.zeros(pred_len, 1, 1, self.short_period_len))  # 短期预测注意力
        self.I_se = nn.Parameter(torch.zeros(hist_len // cycle_len, hist_len // cycle_len, 1, 1))  # 季节性注意力
        self.E_se = nn.Parameter(torch.zeros(pred_len // cycle_len + 1, hist_len // cycle_len, 1, 1))  # 季节性预测注意力
        
        # 特征融合层
        self.fusion_conv = nn.Conv2d(
            in_channels=hidden_dim * 6,  # 长期、季节性、短期各2个特征(残差和均值)
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        # 输出层
        self.output_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=1
        )

        # 添加多项式特征转换和卷积层
        self.kernel_size = 3
        self.poly = nn.Conv2d(hidden_dim * 6, hidden_dim * 2, 1)  # 多项式特征转换
        self.residual_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, self.kernel_size)  # 残差卷积
        self.skip_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)  # 跳跃连接卷积

    def forward(self, x):
        """
        输入 x: [batch_size, hist_len, num_nodes, feature_dim]
        输出: [batch_size, pred_len, num_nodes, 1]
        """
        b, t, n, c = x.shape
        x = x.permute(0, 3, 2, 1)  # [batch_size, feature_dim, num_nodes, seq_len]
        
        # 首先通过输入投影层调整特征维度
        x = self.input_projection(x)  # [batch_size, hidden_dim, num_nodes, seq_len]
        
        # 然后再进行季节性建模
        x_reshaped = x.reshape(b, self.hidden_dim, n, -1, self.cycle_len)
        
        # 存储所有特征和预测
        s = []  # 历史特征
        hat_s = []  # 预测特征
        
        # 1. 长期依赖建模
        mu_lt = x.mean(-1, keepdim=True).repeat(1, 1, 1, t)
        square_mu_lt = (x ** 2).mean(-1, keepdim=True).repeat(1, 1, 1, t)
        var_lt = square_mu_lt - mu_lt ** 2 + epsilon
        r_lt = (x - mu_lt) / (var_lt ** 0.5)
        
        hat_mu_lt = mu_lt[..., -1:].repeat(1, 1, 1, self.pred_len)
        hat_r_lt = r_lt[..., -1:].repeat(1, 1, 1, self.pred_len)
        s.extend([r_lt, mu_lt])
        hat_s.extend([hat_r_lt, hat_mu_lt])
        
        # 2. 季节性建模
        I_se = torch.softmax(self.I_se, dim=1)  # [hist_len//cycle_len, hist_len//cycle_len, 1, 1]
        E_se = torch.softmax(self.E_se, dim=1)  # [pred_len//cycle_len + 1, hist_len//cycle_len, 1, 1]
        
        # 重塑输入以进行季节性建模
        # 将序列分成多个周期
        x_reshaped = x_reshaped.permute(0, 1, 3, 4, 2)  # [batch, hidden_dim, num_cycles, cycle_len, num_nodes]
        
        # 计算季节性均值
        mu_se = F.conv2d(
            x_reshaped.reshape(b * self.hidden_dim, -1, self.cycle_len, n),  # [b*h, num_cycles, cycle_len, n]
            I_se.transpose(-1, -2)  # 调整卷积核维度
        )
        mu_se = mu_se.reshape(b, self.hidden_dim, n, t)
        
        # 计算季节性二阶矩
        # 计算季节性二阶矩: E[x²]_se = ∑(α_se_i * x_i²), 其中α_se_i是季节性注意力权重
        square_mu_se = F.conv2d(
            (x_reshaped ** 2).reshape(b * self.hidden_dim, -1, self.cycle_len, n),
            I_se.transpose(-1, -2)
        )
        # 将季节性二阶矩重塑为原始维度: [batch_size, hidden_dim, num_nodes, seq_len]
        # E[x²]_se = reshape(E[x²]_se, [b, h, n, t])
        square_mu_se = square_mu_se.reshape(b, self.hidden_dim, n, t)
        
        # 计算季节性方差和残差
        # 计算季节性方差: σ²_se = E[x²]_se - (E[x]_se)² + ε
        var_se = square_mu_se - mu_se ** 2 + epsilon
        # 计算季节性残差: r_se = (x - μ_se) / √(σ²_se)
        r_se = (x - mu_se) / (var_se ** 0.5)
        
        # 预测未来的季节性模式
        # 预测未来的季节性均值: μ̂_se = ∑(β_se_i * μ_se_i)
        # 其中 β_se_i 是预测注意力权重, μ_se_i 是历史季节性均值
        x_future = mu_se.reshape(b, self.hidden_dim, n, -1, self.cycle_len).permute(0, 1, 3, 4, 2)
        hat_mu_se = F.conv2d(
            x_future.reshape(b * self.hidden_dim, -1, self.cycle_len, n),
            E_se.transpose(-1, -2)  # β_se: [pred_len//cycle_len + 1, hist_len//cycle_len, 1, 1]
        ).reshape(b, self.hidden_dim, n, -1)[..., :self.pred_len]  # [b, h, n, pred_len]
        
        # 预测未来的季节性残差: r̂_se = ∑(β_se_i * r_se_i)
        # 其中 β_se_i 是预测注意力权重, r_se_i 是历史季节性残差
        x_future = r_se.reshape(b, self.hidden_dim, n, -1, self.cycle_len).permute(0, 1, 3, 4, 2)
        hat_r_se = F.conv2d(
            x_future.reshape(b * self.hidden_dim, -1, self.cycle_len, n),
            E_se.transpose(-1, -2)  # β_se: [pred_len//cycle_len + 1, hist_len//cycle_len, 1, 1]
        ).reshape(b, self.hidden_dim, n, -1)[..., :self.pred_len]  # [b, h, n, pred_len]
        
        # 将季节性特征添加到特征列表
        # features = [..., r_se, μ_se]
        s.extend([r_se, mu_se])
        hat_s.extend([hat_r_se, hat_mu_se])
        
        # 3. 短期依赖建模
        I_st = torch.softmax(self.I_st, dim=-1)  # α_st = softmax(I_st)
        E_st = torch.softmax(self.E_st, dim=-1)  # β_st = softmax(E_st)
        
        # 对输入序列进行填充,用于滑动窗口计算
        # 对输入序列进行填充: x_pad = [0, ..., 0, x_1, x_2, ..., x_t]
        # 其中填充了short_period_len-1个0在序列前面
        x_pad = F.pad(x, (self.short_period_len - 1, 0), "constant", 0)
        
        # 计算短期均值: μ_st = ∑(α_st_i * x_i)
        mu_st = F.conv2d(x_pad.reshape(b * self.hidden_dim, 1, n, -1), I_st).reshape(b, self.hidden_dim, n, t)
        
        # 计算二阶矩: E[x^2] = ∑(α_st_i * x_i^2)
        square_mu_st = F.conv2d(x_pad.reshape(b * self.hidden_dim, 1, n, -1) ** 2, I_st).reshape(b, self.hidden_dim, n, t)
        
        # 计算方差: Var(x) = E[x^2] - (E[x])^2 + ε
        var_st = square_mu_st - mu_st ** 2 + epsilon
        
        # 计算标准化残差: r_st = (x - μ_st) / sqrt(Var(x))
        r_st = (x - mu_st) / (var_st ** 0.5)
        
        # 预测期间的短期均值: μ_st_pred = ∑(β_st_i * μ_st_i)
        hat_mu_st = F.conv2d(mu_st[..., -self.short_period_len:].reshape(b * self.hidden_dim, 1, n, -1), E_st).reshape(b, self.hidden_dim, n, self.pred_len)
        
        # 预测期间的短期残差: r_st_pred = ∑(β_st_i * r_st_i)
        hat_r_st = F.conv2d(r_st[..., -self.short_period_len:].reshape(b * self.hidden_dim, 1, n, -1), E_st).reshape(b, self.hidden_dim, n, self.pred_len)
        
        # 将短期特征添加到特征列表
        s.extend([r_st, mu_st])
        hat_s.extend([hat_r_st, hat_mu_st])
        
        # 特征融合和处理
        s = torch.cat(s, dim=1)  # [batch_size, hidden_dim*6, num_nodes, hist_len]
        hat_s = torch.cat(hat_s, dim=1)  # [batch_size, hidden_dim*6, num_nodes, pred_len]
        x = torch.cat([s, hat_s], dim=-1)  # 在时间维度上连接
        
        # 填充和特征转换
        x = F.pad(x, mode='constant', pad=(self.kernel_size-1, 0))
        x = self.poly(x)  # 多项式特征转换，输入通道数为hidden_dim*6
        
        # 分离历史和预测特征
        x_z = x[..., :-self.pred_len]  # 提取历史特征
        s = x[..., -self.pred_len:]    # 提取预测特征
        
        # 残差和跳跃连接处理
        x_z = self.residual_conv(x_z)  # 残差卷积                
        s = self.skip_conv(s)  # 跳跃连接
        
        # 最终输出处理
        out = self.output_layer(s)  # [batch_size, 1, num_nodes, pred_len]
        out = out.permute(0, 3, 2, 1)  # [batch_size, pred_len, num_nodes, 1]
        
        return out

class PM25_GNN_SCNN(nn.Module):
    """PM2.5预测模型，结合GNN和改进的时间序列建模"""
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std, cycle_len, short_period_len):
        super(PM25_GNN_SCNN, self).__init__()
        
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.cycle_len = cycle_len
        
        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 13
        
        # GNN空间特征提取
        self.graph_gnn = GraphGNN(
            self.device, 
            edge_index, 
            edge_attr, 
            self.in_dim, 
            self.gnn_out, 
            wind_mean, 
            wind_std,
        )
        
        #时间序列建模
        self.time_module = TimeSeriesModule(
            input_dim=self.in_dim + self.gnn_out,  # GNN输出 + 原始特征
            hidden_dim=self.hid_dim,
            num_nodes=city_num,
            hist_len=hist_len,  # 12
            pred_len=pred_len,  # 12
            cycle_len=cycle_len,
            short_period_len=short_period_len
        )

    def forward(self, pm25_hist, feature):
        """
        Args:
            pm25_hist: [batch_size, hist_len, num_nodes, 1]
            feature: [batch_size, hist_len + pred_len, num_nodes, feature_dim]
        """
        batch_size = pm25_hist.size(0)
        
        # 准备输入序列
        input_sequence = []
        for i in range(self.hist_len):
            # 1. 获取当前时间步的特征
            curr_features = feature[:, i]
            
            # 2. 合并特征
            x = torch.cat((pm25_hist[:, i], curr_features), dim=-1)
            
            # 3. GNN空间特征提取
            x_gnn = self.graph_gnn(x)
            
            # 4. 合并GNN特征和原始特征
            x = torch.cat([x_gnn, x], dim=-1)
            input_sequence.append(x)
        
        # 将序列堆叠成时间序列
        x = torch.stack(input_sequence, dim=1)  # [batch_size, hist_len, num_nodes, feature_dim]
        
        # 时间序列建模生成预测
        pm25_pred = self.time_module(x)  # [batch_size, pred_len, num_nodes, 1]
        
        return pm25_pred

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
