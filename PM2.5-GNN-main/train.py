import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset import HazeData
import time
import matplotlib.dates as mdates
import datetime as dt

from model.MLP import MLP
from model.HA import HA
from model.nodesFC_GRU import nodesFC_GRU
from model.MTGNN import MTGNN
from model.GNN_MLP import PM25_GNN_MLP
from model.GNN_SCNN import PM25_GNN_SCNN
from model.GAGNN import GAGNN
from model.GNN_Direct import PM25_GNN_Direct  
from model.iTransformer import iTransformer
from model.Airformer import Airformer
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
import torch.nn.functional as F

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph(region='jjj')
city_num = graph.node_num

# 恢复使用config中的配置
# 从config.yaml的train中获取配置
batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
exp_repeat = config['train']['exp_repeat']
cycle_len = config['train']['cycle_len']
short_period_len = config['train']['short_period_len']



# 从config.yaml的results_dir中获取结果保存路径
results_dir = file_dir['results_dir']
# 从config.yaml的experiments中获取实验配置
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train',region='jjj')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val',region='jjj')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test',region='jjj')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75  # PM2.5浓度阈值
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    mape = np.mean(np.mean(np.abs((predict - label) / label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far,mape


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'cycle_len: %s\n' % cycle_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info


def get_model():
    if exp_model == 'HA':
        return HA(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'MTGNN':
        return MTGNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_MLP':
        return PM25_GNN_MLP(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_SCNN':
        return PM25_GNN_SCNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std, cycle_len, short_period_len)
    elif exp_model == 'GAGNN':
        return GAGNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_Direct':  # 添加新模型
        return PM25_GNN_Direct(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'iTransformer':
        return iTransformer(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'Airformer':
        return Airformer(hist_len, pred_len, in_dim, city_num, batch_size, device)
    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    
    for batch_idx, data in tqdm(enumerate(train_loader)):
        try:
            optimizer.zero_grad()
            
            pm25, feature, time_arr = data
            pm25 = pm25.to(device)
            feature = feature.to(device)
            
            # 检查输入数据
            if torch.isnan(pm25).any() or torch.isnan(feature).any():
                print(f"批次{batch_idx}的输入数据包含NaN，跳过此批次")
                continue
            
            pm25_label = pm25[:, hist_len:]
            pm25_hist = pm25[:, :hist_len]
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 前向传播
            pm25_pred = model(pm25_hist, feature)
            
            # 检查预测结果
            if torch.isnan(pm25_pred).any():
                print(f"批次{batch_idx}的预测结果包含NaN，跳过此批次")
                continue
            
            loss = criterion(pm25_pred, pm25_label)
            
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"批次{batch_idx}的损失为NaN或Inf，跳过此批次")
                continue
            
            loss.backward()
            
            # 检查梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"参数{name}的梯度包含NaN")
                        param.grad.data.zero_()
            
            optimizer.step()
            train_loss += loss.item()
            
        except Exception as e:
            print(f"训练批次{batch_idx}发生错误: {str(e)}")
            continue
    
    train_loss /= len(train_loader)
    return train_loss


def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model):
    start_time = time.time()
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0
    # 测试过程
    test_time = time.time() - start_time

    return test_loss, predict_epoch, label_epoch, time_epoch,test_time


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    rmse_list, mae_list, csi_list = [], [], []
    pod_list, far_list, mape_list = [], [], []
    train_time_list, test_time_list = [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)
        
        # 记录开始时间
        start_time = time.time()

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))

        optimizer = torch.optim.AdamW(  # 使用AdamW替代RMSprop
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0
        train_loss_, val_loss_ = 0, 0
        test_loss = float('inf')  # 初始化test_loss为默认值
        predict_epoch = None
        label_epoch = None
        time_epoch = None
        rmse = mae = csi = pod = far = mape = float('inf')  # 初始化指标为默认值

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            val_loss = val(val_loader, model)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)
            
            # 检查损失是否为NaN
            if np.isnan(train_loss) or np.isnan(val_loss):
                print('警告：训练或验证损失为NaN，尝试降低学习率')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                continue

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                try:
                    test_loss, predict_epoch, label_epoch, time_epoch, test_time = test(test_loader, model)
                    train_loss_, val_loss_ = train_loss, val_loss
                    rmse, mae, csi, pod, far, mape = get_metric(predict_epoch, label_epoch)
                    print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f, MAPE:%0.4f' % 
                          (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far, mape))

                    if save_npy:
                        np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                        np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                        np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)
                except Exception as e:
                    print(f'测试过程发生错误: {str(e)}')
                    # 保持默认值不变

        # 记录结果
        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)
        mape_list.append(mape)
        train_time = time.time() - start_time
        train_time_list.append(train_time)
        test_time_list.append(test_time)

        print('\nNo.%2d experiment results:' % exp_idx)
        print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f, MAPE:%0.4f' % 
              (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far, mape))
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())

    # # 获取GPU内存使用情况
    # #（因为我电脑是集成显卡，所以这个gpu_mem_train和gpu_mem_test无法计算，
    # # 我这里直接令两个都等于0了，国杰和铸峰你们修改的时候记得用上这些代码
    # # 训练结束后获取GPU内存使用情况
   # gpu_mem_train=0
   # gpu_mem_test=0
    gpu_mem_train = torch.cuda.max_memory_allocated() / 1024 ** 2  # 转换为GB

    # 在测试开始前重置内存最大值
    torch.cuda.reset_max_memory_allocated()
    # 测试结束后获取GPU内存使用情况
    gpu_mem_test = torch.cuda.max_memory_allocated() / 1024 ** 2  # 转换为GB


    # 计算训练时间和推理时间
    train_time = sum(train_time_list)
    test_time = sum(test_time_list)

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list)) + \
                     'MAPE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mape_list)) + \
                     '\n=================== Model Info ===================\n'+ \
                     f'model params: {num_params}\n' + \
                     f'GPU memory usage during training: {gpu_mem_train:.4f}GB\n' + \
                     f'GPU memory usage during testing: {gpu_mem_test:.4f}GB\n' + \
                     f'Training time: {train_time:.4f}s\n' + \
                     f'Inference time: {test_time:.4f}s\n' + \
                     f'Model file path: {model_fp}\n'


    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)
    jjj_cities = {
        0: "北京",
        1: "天津",
        2: "石家庄",
        3: "唐山",
        4: "秦皇岛",
        5: "邯郸",
        6: "保定",
        7: "张家口",
        8: "承德",
        9: "廊坊",
        10: "沧州",
        11: "衡水",
        12: "邢台",
    }
    # 在main函数中，创建两个子文件夹用于存储不同版本的图片
    no_shift_dir = os.path.join(exp_model_dir, 'no_shift_plots')
    shifted_dir = os.path.join(exp_model_dir, 'shifted_plots')
    os.makedirs(no_shift_dir, exist_ok=True)
    os.makedirs(shifted_dir, exist_ok=True)

    for city_idx in range(13):  # 13个城市
        # 提取对应城市的数据
        predict_city = predict_epoch[:, :, city_idx, 0]  # (192, 25)
        label_city = label_epoch[:, :, city_idx, 0]  # (192, 25)
        time_city = time_epoch[:, city_idx]  # (192,)

        # 提取最后一位作为预测值
        predict_values = predict_city[:, -1]  # (192,)
        label_values = label_city[:, -1]  # (192,)
        
        # 创建时间列表，平移量为hist_len + pred_len
        time_list = [dt.datetime.utcfromtimestamp(t) for t in time_city]
        total_shift = pred_len
        predict_time_list = [t - dt.timedelta(hours=total_shift) for t in time_list]

        city_name = jjj_cities[city_idx]
        
        # 1. 绘制不平移的版本
        plt.figure(figsize=(12, 6))
        plt.plot(time_list, label_values, label='真实值', color='blue', linewidth=2)
        plt.plot(time_list, predict_values, label='预测值', color='red', 
                linestyle='--', linewidth=2)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
        plt.gcf().autofmt_xdate()
        plt.xlabel('时间')
        plt.ylabel('PM2.5浓度 (μg/m³)')
        # plt.title(f'{city_name} PM2.5预测与真实值对比 (未平移)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(no_shift_dir, f'{city_name}_PM2.5_prediction.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 绘制平移后的版本（删除不匹配部分）
        plt.figure(figsize=(12, 6))
        
        # 删除不匹配的部分
        valid_start = total_shift
        valid_end = len(predict_time_list)
        
        # 截取有效的时间范围
        valid_time_list = time_list[valid_start:]  # 真实值时间
        valid_predict_time_list = predict_time_list[:-total_shift]  # 预测值时间
        valid_label_values = label_values[:-total_shift]  # 真实值
        valid_predict_values = predict_values[valid_start:]  # 预测值

        plt.plot(valid_time_list, valid_label_values, 
                label='真实值', color='blue', linewidth=2)
        plt.plot(valid_time_list, valid_predict_values, 
                label=f'预测值({total_shift*3}小时预测)', 
                color='red', linestyle='--', linewidth=2)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
        plt.gcf().autofmt_xdate()
        plt.xlabel('时间')
        plt.ylabel('PM2.5浓度 (μg/m³)')
        # plt.title(f'{city_name} PM2.5预测与真实值对比 (时间平移{total_shift*3}小时)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(shifted_dir, f'{city_name}_PM2.5_prediction.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# 添加城市权重的损失
def weighted_mse_loss(pred, target, city_weights):
    loss = 0
    for i in range(pred.size(2)):  # 遍历每个城市
        city_pred = pred[..., i, :]
        city_target = target[..., i, :]
        city_weight = city_weights[i]
        loss += city_weight * F.mse_loss(city_pred, city_target)
    return loss

if __name__ == '__main__':
    main()
