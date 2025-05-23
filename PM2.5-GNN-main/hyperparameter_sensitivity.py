import os
import torch
import matplotlib.pyplot as plt
from train import get_model, train, val
from dataset import HazeData
from graph import Graph
from util import config, file_dir
import datetime

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化图和数据集
graph = Graph(region='jjj')
dataset_num = config['experiments']['dataset_num']
batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
weight_decay = config['train']['weight_decay']
lr = config['train']['lr']

# 创建保存目录
save_dir = 'PM2.5-GNN-main/sensitive_save'
os.makedirs(save_dir, exist_ok=True)

# def test_learning_rates():
#     learning_rates = [0.001, 0.0005, 0.0001]
#     results = {}

#     for lr in learning_rates:
#         print(f"Testing with learning rate: {lr}")
#         model = get_model().to(device)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
#         train_loader, val_loader = get_data_loaders()
#         train_loss_list, val_loss_list = run_training(model, optimizer, train_loader, val_loader)
        
#         results[lr] = {'train_loss': train_loss_list, 'val_loss': val_loss_list}
    
#     print_results(results, "Learning Rate")
#     plot_results(results, "Learning Rate")

# def test_optimizers():
#     optimizers = {
#         'AdamW': torch.optim.AdamW,
#         'SGD': torch.optim.SGD,
#         'RMSprop': torch.optim.RMSprop
#     }
#     results = {}

#     for opt_name, opt_class in optimizers.items():
#         print(f"Testing with optimizer: {opt_name}")
#         model = get_model().to(device)
#         optimizer = opt_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        
#         train_loader, val_loader = get_data_loaders()
#         train_loss_list, val_loss_list = run_training(model, optimizer, train_loader, val_loader)
        
#         results[opt_name] = {'train_loss': train_loss_list, 'val_loss': val_loss_list}
    
#     print_results(results, "Optimizer")
#     plot_results(results, "Optimizer")

def test_window_sizes():
    window_sizes = [8,16,24]
    results = {}

    for hist_len in window_sizes:
        print(f"Testing with window size: hist_len={hist_len}")
        train_loader, val_loader = get_data_loaders(hist_len)
        
        model = get_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_loss_list, val_loss_list = run_training(model, optimizer, train_loader, val_loader)
        
        results[hist_len] = {'train_loss': train_loss_list, 'val_loss': val_loss_list}
    
    print_results(results, "Window Size")
    plot_results(results, "Window Size")

def get_data_loaders(hist_len=None, pred_len=None):
    if hist_len is None:
        hist_len = config['train']['hist_len']
    if pred_len is None:
        pred_len = config['train']['pred_len']
    
    train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train', region='jjj')
    val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val', region='jjj')
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader

def run_training(model, optimizer, train_loader, val_loader):
    train_loss_list, val_loss_list = [], []
    for epoch in range(epochs):
        train_loss = train(train_loader, model, optimizer)
        val_loss = val(val_loader, model)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    return train_loss_list, val_loss_list

def print_results(results, test_type):
    for param, result in results.items():
        print(f"{test_type}: {param}")
        print(f"Train Loss: {result['train_loss']}")
        print(f"Val Loss: {result['val_loss']}")

def plot_results(results, test_type):
    plt.figure(figsize=(10, 5))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
    for idx, (param, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]  # 循环使用颜色
        plt.plot(result['train_loss'], label=f'Train Loss ({test_type}={param})', color=color)
        plt.plot(result['val_loss'], label=f'Val Loss ({test_type}={param})', linestyle='--', color=color)
    
    plt.title(f'{test_type} Sensitivity')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 获取当前日期
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # 保存图像
    plt.savefig(os.path.join(save_dir, f'{test_type}_sensitivity_{current_date}.png'))
    plt.close()

if __name__ == '__main__':
    # test_learning_rates()
    # test_optimizers()
    test_window_sizes() 
