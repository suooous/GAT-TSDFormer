import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset import HazeData


import arrow
import torch
from torch import nn
import datetime as dt
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

#import contextily as ctx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

knowair = np.load(file_dir['knowair_fp'])
knowair=knowair[:,:13,:]
print(knowair.shape)  # 查看数组的维度
# print(knowair.dtype)  # 查看数组的数据类型
# print(knowair[:5])  # 查看数组的前几行数据
import os
pm25 = knowair[:,:,-1:]
print(pm25.shape)  # 查看数组的维度


class AirQualityAnalysis:
    def __init__(self, knowair):
        self.knowair = knowair
        # 分离特征和PM2.5数据 [T0](1)
        self.feature = self.knowair[:, :, :-1]
        self.pm25 = self.knowair[:, :, -1:]

        # 定义城市名称映射
        self.cities = {
            0: "北京", 1: "天津", 2: "石家庄", 3: "唐山",
            4: "秦皇岛", 5: "邯郸", 6: "保定", 7: "张家口",
            8: "承德", 9: "廊坊", 10: "沧州", 11: "衡水",
            12: "邢台"
        }

        # 生成时间序列并创建数据框
        start_date = pd.date_range(start='2016-09-01', periods=len(self.pm25), freq='H')
        self.df = pd.DataFrame({
            'city': np.repeat(np.arange(13), len(self.pm25)),
            'pm25': self.pm25.reshape(-1),
            'datetime': np.tile(start_date, 13)
        })

        # 添加年份和月份列
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month

    def plot_monthly_boxplots(self):
        # 绘制所有城市汇总的月度箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y='pm25', data=self.df)
        plt.title('京津冀地区PM2.5月度分布箱线图')
        plt.xlabel('月份')
        plt.ylabel('PM2.5 浓度($\mu g/m^3$)')

        # 设置月份标签
        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                        '7月', '8月', '9月', '10月', '11月', '12月']
        plt.xticks(range(12), month_labels)

        # 添加网格线使图表更清晰
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()
        plt.savefig('京津冀地区PM2.5月度分布箱线图.png')

    def plot_yearly_boxplots(self):
        # 绘制年度箱线图
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='year', y='pm25', hue='city', data=self.df)
        plt.title('年度PM2.5箱线图')
        plt.xlabel('年份')
        plt.ylabel('PM2.5 浓度($\mu g/m^3$)')
        plt.legend(title='城市', labels=list(self.cities.values()), loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.savefig('年度PM2.5箱线图.png')
    def plot_city_boxplots(self):
        # 绘制城市箱线图
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='city', y='pm25', data=self.df)
        plt.title('城市PM2.5箱线图')
        plt.xlabel('城市')
        plt.ylabel('PM2.5 浓度($\mu g/m^3$)')
        plt.xticks(range(len(self.cities)), list(self.cities.values()), rotation=45)
        plt.show()
        plt.savefig('城市PM2.5箱线图.png')

# knowair为您的数据，shape为(11688, 13, 1)
analysis = AirQualityAnalysis(knowair)
analysis.plot_monthly_boxplots()
analysis.plot_yearly_boxplots()
analysis.plot_city_boxplots()
#
# # 将数据转换为DataFrame
# df = pd.DataFrame({
#     'time': pd.to_datetime(pm25[:, :, 0].reshape(-1), unit='s'),
#     'city': np.repeat(np.arange(13), 11688),  # 为13个城市创建重复的索引
#     'pm25': pm25.reshape(-1)  # 直接将pm25展平
# })
#
# # 提取年和月
# df['year'] = df['time'].dt.year
# df['month'] = df['time'].dt.month
# def draw_box():
#     # 按年份分组绘制箱线图
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x='year', y='pm25', hue='city', data=df)
#     plt.title('Yearly PM2.5 Boxplot by City')
#     plt.show()
#
#     # 按月份分组绘制箱线图
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x='month', y='pm25', hue='city', data=df)
#     plt.title('Monthly PM2.5 Boxplot by City')
#     plt.show()
#
#     # 按城市分组绘制箱线图
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x='city', y='pm25', data=df)
#     plt.title('PM2.5 Boxplot by City')
#     plt.show()


def load_all_npy(directory):
    """
    加载指定目录下的所有npy文件
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录 {directory} 不存在")
    npy_files = glob.glob(os.path.join(directory, '*.npy'))
    if not npy_files:
        raise ValueError(f"目录 {directory} 中没有npy文件")

    # 使用numpy.stack将所有npy数组堆叠成一个更大的数组
    npy_data_list = []
    for file in npy_files:
        data = np.load(file)
        npy_data_list.append(data)

    # 将列表转换为numpy数组
    npy_data = np.array(npy_data_list)
    return npy_data

# 定义目录路径
#directory = r'E:\jupyter\match\create\data\results\1_24\3\PM25_GNN\20250211171823\00'

# # 获取所有 .npy 文件
# npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
#
# # 加载并打印每个文件的信息
# for file in npy_files:
#     file_path = os.path.join(directory, file)
#     try:
#         data = np.load(file_path)
#         print(f"文件：{file}")
#         print(f"形状：{data.shape}")
#         print(f"数据类型：{data.dtype}")
#         print(f"前五个元素：{data[:5]}\n")
#     except Exception as e:
#         print(f"加载文件 {file} 出错：{str(e)}")


def vis_graph(graph):
    # 获取节点信息
    idx, cities, lons, lats = graph.traverse_graph()

    # 获取边的连接信息
    lines = graph.gen_lines()

    # 创建地图可视化
    fig, ax = plt.subplots(figsize=(12, 8))

    # 添加底图
#    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Stamen.TerrainBackground)

    # 绘制节点
    ax.scatter(lons, lats, c='red', s=50, zorder=2)

    # 绘制边
    for line in lines:
        ax.plot(line[0], line[1], 'b-', alpha=0.3, zorder=1)

    # 添加城市标签
    for i in range(len(cities)):
        ax.annotate(cities[i], (lons[i], lats[i]), zorder=3)

    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_title('京津冀地区空气质量监测站点网络图')

    plt.show()
    plt.savefig('京津冀地区空气质量监测站点网络图.png', dpi=300, bbox_inches='tight')

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph(region='jjj')
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def inspect_haze_data(haze_data):
    """
    检查HazeData对象的详细信息
    """
    # 查看所有可用属性
    attributes = [attr for attr in dir(haze_data) if not attr.startswith('_')]
    print("可用属性列表：")
    for attr in attributes:
        value = getattr(haze_data, attr)
        print(f"{attr}: {type(value)}")

        # 如果属性是数组或张量，打印其形状
        if hasattr(value, 'shape'):
            print(f"{attr} shape: {value.shape}")


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    #vis_graph(graph)
    #print(type(train_data))
    # 查看对象的所有属性和方法
    #print(dir(train_data))

    # 查看对象的基本信息
    #print(train_data.__dict__)

    #inspect_haze_data(train_data)
    1
    #draw_box()
if __name__ == '__main__':
    main()
