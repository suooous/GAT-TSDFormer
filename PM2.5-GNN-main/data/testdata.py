import numpy as np
from datetime import datetime, timedelta

# 加载数据
data_path = r'D:\timeDependent\pm2.5project_newest\PM2.5-GNN-main\data\KnowAir.npy'
data = np.load(data_path, allow_pickle=True)

# 首先打印数据类型和形状
print("Data type:", type(data))
print("Data shape:", data.shape)

# 计算时间步
start_time = datetime(2015, 1, 1, 0, 0)  # 从config.yaml中的data_start
time_steps = []
for i in range(data.shape[0]):
    time_steps.append(start_time + timedelta(hours=3*i))

# 打印时间信息
print("\nTime steps analysis:")
print(f"Total time steps: {len(time_steps)}")
print(f"Start time: {time_steps[0]}")
print(f"End time: {time_steps[-1]}")
print(f"Time interval: {time_steps[1] - time_steps[0]}")

# 打印特征信息
print("\nFeature analysis:")
feature_names = [
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'boundary_layer_height',
    'k_index',
    'relative_humidity+950',
    'surface_pressure',
    'total_precipitation',
    'u_component_of_wind+950',
    'v_component_of_wind+950',
    'vertical_velocity+950',
    'vorticity+950',
    # ... 其他特征
    'PM2.5'  # 最后一个特征是PM2.5值
]

print(f"Number of features: {data.shape[2]}")
print("Sample data for first time step, first city:")
for i, name in enumerate(feature_names[:data.shape[2]]):
    print(f"{name}: {data[0, 0, i]}")