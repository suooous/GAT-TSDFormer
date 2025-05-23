import yaml
import sys
import os
import numpy as np
import platform
import torch

# 替换原来的 os.uname().nodename
nodename = platform.node()

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


#nodename = os.uname().nodename
file_dir = config['filepath'][nodename]


def main():
    pass


def preprocess_data(data):
    # 确保数据中没有 NaN 或 Inf
    data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    return data


if __name__ == '__main__':
    main()
