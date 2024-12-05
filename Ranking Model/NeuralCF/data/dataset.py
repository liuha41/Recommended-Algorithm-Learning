import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class MovieLensDataset:
    def __init__(self, file, device=torch.device('cpu')):
        # 初始化设备
        self.device = device

        # 加载数据
        dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float16, 'timestamp': str}
        data_df = pd.read_csv(file, sep='::', header=None, dtype=dtype, names=['userId', 'movieId', 'rating', 'timestamp'])

        # 转换评分为二分类标签（>3为1，<=3为0）
        data_df['rating'] = (data_df['rating'] > 3).astype(np.int8)

        # 仅保留必要列
        self.data = data_df[['userId', 'movieId', 'rating']].values

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        # 确定每个字段的特征维度
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        # 按比例划分数据集
        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2021)
        valid, test = train_test_split(valid_test, train_size=valid_size / (valid_size + test_size), random_state=2021)

        # 转换为PyTorch张量并转移到指定设备
        to_tensor = lambda x: torch.tensor(x, dtype=torch.long).to(self.device)
        train_X, train_y = to_tensor(train[:, :-1]).to(self.device), torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(self.device)
        valid_X, valid_y = to_tensor(valid[:, :-1]).to(self.device), torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(self.device)
        test_X, test_y = to_tensor(test[:, :-1]).to(self.device), torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(self.device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

# 辅助函数，加载数据并划分数据集
def get_data(path, device=torch.device('cpu')):
    dataset = MovieLensDataset(path, device)
    return dataset.train_valid_test_split()
