import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch

class CriteoDataProcessor:
    def __init__(self, file_path, num_features, cat_features, target_column):
        """
        初始化数据处理器
        :param file_path: 数据文件路径
        :param num_features: 数值特征列名列表
        :param cat_features: 类别特征列名列表
        :param target_column: 目标列名
        """
        self.file_path = file_path
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_column = target_column
        self.cloums = [target_column] + num_features + cat_features
        self.label_encoders = {col: LabelEncoder() for col in self.cat_features}
        self.scaler = MinMaxScaler()
        self.cat_feature_dims = {}  # 保存类别特征维度

    def load_and_preprocess(self):
        """
        加载数据并进行预处理
        :return: 处理后的数据
        """
        # 加载数据
        data = pd.read_csv(self.file_path, sep='\t' ,names= self.cloums)

        # 填充缺失值
        for col in self.num_features:
            data[col].fillna(0, inplace=True)
        for col in self.cat_features:
            data[col].fillna('<UNK>', inplace=True)

        # 编码类别特征
        for col in self.cat_features:
            data[col] = self.label_encoders[col].fit_transform(data[col])
            self.cat_feature_dims[col] = data[col].nunique()

        # 标准化数值特征
        data[self.num_features] = self.scaler.fit_transform(data[self.num_features])

        return data

    def split_data(self, data, test_size=0.2, val_size=0.1, random_state=42):
        """
        划分数据集为训练、验证和测试集
        :param data: 数据集
        :param test_size: 测试集占比
        :param val_size: 验证集占比（基于训练集划分）
        :param random_state: 随机种子
        :return: 划分后的训练、验证和测试数据集
        """
        train_data, temp_data = train_test_split(data, test_size=test_size, random_state=random_state)
        val_data, test_data = train_test_split(temp_data, test_size=val_size / (1 - test_size),
                                               random_state=random_state)
        return train_data, val_data, test_data


class CriteoDataset(Dataset):
    def __init__(self, data, num_features, cat_features, target_column):
        """
        自定义Dataset
        :param data: 数据集
        :param feature_columns: 特征列名列表
        :param target_column: 目标列名
        """
        self.data = data
        self.num_features = num_features
        self.cat_features = cat_features
        self.target_column = target_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        num_features = torch.tensor(row[self.num_features].values, dtype=torch.float32)
        cat_features = torch.tensor(row[self.cat_features].values, dtype=torch.long)
        target = torch.tensor(row[self.target_column], dtype=torch.float32)
        return num_features, cat_features, target


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    创建DataLoader
    :param dataset: 数据集对象
    :param batch_size: 批量大小
    :param shuffle: 是否打乱数据
    :param num_workers: 工作线程数
    :return: DataLoader对象
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
