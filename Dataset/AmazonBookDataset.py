import torch
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
def AmazonBookPreprocess(dataframe, seq_len=40):
    """
    数据集处理
    :param dataframe: 未处理的数据集
    :param seq_len: 数据序列长度
    :return data: 处理好的数据集
    """
    # 1.按'|'切割，用户历史购买数据，获取item的序列和类别的序列
    data = dataframe.copy()
    data['hist_item_list'] = dataframe.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
    data['hist_cate_list'] = dataframe.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

    # 2.获取cate的所有种类，为每个类别设置一个唯一的编码
    cate_list = list(data['cateID'])
    _ = [cate_list.extend(i) for i in data['hist_cate_list'].values]
    # 3.将编码去重
    cate_set = set(cate_list + ['0'])  # 用 '0' 作为padding的类别

    # 4.截取用户行为的长度,也就是截取hist_cate_list的长度，生成对应的列名
    cols = ['hist_cate_{}'.format(i) for i in range(seq_len)]

    # 5.截取前40个历史行为，如果历史行为不足40个则填充0
    def trim_cate_list(x):
        if len(x) > seq_len:
            # 5.1历史行为大于40, 截取后40个行为
            return pd.Series(x[-seq_len:], index=cols)
        else:
            # 5.2历史行为不足40, padding到40个行为
            pad_len = seq_len - len(x)
            x = x + ['0'] * pad_len
            return pd.Series(x, index=cols)

    # 6.预测目标为试题的类别
    labels = data['label']
    data = data['hist_cate_list'].apply(trim_cate_list).join(data['cateID'])

    # 7.生成类别对应序号的编码器，如book->1,Russian->2这样
    cate_encoder = LabelEncoder().fit(list(cate_set))
    # 8.这里分为两步，第一步为把类别转化为数值，第二部为拼接上label
    data = data.apply(cate_encoder.transform).join(labels)
    fields = data.max().max()
    return data, fields

def create_amazon_book_dataloader(path, batch_size):
    """
    加载 AmazonBook 数据集并生成 DataLoader
    :param path: 数据集路径
    :param batch_size: 每批数据大小
    :return: 训练集、验证集、测试集的 DataLoader 和类别数
    """
    data = pd.read_csv(path)
    processed_data, fields = AmazonBookPreprocess(data)

    # 划分数据集（训练集:验证集:测试集 = 60%:20%:20%）
    X, y = processed_data.iloc[:, :-1], processed_data['label']
    train_X, tmp_X, train_y, tmp_y = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(tmp_X, tmp_y, test_size=0.5, stratify=tmp_y, random_state=42)

    # 转为 Tensor
    def to_tensor(X, y):
        return torch.tensor(X.values, dtype=torch.long), torch.tensor(y.values, dtype=torch.long)

    train_X, train_y = to_tensor(train_X, train_y)
    val_X, val_y = to_tensor(val_X, val_y)
    test_X, test_y = to_tensor(test_X, test_y)

    # 创建 DataLoader
    train_loader = Data.DataLoader(Data.TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(Data.TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(Data.TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, fields

