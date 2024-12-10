import torch
import torch.nn as nn
import torch.nn.functional as F


class Dnn(nn.Module):
    """
    全连接DNN网络，包含Dropout和ReLU激活函数。
    """

    def __init__(self, hidden_units, dropout=0.0):
        """
        初始化DNN网络。

        参数:
            hidden_units (list): 每层神经元个数的列表，例如[256, 128, 64]。
            dropout (float): Dropout的失活率，用于正则化。
        """
        super(Dnn, self).__init__()
        # 定义每层的全连接层
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(hidden_units[:-1], hidden_units[1:])
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 逐层前向传播，每层包含线性变换和ReLU激活
        for layer in self.layers:
            x = F.relu(layer(x))
        # 最后添加Dropout层
        return self.dropout(x)


class NFM(nn.Module):
    """
    神经因子分解机 (NFM) 模型。
    """

    def __init__(self, sparse_fea_list, hidden_units, embed_dim=8):
        """
        初始化NFM模型。

        参数:
            sparse_fea_list (list): 每个稀疏特征的类别总数。
            hidden_units (list): DNN隐藏层的神经元数量。
            embed_dim (int): 稀疏特征Embedding的维度，默认8。
        """
        super(NFM, self).__init__()
        self.dense_features = [f'I{i}' for i in range(1, 14)]  # 数值型特征
        self.sparse_features = [f'C{i}' for i in range(1, 27)]  # 离散型特征
        self.sparse_fea_list = sparse_fea_list
        self.sparse_features_map = dict(zip(self.sparse_features, self.sparse_fea_list))

        # 定义稀疏特征的Embedding层
        self.embed_layers = nn.ModuleDict({
            f'embed_{key}': nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for key, val in self.sparse_features_map.items()
        })

        # 输入维度 = 数值特征维度 + 稀疏特征Embedding维度
        input_dim = len(self.dense_features) + embed_dim
        hidden_units.insert(0, input_dim)

        self.bn = nn.BatchNorm1d(input_dim)  # BatchNorm层
        self.dnn_network = Dnn(hidden_units)  # DNN部分
        self.dnn_final_linear = nn.Linear(hidden_units[-1], 1)  # 输出层

    def forward(self, x_dense, x_sparse):
        """
        前向传播。

        参数:
            x_dense (Tensor): 数值型特征 (batch_size, dense_feature_dim)。
            x_sparse (Tensor): 离散型特征 (batch_size, sparse_feature_dim)。

        返回:
            Tensor: 模型预测结果 (batch_size, 1)。
        """
        # 稀疏特征转为长整型
        x_sparse = x_sparse.long()

        # 对每个稀疏特征进行Embedding，结果形状为 (batch_size, sparse_feature_dim, embed_dim)
        sparse_embeds = torch.stack([
            self.embed_layers[f'embed_{key}'](x_sparse[:, i])
            for i, key in enumerate(self.sparse_features_map.keys())
        ], dim=1)

        # 特征交叉操作，根据公式计算交叉部分，结果形状为 (batch_size, embed_dim)
        embed_cross = 0.5 * (
                torch.pow(sparse_embeds.sum(dim=1), 2) -
                torch.sum(torch.pow(sparse_embeds, 2), dim=1)
        )

        # 数值特征与交叉后的稀疏特征拼接，形状为 (batch_size, dense_dim + embed_dim)
        x = torch.cat([embed_cross, x_dense], dim=-1)
        x = self.bn(x)  # 归一化

        # DNN部分，最终输出
        dnn_out = self.dnn_final_linear(self.dnn_network(x))
        outputs = torch.sigmoid(dnn_out)  # 使用Sigmoid激活函数输出概率
        return outputs
