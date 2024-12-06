import torch
import torch.nn as nn


class Wide(nn.Module):
    """
    Wide部分，用于处理稠密特征，通过一个线性层输出。
    """
    def __init__(self, dense_fea_num):
        super(Wide, self).__init__()
        self.linear = nn.Linear(dense_fea_num, 1)

    def forward(self, x_dense):
        return self.linear(x_dense)


class Deep(nn.Module):
    """
    Deep部分，用于处理稀疏和稠密特征的联合表示。
    """
    def __init__(self, dense_fea_num, sparse_fea_num, sparse_fea_list, embed_dim, hidden_layers):
        """
        初始化Deep模块。

        :param dense_fea_num: 稠密特征数量
        :param sparse_fea_num: 稀疏特征数量
        :param sparse_fea_list: 每个稀疏特征的取值个数列表
        :param embed_dim: 稀疏特征嵌入的维度
        :param hidden_layers: MLP隐藏层的维度列表
        """
        super(Deep, self).__init__()
        self.sparse_fea_num = sparse_fea_num
        self.embed_layers = nn.ModuleList(
            nn.Embedding(num_embeddings, embed_dim) for num_embeddings in sparse_fea_list
        )
        input_dim = dense_fea_num + sparse_fea_num * embed_dim
        self.mlp = nn.Sequential(
            nn.Flatten(),
            *self._build_mlp(input_dim, hidden_layers),
            nn.LeakyReLU()
        )

    @staticmethod
    def _build_mlp(input_dim, hidden_layers):
        layers = []
        for out_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, out_dim))
            layers.append(nn.LeakyReLU())
            input_dim = out_dim
        return layers

    def forward(self, x_dense, x_sparse):
        """
        前向传播。

        :param x_dense: 稠密特征输入 (batch_size, dense_fea_num)
        :param x_sparse: 稀疏特征输入 (batch_size, sparse_fea_num)
        :return: Deep模块的输出
        """
        sparse_embeds = [emb(x_sparse[:, i]) for i, emb in enumerate(self.embed_layers)]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)  # 合并所有稀疏特征的嵌入
        deep_input = torch.cat([x_dense, sparse_embeds], dim=-1)  # 拼接稠密和稀疏特征
        return self.mlp(deep_input)


class WideDeep(nn.Module):
    """
    Wide & Deep模型，用于联合处理稠密和稀疏特征。
    """
    def __init__(self, dense_fea_num, sparse_fea_num, sparse_fea_list, embed_dim, hidden_layers):
        """
        初始化Wide & Deep模型。

        :param dense_fea_num: 稠密特征数量
        :param sparse_fea_num: 稀疏特征数量
        :param sparse_fea_list: 稀疏特征取值个数列表
        :param embed_dim: 稀疏特征嵌入维度
        :param hidden_layers: Deep部分隐藏层的维度列表
        """
        super(WideDeep, self).__init__()
        self.wide = Wide(dense_fea_num)
        self.deep = Deep(dense_fea_num, sparse_fea_num, sparse_fea_list, embed_dim, hidden_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(1 + hidden_layers[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x_dense, x_sparse):
        """
        前向传播。

        :param x_dense: 稠密特征输入
        :param x_sparse: 稀疏特征输入
        :return: 模型预测值
        """
        wide_out = self.wide(x_dense)  # Wide部分输出
        deep_out = self.deep(x_dense, x_sparse)  # Deep部分输出
        combined = torch.cat([wide_out, deep_out], dim=-1)  # 拼接Wide和Deep部分
        return self.output_layer(combined)

    def save_model(self, model_path):
        """保存模型权重。"""
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        """加载模型权重。"""
        self.load_state_dict(torch.load(model_path))
