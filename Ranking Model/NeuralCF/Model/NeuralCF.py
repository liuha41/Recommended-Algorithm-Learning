# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class NeuralCF(nn.Module):
#
#     def __init__(self, field_dims, embed_dim=4):
#         super(NeuralCF, self).__init__()
#         self.embed1 = FeaturesEmbedding(field_dims, embed_dim)
#         self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
#
#         self.mlp = MultiLayerPerceptron([len(field_dims) * embed_dim, 128, 64])
#         self.fc = nn.Linear(embed_dim + 64, 1)
#
#     def forward(self, x):
#         embeddings1 = self.embed1(x)
#         gmf_output = embeddings1[:, 0].mul(embeddings1[:, 1]).squeeze(-1)
#
#         embeddings2 = self.embed2(x)
#         mlp_input = embeddings2.reshape(x.shape[0], -1)
#         mlp_output = self.mlp(mlp_input)
#
#         concated = torch.hstack([gmf_output, mlp_output])
#         output = self.fc(concated)
#         output = torch.sigmoid(output)
#         return output
#
# class FeaturesEmbedding(nn.Module):
#
#     def __init__(self, field_dims, embed_dim):
#         super(FeaturesEmbedding, self).__init__()
#         self.embedding = Embedding(sum(field_dims), embed_dim)
#
#         # e.g. field_dims = [2, 3, 4, 5], offsets = [0, 2, 5, 9]
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
#
#     def forward(self, x):
#         """
#         :param x: shape (batch_size, num_fields)
#         :return: shape (batch_size, num_fields, embedding_dim)
#         """
#         x = x + x.new_tensor(self.offsets)
#         return self.embedding(x)
#
# class Embedding:
#
#     def __new__(cls, num_embeddings, embed_dim):
#         if torch.cuda.is_available():
#             embedding = nn.Embedding(num_embeddings, embed_dim)
#             nn.init.xavier_uniform_(embedding.weight.data)
#             return embedding
#         else:
#             return CpuEmbedding(num_embeddings, embed_dim)
#
# # 在 cpu 下，比 nn.Embedding 快，但是在 gpu 的序列模型下比后者慢太多了
# class CpuEmbedding(nn.Module):
#
#     def __init__(self, num_embeddings, embed_dim):
#         super(CpuEmbedding, self).__init__()
#
#         self.weight = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
#         nn.init.xavier_uniform_(self.weight.data)
#
#     def forward(self, x):
#         """
#         :param x: shape (batch_size, num_fields)
#         :return: shape (batch_size, num_fields, embedding_dim)
#         """
#         return self.weight[x]
#
# class MultiLayerPerceptron(nn.Module):
#
#     def __init__(self, layer, batch_norm=True):
#         super(MultiLayerPerceptron, self).__init__()
#         layers = []
#         input_size = layer[0]
#         for output_size in layer[1: -1]:
#             layers.append(nn.Linear(input_size, output_size))
#             if batch_norm:
#                 layers.append(nn.BatchNorm1d(output_size))
#             layers.append(nn.ReLU())
#             input_size = output_size
#         layers.append(nn.Linear(input_size, layer[-1]))
#
#         self.mlp = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.mlp(x)


import torch
import torch.nn as nn
import numpy as np

class NeuralCF(nn.Module):
    def __init__(self, field_dims, embed_dim=4):
        super(NeuralCF, self).__init__()
        # 初始化GMF和MLP的Embedding
        self.embed_gmf = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_mlp = FeaturesEmbedding(field_dims, embed_dim)

        # 初始化MLP和最终全连接层
        self.mlp = MultiLayerPerceptron([len(field_dims) * embed_dim, 128, 64])
        self.fc = nn.Linear(embed_dim + 64, 1)

    def forward(self, x):
        # GMF部分
        gmf_output = self.embed_gmf(x)[:, 0] * self.embed_gmf(x)[:, 1]

        # MLP部分
        mlp_input = self.embed_mlp(x).reshape(x.size(0), -1)
        mlp_output = self.mlp(mlp_input)

        # 合并GMF和MLP输出
        concated = torch.cat([gmf_output, mlp_output], dim=1)
        output = torch.sigmoid(self.fc(concated))
        return output


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        # 初始化Embedding层，偏移值用于处理多字段特征
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)

    def forward(self, x):
        # 根据偏移值调整输入以索引Embedding
        x = x + x.new_tensor(self.offsets)
        return self.embedding(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, layers, batch_norm=True):
        super(MultiLayerPerceptron, self).__init__()
        mlp_layers = []
        for i in range(len(layers) - 1):
            mlp_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # 除最后一层外添加激活和BatchNorm
                if batch_norm:
                    mlp_layers.append(nn.BatchNorm1d(layers[i + 1]))
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)
