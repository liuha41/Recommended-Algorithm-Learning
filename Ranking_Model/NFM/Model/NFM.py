import torch.nn as nn
import torch.nn.functional as F
import torch

class Dnn(nn.Module):
    """
    Dnn 网络
    """
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 、
                      比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout: 失活率
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)
        return x



class NFM(nn.Module):

    def __init__(self, sparse_fea_list, hidden_units, embed_dim=8):
        """
               DeepCrossing：
                   feature_info: 特征信息（数值特征， 类别特征， 类别特征embedding映射)
                   hidden_units: 列表， 隐藏单元
                   dropout: Dropout层的失活比例
                   embed_dim: embedding维度
               """
        super(NFM, self).__init__()

        self.dense_features = [f'I{i}' for i in range(1, 14)]
        self.sparse_features = [f'C{i}' for i in range(1, 27)]

        self.sparse_fea_list = sparse_fea_list

        self.sparse_features_map = dict(zip(self.sparse_features, self.sparse_fea_list))

        self.embed_layers = nn.ModuleDict(
            {
                'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
                for key, val in self.sparse_features_map.items()
            }
        )

        # 注意 这里的总维度  = 数值型特征的维度 + 离散型变量每个特征要embedding的维度
        dim_sum = len(self.dense_features) + embed_dim
        hidden_units.insert(0, dim_sum)

        # bn
        self.bn = nn.BatchNorm1d(dim_sum)

        # dnn网络
        self.dnn_network = Dnn(hidden_units)

        # dnn的线性层
        self.dnn_final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x_dense, x_sparse):
        # 1、先把输入向量x分成两部分处理、因为数值型和类别型的处理方式不一样
        dense_input, sparse_inputs = x_dense, x_sparse
        # 2、转换为long形
        sparse_inputs = sparse_inputs.long()

        # 2、不同的类别特征分别embedding  [(batch_size, embed_dim)]
        sparse_embeds = [
            self.embed_layers['embed_' + key](sparse_inputs[:, i]) for key, i in
            zip(self.sparse_features_map.keys(), range(sparse_inputs.shape[1]))
        ]
        # 3、embedding进行堆叠
        sparse_embeds = torch.stack(sparse_embeds) # (离散特征数, batch_size, embed_dim)
        sparse_embeds = sparse_embeds.permute((1,0,2))  # (batch_size, 离散特征数, embed_dim)

        # 这里得到embedding向量 sparse_embeds的shape为(batch_size, 离散特征数, embed_dim)
        # 然后就进行特征交叉层，按照特征交叉池化层化简后的公式  其代码如下
        # 注意：
        # 公式中的x_i乘以v_i就是 embedding后的sparse_embeds
        # 通过设置dim=1,把dim=1压缩（行的相同位置相加、去掉dim=1），即进行了特征交叉
        embed_cross = 1 / 2 * (
                torch.pow(torch.sum(sparse_embeds, dim=1), 2) - torch.sum(torch.pow(sparse_embeds, 2), dim=1)
        )  # (batch_size, embed_dim)

        # 4、数值型和类别型特征进行拼接  (batch_size, embed_dim + dense_input维度 )
        x = torch.cat([embed_cross, dense_input], dim=-1)

        x = self.bn(x)

        # Dnn部分，使用全部特征
        dnn_out = self.dnn_final_linear(self.dnn_network(x))

        # out
        outputs = torch.sigmoid(dnn_out)

        return outputs

if __name__ == '__main__':
    x = torch.rand(size=(2, 5), dtype=torch.float32)
    feature_info = [
        ['I1', 'I2'],  # 连续性特征
        ['C1', 'C2', 'C3'],  # 离散型特征
        {
            'C1': 20,
            'C2': 20,
            'C3': 20
        }
    ]
    # 建立模型
    hidden_units = [128, 64, 32]

    net = NFM(feature_info, hidden_units)
    print(net)
    print(net(x))
