import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetwork(nn.Module):
    """
    Cross Network: Performs explicit feature crossing
    """

    def __init__(self, layer_num, input_dim):
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1)) for _ in range(layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(layer_num)
        ])

    def forward(self, x):
        """
        Args:
        - x: Input tensor of shape (batch_size, input_dim)
        Returns:
        - Tensor of shape (batch_size, input_dim)
        """
        x_0 = x  # Keep the original input for use in each layer
        for i in range(self.layer_num):
            x_w = torch.matmul(x, self.cross_weights[i])  # (batch_size, 1)
            x = x_0 * x_w.squeeze(1).unsqueeze(1) + self.cross_bias[i] + x  # Element-wise multiply and add bias
        return x


class Dnn(nn.Module):
    """
    Deep Neural Network Module
    """

    def __init__(self, hidden_units, dropout=0.):
        super(Dnn, self).__init__()
        layers = []
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)


class DCN(nn.Module):
    def __init__(self, sparse_fea_list, dense_fea_dim, embed_dim, hidden_units, layer_num, dnn_dropout=0.):
        """
        Deep & Cross Network for Criteo dataset.

        Args:
        - sparse_fea_list (list): List of categorical feature sizes.
        - dense_fea_dim (int): Dimension of dense (numerical) features.
        - embed_dim (int): Dimension of embedding vectors for categorical features.
        - hidden_units (list): List of DNN hidden layer sizes.
        - layer_num (int): Number of layers in the cross network.
        - dnn_dropout (float): Dropout rate in the DNN layers.
        """
        super(DCN, self).__init__()
        self.sparse_fea_list = sparse_fea_list
        self.embed_layers = nn.ModuleList([
            nn.Embedding(feat_size, embed_dim) for feat_size in sparse_fea_list
        ])
        input_dim = dense_fea_dim + len(sparse_fea_list) * embed_dim
        hidden_units.insert(0, input_dim)

        self.cross_network = CrossNetwork(layer_num, input_dim)
        self.dnn_network = Dnn(hidden_units, dnn_dropout)
        self.final_linear = nn.Linear(hidden_units[-1] + input_dim, 1)

    def forward(self, x_dense, x_sparse):
        sparse_embeds = [
            self.embed_layers[i](x_sparse[:, i]) for i in range(len(self.sparse_fea_list))
        ]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)  # (batch_size, len(sparse_fea_list) * embed_dim)

        x = torch.cat([x_dense, sparse_embeds], dim=-1)

        cross_out = self.cross_network(x)

        dnn_out = self.dnn_network(x)

        total_x = torch.cat([cross_out, dnn_out], dim=-1)

        outputs = torch.sigmoid(self.final_linear(total_x)).squeeze(1)
        return outputs
