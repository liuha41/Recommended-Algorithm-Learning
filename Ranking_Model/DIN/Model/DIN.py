import torch
import torch.nn as nn

class Dice(nn.Module):
    """Dice 激活函数，用于调节正负样本在神经网络中的比例"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9

    def forward(self, x):
        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        return self.alpha * x * (1 - p) + x * p


class ActivationUnit(nn.Module):
    """计算目标商品与用户行为的注意力系数"""
    def __init__(self, embedding_dim, dropout=0.2, fc_dims=[32, 16]):
        super().__init__()
        layers = []
        input_dim = embedding_dim * 4  # 输入为4倍embedding维度（拼接+相减+相乘）
        for fc_dim in fc_dims:
            layers.append(nn.Linear(input_dim, fc_dim))
            layers.append(Dice())
            layers.append(nn.Dropout(dropout))
            input_dim = fc_dim
        layers.append(nn.Linear(input_dim, 1))  # 输出维度为1
        self.fc = nn.Sequential(*layers)

    def forward(self, query, user_behavior):
        seq_len = user_behavior.shape[1]
        queries = query.repeat(1, seq_len, 1)
        features = torch.cat(
            [queries, user_behavior, queries - user_behavior, queries * user_behavior],
            dim=-1
        )
        return self.fc(features)


class AttentionPoolingLayer(nn.Module):
    """通过注意力机制提取用户兴趣表示"""
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.activation_unit = ActivationUnit(embedding_dim, dropout)

    def forward(self, query, user_behavior, mask):
        attn_scores = self.activation_unit(query, user_behavior)
        weighted_behavior = user_behavior * attn_scores * mask
        return weighted_behavior.sum(dim=1)


class DeepInterestNet(nn.Module):
    """用户兴趣建模与购买预测模型"""
    def __init__(self, feature_dim, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = nn.Embedding(feature_dim + 1, embed_dim)
        self.attention_pooling = AttentionPoolingLayer(embed_dim, dropout)
        layers = []
        input_dim = embed_dim * 2
        for mlp_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, mlp_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = mlp_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        behaviors_x = x[:, :-1]  # 用户历史行为
        mask = (behaviors_x > 0).float().unsqueeze(-1)  # 记录非填充位置
        ads_x = x[:, -1]  # 推荐目标
        query_ad = self.embedding(ads_x).unsqueeze(1)
        user_behavior = self.embedding(behaviors_x) * mask
        user_interest = self.attention_pooling(query_ad, user_behavior, mask)
        combined_input = torch.cat([user_interest, query_ad.squeeze(1)], dim=1)
        out = self.mlp(combined_input)
        return torch.sigmoid(out.squeeze(1))
