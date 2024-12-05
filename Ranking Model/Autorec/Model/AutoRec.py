import torch
import torch.nn as nn
import torch.optim as optim

class AutoRec(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(AutoRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = args.hidden_units
        self.lambda_value = args.lambda_value

        # 定义编码器和解码器
        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )
        self.decoder = nn.Linear(self.hidden_units, self.num_items)

    def forward(self, input_data):
        # 前向传播
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded

    def loss(self, decoder, input, optimizer, mask_input):
        # 计算损失，包括均方误差和L2正则化项（修正版）
        mse_loss = ((decoder - input) * mask_input).pow(2).sum()
        l2_reg = sum(torch.norm(param, p=2) ** 2 for param in optimizer.param_groups[0]['params'] if param.dim() == 2)
        total_loss = mse_loss + self.lambda_value * 0.5 * l2_reg
        rmse = torch.sqrt(mse_loss)
        return total_loss, rmse
