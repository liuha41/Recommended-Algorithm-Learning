import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from Model.DeepFM import DeepFM
from data.dataset import *
from conf import config

# 数据预处理配置
file_path = config.Criteo_DataSet_Path + 'train_10k.txt'
num_features = [f'I{i}' for i in range(1, 14)]
cat_features = [f'C{i}' for i in range(1, 27)]
target_column = "Label"

# 模型与训练配置
embed_dim = 8
hidden_layer_list = [256, 128, 64]
layer_num = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4096

# 数据处理与加载
processor = CriteoDataProcessor(file_path, num_features, cat_features, target_column)
data = processor.load_and_preprocess()
train_data, val_data, test_data = processor.split_data(data)

sparse_fea_list = list(processor.cat_feature_dims.values())
print("类别特征维度:", sparse_fea_list)

train_dataset = CriteoDataset(train_data, num_features, cat_features, target_column)
val_dataset = CriteoDataset(val_data, num_features, cat_features, target_column)
test_dataset = CriteoDataset(test_data, num_features, cat_features, target_column)

train_loader = create_dataloader(train_dataset, batch_size=batch_size)
val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

model = DeepFM(sparse_fea_list)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

# 计算指标函数
def compute_auc_rmse(loader, model, device):
    """计算AUC和RMSE"""
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x_dense, x_sparse, label in loader:
            x_dense, x_sparse, label = x_dense.to(device), x_sparse.to(device), label.to(device)
            pred = model(x_dense, x_sparse).squeeze()
            y_true.append(label.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    rmse_score = mean_squared_error(y_true, y_pred, squared=False)
    return auc_score, rmse_score

# 训练与验证
def train_one_epoch(device):
    """单轮训练"""
    model.train()
    loss_log = []
    for x_dense, x_sparse, label in train_loader:
        x_dense, x_sparse, label = x_dense.to(device), x_sparse.to(device), label.to(device)
        pred = model(x_dense, x_sparse).squeeze()
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    return np.mean(loss_log)

# 主训练逻辑
def run(epochs=100):
    train_losses, val_aucs, val_rmses = [], [], []
    best_loss = np.inf

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(device)

        auc_score, rmse_score = compute_auc_rmse(val_loader, model, device)

        train_losses.append(train_loss)
        val_aucs.append(auc_score)
        val_rmses.append(rmse_score)

        print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}, AUC: {auc_score:.4f}, RMSE: {rmse_score:.4f}")
    if  train_loss < best_loss:
        model.save_model('best_model.pth')

    # 绘制训练结果
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(2, 1, 2)
    plt.plot(val_aucs, label='AUC')
    plt.plot(val_rmses, label='RMSE')
    plt.legend()
    plt.title('Validation Metrics')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
