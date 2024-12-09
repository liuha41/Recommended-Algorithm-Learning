import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error


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

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
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

def Criteo_run(model, train_loader, val_loader, optimizer, loss_fn, epochs, device):
    train_losses, val_aucs, val_rmses = [], [], []
    best_loss = np.inf

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

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


def train(model, train_loader, val_loader, optimizer, loss_fn, epochs, device, dataset, Model_name):
    if dataset == 'criteo':
        Criteo_run(model, train_loader, val_loader, optimizer, loss_fn, epochs, device)