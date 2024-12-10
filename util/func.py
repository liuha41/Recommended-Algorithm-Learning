import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error


# 计算指标函数
def compute_metrics(loader, model, device, dataset):
    """计算AUC和RMSE"""
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if dataset == 'criteo':
                x_dense, x_sparse, label = batch
                x_dense, x_sparse, label = x_dense.to(device), x_sparse.to(device), label.to(device)
                pred = model(x_dense, x_sparse).squeeze()
            elif dataset == 'amazon-books':
                x, label = batch
                x, label = x.to(device), label.to(device)
                pred = model(x).squeeze()
            else:
                raise ValueError("Unsupported dataset. Use 'criteo' or 'amazon-books'.")
            y_true.append(label.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    rmse_score = mean_squared_error(y_true, y_pred, squared=False)
    return auc_score, rmse_score


def train_one_epoch(model, loader, loss_fn, optimizer, device, dataset):
    """单轮训练"""
    model.train()
    loss_log = []
    for batch in loader:
        if dataset == 'criteo':
            x_dense, x_sparse, label = batch
            x_dense, x_sparse, label = x_dense.to(device), x_sparse.to(device), label.to(device)
            pred = model(x_dense, x_sparse).squeeze()
        elif dataset == 'amazon-books':
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
            label = label.float().detach()
        else:
            raise ValueError("Unsupported dataset. Use 'criteo' or 'amazon-books'.")
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    return np.mean(loss_log)


def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, epochs, device, dataset, model_name="model"):
    """统一训练与评估流程"""
    train_losses, val_aucs, val_rmses = [], [], []
    best_loss = np.inf

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, dataset)
        auc_score, rmse_score = compute_metrics(val_loader, model, device, dataset)

        train_losses.append(train_loss)
        val_aucs.append(auc_score)
        val_rmses.append(rmse_score)

        print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}, AUC: {auc_score:.4f}, RMSE: {rmse_score:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"{model_name}_best.pth")

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

# def train(model, train_loader, val_loader, optimizer, loss_fn, epochs, device, dataset, Model_name):
#     if dataset == 'criteo':
#         Criteo_run(model, train_loader, val_loader, optimizer, loss_fn, epochs, device)
#     elif dataset == 'amazon-books':
#         AB_run(model, train_loader, val_loader, optimizer, loss_fn, epochs, device)