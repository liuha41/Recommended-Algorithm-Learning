import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data.dataset import get_data
from Model.NeuralCF import NeuralCF
from conf import config

# 配置设备和路径
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = config.ml_1m_DataSet_Path + "ratings.dat"

# 参数设置
EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCHS = 500

# 加载数据
field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = get_data(data_path, device)

# 初始化模型、优化器和损失函数
model = NeuralCF(field_dims, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
criterion = nn.BCELoss()

# 数据加载器生成器
def create_dataloader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.long).to(device),
                            torch.tensor(y, dtype=torch.float32).to(device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 训练函数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 验证和测试函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    rmse = (total_loss / len(dataloader)) ** 0.5
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy, rmse

# 创建数据加载器
train_loader = create_dataloader(train_X, train_y, BATCH_SIZE, shuffle=True)
valid_loader = create_dataloader(valid_X, valid_y, BATCH_SIZE, shuffle=False)
test_loader = create_dataloader(test_X, test_y, BATCH_SIZE, shuffle=False)

# 训练和验证
train_losses, valid_losses, valid_rmses = [], [], []
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    valid_loss, valid_accuracy, valid_rmse = evaluate(model, valid_loader, criterion)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_rmses.append(valid_rmse)

    # 保存最佳模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, "
          f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid RMSE: {valid_rmse:.4f}")

# 测试阶段
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_accuracy, test_rmse = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test RMSE: {test_rmse:.4f}")

# 绘制训练过程曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(valid_rmses, label='Valid RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.title('RMSE Curve')

plt.tight_layout()
plt.show()
