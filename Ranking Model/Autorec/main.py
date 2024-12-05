import torch
import numpy as np
import math
import argparse
import torch.utils.data as Data
import torch.optim as optim
from data.dataset import get_data
from Model.AutoRec import AutoRec
from conf import config
import matplotlib.pyplot as plt


def train(epoch, model, loader, optimizer, train_mask_r, train_rmse_list, train_loss_list):
    model.train()
    total_rmse, total_loss = 0, 0

    for step, (batch_x, batch_mask_x, _) in enumerate(loader):
        batch_x = batch_x.type(torch.FloatTensor).cuda()
        batch_mask_x = batch_mask_x.type(torch.FloatTensor).cuda()

        decoder = model(batch_x)
        loss, rmse = model.loss(decoder=decoder, input=batch_x, optimizer=optimizer, mask_input=batch_mask_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += rmse

    avg_rmse = np.sqrt(total_rmse.detach().cpu().numpy() / (train_mask_r == 1).sum())
    train_rmse_list.append(avg_rmse)
    train_loss_list.append(total_loss)

    print(f"Epoch {epoch + 1} - Train RMSE: {avg_rmse:.4f}, Loss: {total_loss:.4f}")


def evaluate(epoch, model, test_r, test_mask_r, user_test_set, user_train_set, item_test_set, item_train_set, test_rmse_list):
    model.eval()
    test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor).cuda()
    test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor).cuda()

    with torch.no_grad():
        decoder = model(test_r_tensor)

    unseen_users = list(user_test_set - user_train_set)
    unseen_items = list(item_test_set - item_train_set)

    for user in unseen_users:
        for item in unseen_items:
            if test_mask_r[user, item] == 1:
                decoder[user, item] = 3

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum().item()
    rmse = np.sqrt(mse / (test_mask_r == 1).sum())
    test_rmse_list.append(rmse)

    print(f"Epoch {epoch + 1} - Test RMSE: {rmse:.4f}")


def plot_results(train_loss, train_rmse, test_rmse):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # RMSE curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_rmse, label='Train RMSE', marker='o')
    plt.plot(epochs, test_rmse, label='Test RMSE', marker='x')
    plt.title('RMSE Curve')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I-AutoRec ')
    parser.add_argument('--hidden_units', type=int, default=500)
    parser.add_argument('--lambda_value', type=float, default=1)
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")
    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=1)
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Dataset and parameters
    dataset_path = config.ml_1m_DataSet_Path
    num_users = 6040
    num_items = 3952
    num_total_ratings = 1000209
    train_ratio = 0.9

    # Load dataset
    train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, \
        item_test_set = get_data(dataset_path, num_users, num_items, num_total_ratings, train_ratio)

    # Check CUDA availability
    args.cuda = torch.cuda.is_available()

    rec = AutoRec(args, num_users, num_items)
    if args.cuda:
        rec.cuda()

    optimer = optim.Adam(rec.parameters(), lr=args.base_lr, weight_decay=1e-4)

    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r),
                                       torch.from_numpy(train_r))
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)

    train_loss_list, train_rmse_list, test_rmse_list = [], [], []

    for epoch in range(args.train_epoch):
        train(epoch, rec, loader, optimer, train_mask_r, train_rmse_list, train_loss_list)
        evaluate(epoch, rec, test_r, test_mask_r, user_test_set, user_train_set, item_test_set, item_train_set, test_rmse_list)

    plot_results(train_loss_list, train_rmse_list, test_rmse_list)
