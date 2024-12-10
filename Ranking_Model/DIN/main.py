import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from Model.DIN import DeepInterestNet
from Dataset.AmazonBookDataset import *
from conf.config import *
from util.func import *

def main(args):
    # 数据处理与加载
    train_loader, val_loader, test_loader, fields = create_amazon_book_dataloader(args.file_path, args.batch_size)

    print(fields)

    # 模型初始化
    model = DeepInterestNet(feature_dim=fields, embed_dim=args.embed_dim, mlp_dims=args.hidden_layer_list, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss()

    # 模型训练
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, args.epochs, args.device, args.dataset_name,
          args.model_name)

if __name__ == "__main__":
    # 定义参数
    parser = argparse.ArgumentParser(description="模型训练超参数管理")

    # 数据相关参数
    parser.add_argument('--file_path', type=str, default=DataSet_Root + 'amazon-books-100k.txt', help='Criteo数据文件路径')
    # 模型相关参数
    parser.add_argument('--embed_dim', type=int, default=8, help='嵌入维度')
    parser.add_argument('--hidden_layer_list', type=list, default=[64, 32], help='隐藏层结构')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.2, help='失活率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备类型')

    # 其他参数
    parser.add_argument('--dataset_name', type=str, default='amazon-books', help='数据集名称')
    parser.add_argument('--model_name', type=str, default='DIN', help='模型名称')

    args = parser.parse_args()
    main(args)
