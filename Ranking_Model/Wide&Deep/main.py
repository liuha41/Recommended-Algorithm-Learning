import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from Model.Wide_Deep import WideDeep
from Dataset.CriteoDataset import *
from conf.config import *
from util.func import *

def main(args):
    # 数据处理与加载
    processor = CriteoDataProcessor(args.file_path, args.num_features, args.cat_features, args.target_column)
    data = processor.load_and_preprocess()
    train_data, val_data, test_data = processor.split_data(data)

    sparse_fea_list = list(processor.cat_feature_dims.values())

    train_dataset = CriteoDataset(train_data, args.num_features, args.cat_features, args.target_column)
    val_dataset = CriteoDataset(val_data, args.num_features, args.cat_features, args.target_column)
    test_dataset = CriteoDataset(test_data, args.num_features, args.cat_features, args.target_column)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型初始化
    model = WideDeep(len(args.num_features), len(args.cat_features), sparse_fea_list, args.embed_dim, args.hidden_layer_list).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss()

    # 模型训练
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, args.epochs, args.device, args.dataset_name,
          args.model_name)

if __name__ == "__main__":
    # 定义参数
    parser = argparse.ArgumentParser(description="模型训练超参数管理")

    # 数据相关参数
    parser.add_argument('--file_path', type=str, default=Criteo_DataSet_Path + 'criteo-100k.txt', help='Criteo数据文件路径')
    parser.add_argument('--num_features', type=list, default=[f'I{i}' for i in range(1, 14)], help='数值特征名称')
    parser.add_argument('--cat_features', type=list, default=[f'C{i}' for i in range(1, 27)], help='类别特征名称')
    parser.add_argument('--target_column', type=str, default='Label', help='目标列名称')

    # 模型相关参数
    parser.add_argument('--embed_dim', type=int, default=8, help='嵌入维度')
    parser.add_argument('--hidden_layer_list', type=list, default=[256, 128, 64], help='隐藏层结构')
    parser.add_argument('--batch_size', type=int, default=4096, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备类型')

    # 其他参数
    parser.add_argument('--dataset_name', type=str, default='criteo', help='数据集名称')
    parser.add_argument('--model_name', type=str, default='Wide&Deep', help='模型名称')

    args = parser.parse_args()
    main(args)
