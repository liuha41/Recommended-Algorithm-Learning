import numpy as np


def get_data(path, num_users, num_items, num_total_ratings, train_ratio):
    # 读取所有评分数据
    with open(f"{path}ratings.dat", 'r') as fp:
        lines = fp.readlines()

    # 初始化矩阵和集合
    train_r = np.zeros((num_users, num_items), dtype=int)
    test_r = np.zeros((num_users, num_items), dtype=int)
    train_mask_r = np.zeros((num_users, num_items), dtype=bool)
    test_mask_r = np.zeros((num_users, num_items), dtype=bool)

    user_train_set = set()
    item_train_set = set()
    user_test_set = set()
    item_test_set = set()

    # 设置随机种子并生成随机索引
    np.random.seed(42)
    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[:int(num_total_ratings * train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

    def process_lines(idx_list, rating_matrix, mask_matrix, user_set, item_set):
        for idx in idx_list:
            line = lines[idx]
            try:
                user, item, rating, _ = line.strip().split("::")
                user_idx, item_idx = int(user) - 1, int(item) - 1
                rating_matrix[user_idx, item_idx] = int(rating)
                mask_matrix[user_idx, item_idx] = True
                user_set.add(user_idx)
                item_set.add(item_idx)
            except ValueError:
                print(f"Invalid line format: {line}")

    # 处理训练数据和测试数据
    process_lines(train_idx, train_r, train_mask_r, user_train_set, item_train_set)
    process_lines(test_idx, test_r, test_mask_r, user_test_set, item_test_set)

    return train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set