import numpy as np
import torch, time
from sklearn.cluster import KMeans
import numpy as np

def kmeans_partition(U, num_groups):
    """
    使用 K-means 将数据分组
    :param U: 输入数据矩阵，形状为 (N, D)，N 为样本数量，D 为特征维度
    :param num_groups: 需要划分的组数 (即 K-means 的 K 值)
    :return: 分组后的结果，列表形式
    """
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=num_groups, random_state=42)
    labels = kmeans.fit_predict(U)  # 获取每个样本的簇标签

    # 根据标签分组
    groups = [[] for _ in range(num_groups)]
    for idx, label in enumerate(labels):
        groups[label].append(idx)  # 将样本添加到对应簇的组中

    return groups


def greedy_core_set_selection(L, U_groups, n_selections_per_group):
    selected_indices_per_group = []
    L_T = L.t()

    # sim_with_L_list = []
    # max_sim_L_list = []
    # for U in U_groups:
    #     if len(U) == 0:
    #         sim_with_L_list.append(None)
    #         max_sim_L_list.append(None)
    #         continue
    #     sim_with_L = torch.matmul(U, L_T)
    #     sim_with_L_list.append(sim_with_L)
    #     max_sim_L, _ = sim_with_L.max(dim=1)
    #     max_sim_L_list.append(max_sim_L) # [max dist with labeled samples]

    for group_idx, U in enumerate(U_groups):
        if len(U) == 0:
            selected_indices_per_group.append([])
            continue
        selected_indices = []
        selected_vectors = []

        candidate_indices = torch.arange(U.size(0))
        # max_sim = max_sim_L_list[group_idx].clone()

        max_sim = torch.ones(U.size(0),1)
        for _ in range(min(n_selections_per_group, U.size(0))):
            min_max_sim, min_idx = torch.min(max_sim[candidate_indices], dim=0)
            selected_idx = candidate_indices[min_idx].item()

            selected_indices.append(selected_idx)
            selected_vectors.append(U[selected_idx])

            candidate_indices = candidate_indices[candidate_indices != selected_idx]

            if len(selected_vectors) < n_selections_per_group and len(candidate_indices) > 0:
                maz_sim = torch.matmul(U[candidate_indices], U[selected_idx].unsqueeze(1)).squeeze(1)
                # max_sim[candidate_indices] = torch.max(max_sim[candidate_indices], new_sim)

        selected_indices_per_group.append(selected_indices)

    return selected_indices_per_group


def standardize_and_normalize(U, L):
    """
    对特征矩阵进行标准化和归一化。

    参数：
    - X: torch.Tensor，形状为 (样本数量, 特征数量)

    返回：
    - X_normalized: 标准化并归一化后的特征矩阵，形状与 X 相同
    """
    # 标准化（对每个特征减去均值，除以标准差）
    # 计算每个特征的均值和标准差
    N_U = len(U)
    X = torch.concatenate(U, L)
    means = X.mean(dim=0, keepdim=True)        # Shape: (1, 特征数量)
    stds = X.std(dim=0, unbiased=False, keepdim=True)  # Shape: (1, 特征数量)
    
    # 避免除以零，对于标准差为零的特征，设置为1
    stds[stds == 0] = 1.0

    X_standardized = (X - means) / stds

    # 归一化（对每个样本除以其范数）
    # 计算每个样本的范数（L2 范数）
    norms = X_standardized.norm(p=2, dim=1, keepdim=True)  # Shape: (样本数量, 1)

    # 避免除以零，对于范数为零的样本，设置为1
    norms[norms == 0] = 1.0

    X_normalized = X_standardized / norms

    return X_normalized[:N_U], X_normalized[N_U:]



if __name__ == '__main__':
    # 示例数据
    N_L = 1000
    N_U = 10000
    D = 5
    quantize_to_bins = True

    torch.manual_seed(0)

    L = torch.randn(N_L, D)

    U = torch.randn(N_U, D)

    # 使用 LSH 将未标记样本分组
    num_groups = 32

    # 示例数据，生成100个样本，每个样本128维度
    data = np.random.rand(100, 128)

    # 使用 Annoy 的 LSH 将数据分为 10 组，使用欧氏距离
    groups = kmeans_partition(U, num_groups=10)

    # 打印分组结果
    for i, group in enumerate(groups):
        print(f"Group {i}: {len(group)}")

