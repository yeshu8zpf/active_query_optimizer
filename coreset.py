import torch, time
import numpy as np

def compute_similarity_matrix(X, L):
    # 计算X和L之间的余弦相似度矩阵
    norms_X = X.norm(dim=1, keepdim=True)
    normalized_X = X / norms_X
    norms_L = L.norm(dim=1, keepdim=True)
    normalized_L = L / norms_L

    similarity_matrix_XL = torch.mm(normalized_X, normalized_L.t())
    similarity_matrix_XX = torch.mm(normalized_X, normalized_X.t())

    return similarity_matrix_XX, similarity_matrix_XL

def greedy_core_set_selection(X, L, unlabeled_indices, groups, n_selections, similarity_matrix_XX, similarity_matrix_XL):
    S = []  # 最终选中的样本集
    labeled_indices = list(range(L.size(0)))  # L中的索引
    for group_indices in groups:
        group_unlabeled_indices = [idx for idx in group_indices if idx in unlabeled_indices]
        Si = []  # 当前群组选中的样本集

        # 对于每个群组根据公式选择样本
        for _ in range(n_selections):
            min_max_sim = float('inf')
            selected = None
            current_comparison_set = labeled_indices + Si
            for idx in group_unlabeled_indices:
                if idx not in Si:
                    # 计算与L_t和S_i的最大相似度
                    max_sim = max(similarity_matrix_XL[idx, :].max(), similarity_matrix_XX[idx, current_comparison_set].max())
                    if max_sim < min_max_sim:
                        min_max_sim = max_sim
                        selected = idx
            Si.append(selected)
            group_unlabeled_indices.remove(selected)

        S.extend(Si)

    # 更新未标记集
    new_unlabeled_indices = [idx for idx in unlabeled_indices if idx not in S]
    
    return new_unlabeled_indices, S

if __name__ == '__main__':
    t1 = time.time()
    # 示例数据
    torch.manual_seed(0)
    X = torch.randn(20000, 5, device=torch.device('cuda:0'))  # 100个样本，每个样本5个特征
    L_t = torch.randn(1000, 5, device=torch.device('cuda:0'))  # 20个已标记样本
    unlabeled_indices = list(range(20000))  # 假设20-99样本未标记
    groups = [list(range(0, 5000)), list(range(5000, 11000)), list(range(11000, 20000))]  # 三个密度群组
    n_selections = 500  # 每个群组要选择5个样本

    # 计算相似度矩阵
    similarity_matrix_XX, similarity_matrix_XL = compute_similarity_matrix(X, L_t)

    # 执行核心集选择
    new_unlabeled_indices, selected_indices = greedy_core_set_selection(X, L_t, unlabeled_indices, groups, n_selections, similarity_matrix_XX, similarity_matrix_XL)
    print("Selected core-set indices:", selected_indices)
    print(f'time:{time.time()-t1}')
