import numpy as np
from copy import deepcopy
from collections import defaultdict

def collect_filter_positions(node, filter_length, filter_positions):
    """
    递归地收集节点特征中过滤值的位置。

    参数：
    - node: 当前节点。
    - filter_length: 过滤特征的长度。
    - filter_positions: 用于收集过滤值位置的集合。
    """
    if node is None:
        return

    features = node.get_feature()

    # 过滤特征位于特征向量的最后 filter_length 个元素
    filter_features = features[-filter_length:]

    # 每个过滤条件占据 7 个元素（6 个操作符 + 1 个过滤值）
    num_filters = filter_length // 7

    for i in range(num_filters):
        start_idx = i * 7
        operator_flags = filter_features[start_idx:start_idx + 6]
        value_idx = start_idx + 6  # 第 7 个元素为过滤值

        # 检查是否有操作符被设置为 1
        if np.any(operator_flags):
            # 记录过滤值的位置（在过滤特征中的索引）
            filter_positions.add(value_idx)

    # 递归遍历左子节点和右子节点
    collect_filter_positions(node.left, filter_length, filter_positions)
    collect_filter_positions(node.right, filter_length, filter_positions)

def adjust_filters_in_tree(node, filter_length, adjustment_factors):
    """
    递归地调整节点特征中的过滤值，使用给定的调整因子。

    参数：
    - node: 当前节点。
    - filter_length: 过滤特征的长度。
    - adjustment_factors: 过滤值位置到调整因子的映射。
    """
    if node is None:
        return

    features = node.get_feature().copy()  # 复制以避免修改原始特征
    filter_features = features[-filter_length:]

    num_filters = filter_length // 7

    for i in range(num_filters):
        start_idx = i * 7
        operator_flags = filter_features[start_idx:start_idx + 6]
        value_idx = start_idx + 6  # 第 7 个元素为过滤值

        # 检查是否有操作符被设置为 1
        if np.any(operator_flags):
            if value_idx in adjustment_factors:
                factor = adjustment_factors[value_idx]
                filter_features[value_idx] *= factor

    features[-filter_length:] = filter_features
    node.set_feature(features)

    # 递归遍历左子节点和右子节点
    adjust_filters_in_tree(node.left, filter_length, adjustment_factors)
    adjust_filters_in_tree(node.right, filter_length, adjustment_factors)

def augment_plan_pair_encodings(plan_pair_encodings, filter_length, clusters, total_new_samples):
    """
    通过同步调整过滤值，生成新的计划对编码。

    参数：
    - plan_pair_encodings: 原始的计划对编码列表，每个元素是一个包含两个树的元组。
    - filter_length: 过滤特征的长度。
    - clusters: 一个字典，键为簇的 ID，值为属于该簇的计划对的索引列表。
    - total_new_samples: 要生成的新样本总数。

    返回：
    - augmented_plan_pairs: 扩充后的计划对编码列表。
    """
    augmented_plan_pairs = []

    # 计算每个簇的大小和总的计划对数量
    cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in clusters.items()}
    total_plan_pairs = sum(cluster_sizes.values())

    # 计算每个簇的占比
    cluster_proportions = {cluster_id: size / total_plan_pairs for cluster_id, size in cluster_sizes.items()}

    # 计算每个簇需要生成的新样本数量（簇占比越小，生成的样本数越多）
    total_proportions = sum(1 - proportion for proportion in cluster_proportions.values())
    cluster_new_samples = {
        cluster_id: int(((1 - proportion) / total_proportions) * total_new_samples)
        for cluster_id, proportion in cluster_proportions.items()
    }

    # 处理由于取整导致的样本数量不足问题
    allocated_samples = sum(cluster_new_samples.values())
    remaining_samples = total_new_samples - allocated_samples
    if remaining_samples > 0:
        cluster_ids = list(clusters.keys())
        for _ in range(remaining_samples):
            cluster_id = np.random.choice(cluster_ids)
            cluster_new_samples[cluster_id] += 1

    # 对于每个簇
    for cluster_id, indices in clusters.items():
        num_new_samples = cluster_new_samples[cluster_id]
        if num_new_samples <= 0:
            continue

        plan_pairs_in_cluster = [plan_pair_encodings[i] for i in indices]

        # 计算每个计划对需要生成的新样本数量
        num_plan_pairs = len(plan_pairs_in_cluster)
        if num_plan_pairs == 0:
            continue

        samples_per_pair = max(num_new_samples // num_plan_pairs, 1)

        for plan_pair in plan_pairs_in_cluster:
            left_tree, right_tree = plan_pair

            for _ in range(samples_per_pair):
                # 收集两个计划中的过滤值位置
                filter_positions = set()
                collect_filter_positions(left_tree, filter_length, filter_positions)
                collect_filter_positions(right_tree, filter_length, filter_positions)

                # 为每个过滤位置生成相同的调整因子
                adjustment_factors = {pos: np.random.uniform(0.9, 1.1) for pos in filter_positions}

                # 深拷贝原始树，避免修改原始数据
                new_left_tree = deepcopy(left_tree)
                new_right_tree = deepcopy(right_tree)

                # 在两个树中同步调整过滤值
                adjust_filters_in_tree(new_left_tree, filter_length, adjustment_factors)
                adjust_filters_in_tree(new_right_tree, filter_length, adjustment_factors)

                # 将新的计划对添加到列表中
                augmented_plan_pairs.append((new_left_tree, new_right_tree))

    return augmented_plan_pairs

# 示例节点类
class Node:
    def __init__(self, feature, left=None, right=None):
        self.feature = feature  # 节点特征向量
        self.left = left        # 左子节点
        self.right = right      # 右子节点

    def get_feature(self):
        return self.feature

    def set_feature(self, feature):
        self.feature = feature

# 示例使用
if __name__ == "__main__":
    # 原本的sample_entity需要在__init__函数中添加定义self.feature=None, 还需要添加方法set_feature(feature)

    # 假设每个节点的特征向量长度为 13，过滤特征长度为 7（一个过滤列，6 个操作符 + 1 个过滤值）
    filter_length = 7

    # 创建示例树
    # 创建叶子节点
    # 过滤特征：[操作符标志（6 个），过滤值]
    leaf_node1 = Node(np.array([1, 2, 3, 4, 5, 6, 0, 1, 0, 0, 0, 0, 0.3]))  # 有 '<' 操作符，值为 0.3
    leaf_node2 = Node(np.array([1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 0, 0, 0.5]))  # 有 '=' 操作符，值为 0.5

    # 创建根节点
    root_node1 = Node(np.array([1, 2, 3, 4, 5, 6, 0, 0, 1, 0, 0, 0, 0.7]), left=leaf_node1, right=leaf_node2)

    # 创建另一个示例树
    leaf_node3 = Node(np.array([1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 0, 0, 0.2]))  # 有 '>' 操作符，值为 0.2
    leaf_node4 = Node(np.array([1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 1, 0, 0.6]))  # 有 '>=' 操作符，值为 0.6

    root_node2 = Node(np.array([1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 0, 0, 0.8]), left=leaf_node3, right=leaf_node4)

    # 创建计划对编码列表
    plan_pair_encodings = [
        (root_node1, root_node2),
        # 可以添加更多的计划对
    ]

    # 示例聚类结果
    clusters = {
        0: [0],  # 簇 0 包含索引为 0 的计划对
        # 可以添加更多的簇
    }

    total_new_samples = 5  # 希望生成的新样本总数

    # 执行数据增强
    augmented_plan_pairs = augment_plan_pair_encodings(plan_pair_encodings, filter_length, clusters, total_new_samples)

    # 输出结果
    print(f"原始计划对数量：{len(plan_pair_encodings)}")
    print(f"扩充后的计划对数量：{len(augmented_plan_pairs)}")

    # 可以检查增强后的计划对的节点特征
    for idx, (left_tree, right_tree) in enumerate(augmented_plan_pairs):
        print(f"\n增强后的计划对 {idx + 1}:")
        print("左树根节点特征：", left_tree.get_feature())
        print("左树左子节点特征：", left_tree.left.get_feature())
        print("左树右子节点特征：", left_tree.right.get_feature())
        print("右树根节点特征：", right_tree.get_feature())
        print("右树左子节点特征：", right_tree.left.get_feature())
        print("右树右子节点特征：", right_tree.right.get_feature())
