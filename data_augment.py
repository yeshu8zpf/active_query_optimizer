import numpy as np
import copy
from collections import defaultdict

def augment_sql_encodings(sql_encodings, join_list, filter_columns, clusters, total_new_samples):
    """
    根据给定的 SQL 编码和聚类结果，按簇的占比分配新样本数量，生成扩充的数据集。

    参数：
    - sql_encodings: 原始 SQL 编码的列表，每个编码是一个向量，包含连接的 join embedding 和 filter embedding。
    - join_list: 所有可能的连接条件列表。
    - filter_columns: 过滤条件涉及的所有列的列表。
    - clusters: 一个字典，键为簇的 ID，值为属于该簇的 SQL 编码的索引列表。
    - total_new_samples: 要生成的新样本总数。

    返回：
    - augmented_encodings: 扩充后的 SQL 编码列表。
    """
    augmented_encodings = []
    total_samples = sum(len(indices) for indices in clusters.values())

    # 计算每个簇的扩充样本数量
    cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in clusters.items()}
    total_original_samples = sum(cluster_sizes.values())

    # 计算每个簇的扩充比例，簇占比越高，扩充的样本数越少
    cluster_proportions = {cluster_id: size / total_original_samples for cluster_id, size in cluster_sizes.items()}

    # 计算每个簇需要生成的新样本数量
    total_proportions = sum(1 - proportion for proportion in cluster_proportions.values())
    cluster_new_samples = {
        cluster_id: int(((1 - proportion) / total_proportions) * total_new_samples)
        for cluster_id, proportion in cluster_proportions.items()
    }

    # 如果由于取整导致总数不足，补充到总的新样本数
    allocated_samples = sum(cluster_new_samples.values())
    remaining_samples = total_new_samples - allocated_samples
    if remaining_samples > 0:
        # 将剩余的样本随机分配给簇
        cluster_ids = list(clusters.keys())
        for i in range(remaining_samples):
            cluster_id = np.random.choice(cluster_ids)
            cluster_new_samples[cluster_id] += 1

    # 假设 join embedding 和 filter embedding 的长度
    join_embedding_length = len(join_list)
    filter_embedding_length = len(filter_columns) * 3  # 每个过滤列有三个值

    for cluster_id, indices in clusters.items():
        cluster_encodings = [sql_encodings[i] for i in indices]

        # 计算当前簇需要生成的新样本数量
        num_new_samples = cluster_new_samples[cluster_id]
        if num_new_samples <= 0:
            continue

        # 建立索引以便快速查找具有相同 join 和 filter 结构的 SQL 编码
        encoding_dict = defaultdict(list)

        for encoding in cluster_encodings:
            # 分割 join embedding 和 filter embedding
            join_embedding = encoding[:join_embedding_length]
            filter_embedding = encoding[join_embedding_length:]

            # 提取 join 条件
            join_conditions = tuple(join_embedding)

            # 提取 filter 条件的结构（是否有过滤条件）
            filter_structure = []
            for i in range(0, len(filter_embedding), 3):
                has_filter = filter_embedding[i]
                filter_structure.append(has_filter)
            filter_structure = tuple(filter_structure)

            key = (join_conditions, filter_structure)
            encoding_dict[key].append(encoding)

        # 对于每个键，生成新样本
        for key, encodings_list in encoding_dict.items():
            if len(encodings_list) >= 2:
                # 根据现有的过滤值范围生成新样本
                # 首先，收集每个过滤列的值范围
                filter_ranges = {}
                for idx in range(len(filter_columns)):
                    values = []
                    for encoding in encodings_list:
                        filter_embedding = encoding[join_embedding_length:]
                        i = idx * 3
                        has_filter = filter_embedding[i]
                        if has_filter:
                            lower_bound = filter_embedding[i + 1]
                            upper_bound = filter_embedding[i + 2]
                            values.append((lower_bound, upper_bound))
                    if values:
                        min_lower = min(v[0] for v in values)
                        max_upper = max(v[1] for v in values)
                        filter_ranges[idx] = (min_lower, max_upper)

                # 根据需要的样本数量，生成新样本
                samples_per_key = int(num_new_samples / len(encoding_dict))
                for _ in range(samples_per_key):
                    template_encoding = encodings_list[0]
                    new_encoding = copy.deepcopy(template_encoding)
                    filter_embedding = new_encoding[join_embedding_length:]

                    for idx, (min_lower, max_upper) in filter_ranges.items():
                        i = idx * 3
                        has_filter = filter_embedding[i]
                        if has_filter:
                            # 在范围内随机生成新的下界和上界
                            new_lower = np.random.uniform(min_lower, max_upper)
                            new_upper = np.random.uniform(new_lower, max_upper)
                            filter_embedding[i + 1] = new_lower
                            filter_embedding[i + 2] = new_upper

                    # 将修改后的 filter_embedding 赋值回去
                    new_encoding[join_embedding_length:] = filter_embedding
                    augmented_encodings.append(new_encoding)
            else:
                # 仅有一个编码，通过调整过滤值生成新样本
                original_encoding = encodings_list[0]
                samples_per_key = int(num_new_samples / len(encoding_dict))
                for _ in range(samples_per_key):
                    new_encoding = copy.deepcopy(original_encoding)
                    filter_embedding = new_encoding[join_embedding_length:]

                    for idx in range(len(filter_columns)):
                        i = idx * 3
                        has_filter = filter_embedding[i]
                        if has_filter:
                            lower_bound = filter_embedding[i + 1]
                            upper_bound = filter_embedding[i + 2]
                            # 在一定范围内调整过滤值
                            new_lower = lower_bound * np.random.uniform(0.9, 1.1)
                            new_upper = upper_bound * np.random.uniform(0.9, 1.1)
                            if new_lower > new_upper:
                                new_lower, new_upper = new_upper, new_lower
                            filter_embedding[i + 1] = new_lower
                            filter_embedding[i + 2] = new_upper

                    new_encoding[join_embedding_length:] = filter_embedding
                    augmented_encodings.append(new_encoding)

    return augmented_encodings

# 示例数据
sql_encodings = [
    # 每个编码是一个向量，join embedding 和 filter embedding 已经连接
    # 假设 join embedding 长度为 4，filter embedding 长度为 6（两个过滤列，每列3个值）
    np.array([1, 0, 1, 0, 1, 10, 20, 1, 5, 15]),  # 编码 0
    np.array([1, 0, 1, 0, 1, 15, 25, 1, 5, 15]),  # 编码 1
    # 可以添加更多的原始 SQL 编码
]

join_list = ['a JOIN b ON a.id = b.id', 'b JOIN c ON b.id = c.id', 'c JOIN d ON c.id = d.id', 'd JOIN e ON d.id = e.id']
filter_columns = ['a.a1', 'b.b1']

# 示例聚类结果
clusters = {
    0: [0, 1],  # 簇 0 包含索引为 0 和 1 的 SQL 编码
    # 可以添加更多的簇
}

total_new_samples = 10  # 希望生成的新样本总数

augmented_encodings = augment_sql_encodings(sql_encodings, join_list, filter_columns, clusters, total_new_samples)

# 输出结果
for encoding in augmented_encodings:
    print(encoding)
