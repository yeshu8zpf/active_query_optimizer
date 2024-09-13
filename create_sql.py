import re, os, random, json
from collections import Counter, defaultdict

def read_query(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        qs = [line.split('#####')[1] for line in lines]
    return qs

def extract_conditions(sql_list):
    """
    从一组 SQL 语句中提取连接条件和筛选条件，返回集合形式。
    
    参数:
    sql_list (list of str): 包含多条 SQL 语句的列表。
    
    返回:
    tuple: 包含两个集合，第一个集合是所有连接条件，第二个集合是所有筛选条件中的 lefthand 表达式。
    """
    join_conditions = set()
    filter_conditions = set()
    num_joins_list = []
    num_filters_list = []

    # 匹配连接条件 (e.g., a.a1 = b.b1)
    join_pattern = re.compile(r'(\w+\.\w+)\s*=\s*(\w+\.\w+)')
    
    # 匹配 WHERE 条件中的筛选条件 (e.g., a.a1 >= 10)
    filter_pattern = re.compile(r'(\w+\.\w+)\s*(=|!=|>|<|>=|<=|LIKE|IN|BETWEEN|IS)\s*(?!\w+\.\w+)[^\s]+')

    for i, sql in enumerate(sql_list):
        # 将 SQL 转成小写
        sql = sql.lower()
        
        # 提取连接条件 (通常出现在 ON 或 WHERE 中)
        join_matches = join_pattern.findall(sql)
        num_joins = len(join_matches)
        num_joins_list.append(num_joins)
        for match in join_matches:
            # 将等式左右部分排序，确保 a.a1=b.b1 和 b.b1=a.a1 被视为相同
            left, right = sorted([match[0], match[1]])
            join_conditions.add(f"{left}={right}")
        
        # 提取 WHERE 子句中的筛选条件

        filter_matches = filter_pattern.findall(sql)
        unique_filters = set([match[0] for match in filter_matches])
        num_filters = len(unique_filters)
        num_filters_list.append(num_filters)
        for match in filter_matches:
            lefthand = match[0]
            filter_conditions.add(lefthand)

    num_joins_set = set(num_joins_list)

    # 使用 Counter 统计每个 num_join 出现的次数
    num_joins_counter = Counter(num_joins_list)

    # 将 num_joins_set 转换为排序列表（可选，根据需求）
    unique_num_joins = sorted(num_joins_set)

    # 生成对应的 sql 数量列表
    num_joins_counts = [num_joins_counter[num] for num in unique_num_joins]

    num_joins_distribute = (unique_num_joins, num_joins_counts)

    num_filters_set = set(num_filters_list)

    # 使用 Counter 统计每个 num_join 出现的次数
    num_filters_counter = Counter(num_filters_list)

    # 将 num_joins_set 转换为排序列表（可选，根据需求）
    unique_num_filters = sorted(num_filters_set)

    # 生成对应的 sql 数量列表
    num_filters_counts = [num_filters_counter[num] for num in unique_num_filters]

    num_filters_distribute = (unique_num_filters, num_filters_counts)


    return join_conditions, filter_conditions, num_joins_distribute, num_filters_distribute


def build_join_graph(join_conditions):
    """
    构建表的连接图。
    
    参数:
    join_conditions (set): 包含连接条件的集合，如 {"a.a1=b.b1", "c.c1=d.d1"}。
    
    返回:
    dict: 以表为节点的邻接表，描述表之间的连接关系。
    """
    graph = defaultdict(list)
    
    for condition in join_conditions:
        # 解析连接条件
        left_table = condition.split('.')[0]
        right_table = condition.split('=')[1].split('.')[0]
        
        # 构建连接图
        graph[left_table].append((right_table, condition))
        graph[right_table].append((left_table, condition))
    
    return graph

def generate_connected_joins(graph, num_joins):
    """
    从连接图中生成连通的 JOIN 条件。
    
    参数:
    graph (dict): 表的连接图，描述表之间的连接关系。
    num_joins (int): 希望生成的 JOIN 条件数量。
    
    返回:
    list: 生成的 JOIN 条件。
    set: 涉及到的表。
    """
    tables = set()
    joins = []
    
    # 随机选择一个起始表
    current_table = random.choice(list(graph.keys()))
    tables.add(current_table)
    
    for _ in range(num_joins):
        if not graph[current_table]:
            break
        
        # 从当前表的邻接表中随机选择一个连接表
        next_table, join_condition = random.choice(graph[current_table])
        
        if next_table not in tables:
            # 如果 next_table 还没有加入过，添加连接条件
            joins.append(join_condition)
            tables.add(next_table)
        
        # 移动到下一个表
        current_table = next_table
    
    return joins, tables


def generate_random_sql(join_conditions, filter_conditions, num_joins_distribute, 
                        num_filters_distribute, rev_alias_map, M, num_sql=20000):
    """
    根据 join_conditions, filter_conditions 和字典 M 生成随机 SQL 查询。
    
    参数:
    join_conditions (set): 包含连接条件的集合，如 {"a.a1=b.b1", "c.c1=d.d1"}。
    filter_conditions (set): 包含过滤条件中列的集合，如 {"a.a1", "a.a2", "c.c2"}。
    num_joins_distribute (tuple): JOIN 数量的分布和权重。
    num_filters_distribute (tuple): WHERE 条件数量的分布和权重。
    rev_alias_map (dict): 表别名的映射，如 {"a": "table_a"}。
    M (dict): 字典，记录每张表每列的最大最小值，如 M["a"]["a1"] = (1, 100)。
    num_sql (int): 生成的 SQL 查询数量，默认为 20000。
    
    返回:
    list: 包含生成的 SQL 查询的列表。
    """
    sql_queries = []
    
    # 构建连接图
    join_graph = build_join_graph(join_conditions)
    
    for _ in range(num_sql):
        # 生成 SELECT 和 FROM 子句
        sql_query = "SELECT COUNT(*) FROM "

        # 随机选择 JOIN 数量
        num_joins = random.choices(num_joins_distribute[0], weights=num_joins_distribute[1], k=1)[0]
        
        # 使用连通性保证 JOIN 条件的生成
        generated_joins, tables = generate_connected_joins(join_graph, num_joins)
        
        # 构建 FROM 子句
        sql_query += ', '.join(f'{rev_alias_map[table]} as {table}' for table in tables) + ' '

        # 构建 JOIN 子句
        sql_query += 'WHERE ' + ' AND '.join(f'{join}' for join in generated_joins)

        # 生成 filter 子句
        where_conditions = []
        num_filters = random.choices(num_filters_distribute[0], weights=num_filters_distribute[1], k=1)[0]

        # 从 filter_conditions 中随机选择 num_filters 个不重复的过滤条件
        available_filters = list(filter_conditions)
        random.shuffle(available_filters)  # 随机打乱过滤条件列表

        for filter_condition in available_filters:
            if len(where_conditions) >= num_filters:
                break  # 已达到所需的过滤条件数量

            # 从 filter_condition 中提取表和列
            table, column = filter_condition.split('.')
            
            if table not in tables:
                continue  # 确保过滤条件应用在已连接的表上
            
            # 从 M 字典中获取该列的最小值和最大值
            min_val, max_val = M[table][column]
            
            # 生成随机过滤条件
            operator = random.choice(["=", "!=", ">", "<", ">=", "<="])
            value = random.randint(min_val, max_val)
            
            where_conditions.append(f"{filter_condition} {operator} {value}")
        
        # 将 WHERE 条件加入 SQL
        if where_conditions:
            sql_query += " AND " + " AND ".join(where_conditions) + ";"
        else:
            sql_query += ";"
        
        # 添加生成的 SQL 查询到列表中
        sql_queries.append(sql_query)
    
    return sql_queries

if __name__ == '__main__':
    sql_list = read_query('data/test/stats_test_sql.txt')
    join_conditions, filter_conditions, num_joins_distribute, num_filters_distribute = extract_conditions(sql_list)
    with open('infos/rev_alias_map', 'r') as f:
        rev_alias_map = json.load(f)
    with open('infos/range_dict', 'r') as f:
        M = json.load(f)
    sql_queries = generate_random_sql(join_conditions,
                        filter_conditions, 
                        num_joins_distribute, 
                        num_filters_distribute, 
                        rev_alias_map, 
                        M, 
                        num_sql=20000)
    lines = [f"{i}#####{sql}\n" for i, sql in enumerate(sql_queries)]
    with open('data/unlabeled_train_data/stats_train_pool.txt', 'w') as f:
        f.writelines(lines)
    pass
    
