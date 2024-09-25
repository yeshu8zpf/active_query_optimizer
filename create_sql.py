import re, os, random, json, psycopg2
from collections import Counter, defaultdict
from psycopg2 import OperationalError, sql

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
    tuple: 包含四个元素：
        - join_conditions (set): 所有连接条件。
        - filter_conditions (set): 所有筛选条件中的左侧表达式。
        - num_joins_distribute (tuple): 连接条件数量的分布。
        - num_filters_distribute (tuple): 筛选条件数量的分布。
    """
    join_conditions = set()
    filter_conditions = set()
    num_joins_list = []
    num_filters_list = []

    # 匹配连接条件 (e.g., a.a1 = b.b1)
    join_pattern = re.compile(r'(\w+\.\w+)\s*=\s*(\w+\.\w+)')

    # 匹配 WHERE 条件中的筛选条件
    filter_pattern = re.compile(
        r'(\w+\.\w+)\s*'                             # 左侧 (table.column)
        r'(=|!=|>|<|>=|<=|LIKE|IN|BETWEEN|IS)\s*'    # 操作符
        r'('
            r"'[^']*'"                                # 单引号字符串
            r'|"[^"]*"'                               # 双引号字符串
            r'|\([^\)]*\)'                            # 括号 (如子查询或列表)
            r'|[^;\s]+'                               # 其他值 (数字、标识符)
        r')'
    )

    # 用于识别 table.column 格式的列名
    column_pattern = re.compile(r'^\w+\.\w+$')

    for sql in sql_list[::-1]:
        # 提取连接条件 (转换为小写以匹配连接条件)
        sql_lower = sql.lower()

        # 提取连接条件 (通常出现在 ON 或 WHERE 中)
        join_matches = join_pattern.findall(sql_lower)
        num_joins = len(join_matches)
        if num_joins > 4:
            pass
        num_joins_list.append(num_joins)
        for match in join_matches:
            # 将等式左右部分排序，确保一致性
            left, right = sorted([match[0], match[1]])
            join_conditions.add(f"{left}={right}")

        # 提取 WHERE 子句中的筛选条件（使用原始 SQL 保持大小写）
        filter_matches = filter_pattern.findall(sql)
        unique_filters = set()
        for match in filter_matches:
            lefthand = match[0]
            righthand = match[2]

            # 检查右侧是否不是列 (即排除连接条件)
            if column_pattern.match(righthand.strip()):
                continue  # 如果右侧是列，则跳过（属于连接条件）

            unique_filters.add(lefthand)

        num_filters = len(unique_filters)
        num_filters_list.append(num_filters)
        filter_conditions.update(unique_filters)

    # 计算分布
    num_joins_counter = Counter(num_joins_list)
    unique_num_joins = sorted(num_joins_counter)
    num_joins_counts = [num_joins_counter[num] for num in unique_num_joins]
    num_joins_distribute = (unique_num_joins, num_joins_counts)

    num_filters_counter = Counter(num_filters_list)
    unique_num_filters = sorted(num_filters_counter)
    num_filters_counts = [num_filters_counter[num] for num in unique_num_filters]
    num_filters_distribute = (unique_num_filters, num_filters_counts)

    return join_conditions, filter_conditions, num_joins_distribute, num_filters_distribute


def generate_random_sql(join_conditions, filter_conditions, num_joins_distribute, 
                       num_filters_distribute, rev_alias_map, M, connection, num_sql=20000):
    """
    根据 join_conditions, filter_conditions 和字典 M 生成随机 SQL 查询。
    
    参数:
    join_conditions (set): 包含连接条件的集合，如 {"a.a1=b.b1", "c.c1=d.d1"}。
    filter_conditions (set): 包含过滤条件中列的集合，如 {"a.a1", "a.a2", "c.c2"}。
    num_joins_distribute (tuple): JOIN 数量的分布和权重。
    num_filters_distribute (tuple): WHERE 条件数量的分布和权重。
    rev_alias_map (dict): 表别名的映射，如 {"a": "table_a"}。
    M (dict): 字典，记录每张表每列的最大最小值或可能的取值列表。
    connection: psycopg2 的数据库连接，用于生成查询字符串。
    num_sql (int): 生成的 SQL 查询数量，默认为 20000。
    
    返回:
    list: 包含生成的 SQL 查询的列表。
    """
    sql_queries = []
    
    # 构建连接图
    join_graph = build_join_graph(join_conditions)
    
    for _ in range(num_sql):
        # 随机选择 JOIN 数量
        num_joins = random.choices(num_joins_distribute[0], weights=num_joins_distribute[1], k=1)[0]
        
        # 使用连通性保证 JOIN 条件的生成
        generated_joins, tables = generate_connected_joins(join_graph, num_joins, join_conditions)
        
        # 构建 SELECT 和 FROM 子句
        select_clause = "SELECT COUNT(*) FROM "
        # 构建 FROM 子句
        from_items = [
            f"{rev_alias_map[table]} AS {table}"
            for table in tables
        ]
        from_clause = ', '.join(from_items)
        
        # 初始化 WHERE 条件列表
        where_conditions = []
        
        # 1. 构建当前查询的列等价类（并查集）
        parent = {}

        def find(x):
            # 并查集查找
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            # 并查集合并
            parent_x = find(x)
            parent_y = find(y)
            if parent_x != parent_y:
                parent[parent_y] = parent_x

        # 初始化所有涉及的列
        all_columns = set()
        for join_condition in generated_joins:
            left_expr, right_expr = join_condition.split('=')
            left_expr = left_expr.strip()
            right_expr = right_expr.strip()
            all_columns.add(left_expr)
            all_columns.add(right_expr)
            union(left_expr, right_expr)  # 合并等价类

            # 将连接条件添加到 WHERE 子句
            condition = f"{left_expr} = {right_expr}"
            where_conditions.append(condition)
        
        # 2. 生成过滤条件
        num_filters = int(random.choices(num_filters_distribute[0], weights=num_filters_distribute[1], k=1)[0]*0.5)+1
    
        # 从 filter_conditions 中随机选择过滤条件
        available_filters = list(filter_conditions)
        random.shuffle(available_filters)  # 随机打乱过滤条件列表

        filter_count = 0
        used_equivalence_classes = set()  # 跟踪已经添加过滤条件的等价类

        for filter_condition in available_filters:
            if filter_count >= num_filters:
                break  # 已达到所需的过滤条件数量

            # 从 filter_condition 中提取表和列
            try:
                table, column = filter_condition.split('.')
            except ValueError:
                print(f"Invalid filter_condition format: {filter_condition}")
                continue  # 跳过格式错误的过滤条件

            if table not in tables:
                continue  # 确保过滤条件应用在已连接的表上

            full_column = f"{table}.{column}"

            # 查找该列所属的等价类
            col_parent = find(full_column)

            if col_parent in used_equivalence_classes:
                continue  # 已经对该等价类施加了过滤条件

            # 从 M 字典中获取该列的取值范围或可能的取值列表
            m_value = M.get(table.lower(), {}).get(column.lower())

            if m_value is None:
                print(f"M[{table}][{column}] is not defined")
                continue

            if isinstance(m_value, list):
                if len(m_value) == 2 and all(isinstance(v, (int, float)) for v in m_value):
                    # 数值型列
                    min_val, max_val = m_value
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val  # 确保 min_val <= max_val
                    operator = random.choices(["=", "!=", ">", "<", ">=", "<="], weights=[1,7,7,7,7,7])[0]

                    # 生成数值型条件
                    if isinstance(min_val, int) and isinstance(max_val, int) and min_val != max_val:
                        value = random.randint(int(min_val), int(max_val))
                    else:
                        value = round(random.uniform(min_val, max_val), 2)  # 保留两位小数

                    condition = f"{table}.{column} {operator} {value}"
                    where_conditions.append(condition)
                    filter_count += 1
                    used_equivalence_classes.add(col_parent)
                elif all(isinstance(v, str) for v in m_value):
                    # 字符串型列
                    operator = random.choice(["=", "!="])
                    value = escape_single_quotes(random.choice(m_value))
                    condition = f"{table}.{column} {operator} '{value}'"
                    where_conditions.append(condition)
                    filter_count += 1
                    used_equivalence_classes.add(col_parent)
                else:
                    print(f"Unsupported data type in M[{table}][{column}]")
                    continue
            else:
                print(f"M[{table}][{column}] is not a list")
                continue

        # 构建完整的 SQL 查询
        sql_query = select_clause + from_clause

        if where_conditions:
            where_clause = ' WHERE ' + ' AND '.join(where_conditions)
            sql_query += where_clause + ';'  # 在末尾添加分号
        else:
            sql_query += ';'  # 如果没有 WHERE 条件，直接加分号

        sql_queries.append(sql_query)
    
    return sql_queries

def build_join_graph(join_conditions):
    """
    构建连接图，用于生成连通的 JOIN 条件。

    参数:
    - join_conditions (set): 包含连接条件的集合，如 {"a.a1=b.b1", "c.c1=d.d1"}。

    返回:
    - dict: 连接图的邻接表表示。
    """
    graph = {}
    for condition in join_conditions:
        left, right = condition.split('=')
        left = left.strip()
        right = right.strip()
        left_table, _ = left.split('.')
        right_table, _ = right.split('.')
        graph.setdefault(left_table, set()).add(right_table)
        graph.setdefault(right_table, set()).add(left_table)
    return graph

def generate_connected_joins(join_graph, num_joins, join_conditions):
    """
    生成连通的 JOIN 条件和涉及的表。

    参数:
    - join_graph (dict): 连接图的邻接表表示。
    - num_joins (int): 需要生成的 JOIN 条件数量。
    - join_conditions (set): 原始连接条件集合。

    返回:
    - tuple: (生成的 JOIN 条件列表, 涉及的表集合)
    """
    if not join_graph:
        return [], set()

    # 使用 BFS 遍历图，收集连接条件
    start_table = random.choice(list(join_graph.keys()))
    visited_tables = set([start_table])
    joins = []
    tables = set([start_table])

    # 初始化队列，存储可以连接的表
    queue = [(start_table, neighbor) for neighbor in join_graph[start_table]]

    while len(joins) < num_joins and queue:
        current_table, neighbor_table = queue.pop(0)
        if neighbor_table in visited_tables:
            continue

        # 查找连接当前表和邻居表的所有可能的连接条件
        possible_conditions = [cond for cond in join_conditions 
                               if (cond.startswith(f"{current_table}.") and cond.split('=')[1].strip().startswith(f"{neighbor_table}."))
                               or (cond.startswith(f"{neighbor_table}.") and cond.split('=')[1].strip().startswith(f"{current_table}."))]
        if not possible_conditions:
            continue  # 如果没有符合的连接条件，跳过

        # 随机选择一个连接条件
        join_condition = random.choice(possible_conditions)
        joins.append(join_condition)

        # 更新已访问的表和队列
        tables.add(neighbor_table)
        visited_tables.add(neighbor_table)

        # 将新表的邻居加入队列
        for next_neighbor in join_graph[neighbor_table]:
            if next_neighbor not in visited_tables:
                queue.append((neighbor_table, next_neighbor))

    return joins, tables

def escape_single_quotes(value):
    """
    转义字符串中的单引号，将每个单引号替换为两个单引号。

    参数:
    - value (str): 需要转义的字符串。

    返回:
    - str: 转义后的字符串。
    """
    if isinstance(value, str):
        return value.replace("'", "''")
    return value

if __name__ == '__main__':
    database = 'stats'

    sql_list = read_query(f'data/test/{database}_test_sql.txt')
    join_conditions, filter_conditions, num_joins_distribute, num_filters_distribute = extract_conditions(sql_list)
    with open(f'infos/{database}/rev_alias_map', 'r') as f:
        rev_alias_map = json.load(f)
    with open(f'infos/{database}/range_dict', 'r') as f:
        M = json.load(f)

    def create_connection(db_name, db_user, db_password, db_host, db_port):
        connection = None
        try:
            connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port
            )
            print("Connection to PostgreSQL DB successful")
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        return connection

    # 使用你的数据库配置替换这些参数
    db_name = database
    db_user = "postgres"
    db_password = "li6545991360"
    db_host = "127.0.0.1"  # 本地主机，对于远程数据库，请使用IP地址或域名
    db_port = "5432"  # PostgreSQL默认端口是5432

    connection = create_connection(db_name, db_user, db_password, db_host, db_port)
    sql_queries = generate_random_sql(join_conditions,
                        filter_conditions, 
                        num_joins_distribute, 
                        num_filters_distribute, 
                        rev_alias_map, 
                        M, 
                        connection,
                        num_sql=20000)
    connection.close()
    lines = [f"{i}#####{sql}\n" for i, sql in enumerate(sql_queries)]
    with open(f'data/unlabeled_train_data/{db_name}_train_pool.txt', 'w') as f:
        f.writelines(lines)
    pass
    
