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
    tuple: 包含两个集合，第一个集合是所有连接条件，第二个集合是所有筛选条件中的 lefthand 表达式。
    """
    join_conditions = set()
    filter_conditions = set()
    num_joins_list = []
    num_filters_list = []

    # 匹配连接条件 (e.g., a.a1 = b.b1)
    join_pattern = re.compile(r'(\w+\.\w+)\s*=\s*(\w+\.\w+)')

    # 匹配 WHERE 条件中的筛选条件
    filter_pattern = re.compile(
        r'(\w+\.\w+)\s*'                             # Left-hand side (table.column)
        r'(=|!=|>|<|>=|<=|LIKE|IN|BETWEEN|IS)\s*'    # Operator
        r'('
            r"'[^']*'"                                # Single-quoted string
            r'|"[^"]*"'                               # Double-quoted string
            r'|\([^\)]*\)'                            # Parentheses (e.g., subqueries or lists)
            r'|[^;\s]+'                               # Other values (numbers, identifiers)
        r')'
    )

    for sql in sql_list:
        # 将 SQL 转成小写以匹配连接条件
        sql_lower = sql.lower()
        
        # 提取连接条件 (通常出现在 ON 或 WHERE 中)
        join_matches = join_pattern.findall(sql_lower)
        num_joins = len(join_matches)
        num_joins_list.append(num_joins)
        for match in join_matches:
            # 将等式左右部分排序，确保 a.a1=b.b1 和 b.b1=a.a1 被视为相同
            left, right = sorted([match[0], match[1]])
            join_conditions.add(f"{left}={right}")

        # 提取 WHERE 子句中的筛选条件（使用原始 SQL）
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

    # 将 num_joins_set 转换为排序列表
    unique_num_joins = sorted(num_joins_set)

    # 生成对应的 sql 数量列表
    num_joins_counts = [num_joins_counter[num] for num in unique_num_joins]

    num_joins_distribute = (unique_num_joins, num_joins_counts)

    num_filters_set = set(num_filters_list)

    # 使用 Counter 统计每个 num_filter 出现的次数
    num_filters_counter = Counter(num_filters_list)

    # 将 num_filters_set 转换为排序列表
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
        generated_joins, tables = generate_connected_joins(join_graph, num_joins)
        
        # 构建 SELECT 和 FROM 子句
        select_clause = "SELECT COUNT(*) FROM "
        # 构建 FROM 子句
        from_items = [
            f"{rev_alias_map[table]} AS {table}"
            for table in tables
        ]
        from_clause = ', '.join(from_items)
        
        # 构建 WHERE 子句
        where_conditions = []
        # 解析 JOIN 条件
        for join_condition in generated_joins:
            # join_condition 是类似 'a.a1=b.b1' 的字符串
            left_expr, right_expr = join_condition.split('=')
            left_table, left_column = left_expr.split('.')
            right_table, right_column = right_expr.split('.')
            condition = f"{left_table}.{left_column} = {right_table}.{right_column}"
            where_conditions.append(condition)
        
        # 生成过滤条件
        num_filters = int(random.choices(num_filters_distribute[0], weights=num_filters_distribute[1], k=1)[0]*0.5) + 1

        # 从 filter_conditions 中随机选择 num_filters 个不重复的过滤条件
        available_filters = list(filter_conditions)
        random.shuffle(available_filters)  # 随机打乱过滤条件列表

        filter_count = 0
        for filter_condition in available_filters:
            if filter_count >= num_filters:
                break  # 已达到所需的过滤条件数量

            # 从 filter_condition 中提取表和列
            table, column = filter_condition.split('.')
            
            if table not in tables:
                continue  # 确保过滤条件应用在已连接的表上
            
            # 从 M 字典中获取该列的取值范围或可能的取值列表
            m_value = M[table.lower()][column.lower()]

            if isinstance(m_value, list):
                if len(m_value) == 2 and all(isinstance(v, (int, float)) for v in m_value):
                    # 数值型列
                    min_val, max_val = m_value
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val  # 确保 min_val <= max_val
                    operator = random.choice(["=", "!=", ">", "<", ">=", "<="])
                    value = random.randint(int(min_val), int(max_val))
                    condition = f"{table}.{column} {operator} {value}"
                    where_conditions.append(condition)
                    filter_count += 1
                elif all(isinstance(v, str) for v in m_value):
                    # 字符串型列
                    operator = random.choice(["=", "!="])
                    value = random.choice(m_value).replace("'", "''")  # 防止单引号引起的错误
                    condition = f"{table}.{column} {operator} '{value}'"
                    where_conditions.append(condition)
                    filter_count += 1
                else:
                    print(f'data type {type(m_value[0])} is not defined')
                    continue
            else:
                print(f'M[{table}][{column}] is not a list')
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
    
