import re, os, random, json, psycopg2
from collections import Counter, defaultdict
from psycopg2 import OperationalError, sql

def read_query(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        qs = [line.split('#####')[1] for line in lines]
    return qs

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
def extract_conditions(sql_list):
    join_conditions = set()
    filter_conditions = set()
    num_filters_list = []
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
        # 提取连接条件
        sql_lower = sql.lower()

        # 提取连接条件 (通常出现在 ON 或 WHERE 中)
        join_matches = join_pattern.findall(sql_lower)
        sorted_join_matches = []
        for match in join_matches:
            # 将等式左右部分排序，确保一致性
            left, right = sorted([match[0], match[1]])
            sorted_join_matches.append(f"{left}={right}")

        sorted_join_matches = sorted(sorted_join_matches)
        combined_conditions = ' AND '.join(sorted_join_matches)
        if combined_conditions.strip() == 'b.userid=v.userid AND c.userid=p.owneruserid AND c.userid=u.id AND p.owneruserid=v.userid':
            print(sql)
        join_conditions.add(combined_conditions.strip())

        # 提取 WHERE 子句中的筛选条件
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

    num_filters_counter = Counter(num_filters_list)
    unique_num_filters = sorted(num_filters_counter)
    num_filters_counts = [num_filters_counter[num] for num in unique_num_filters]
    num_filters_distribute = (unique_num_filters, num_filters_counts)

    return join_conditions, filter_conditions, num_filters_distribute

def generate_random_sql(join_conditions, filter_conditions, num_filters_distribute, rev_alias_map, M,  num_sql=40000):
    
    sql_queries = []
    
    # 将连接条件转换为列表以便随机选择
    join_conditions = list(join_conditions)
    
    for _ in range(num_sql):
        # 随机选择 JOIN 条件
        selected_join_condition = random.choice(join_conditions)
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
        all_columns = set()
        involved_tables = set()
        for join_condition in selected_join_condition.split(' AND '):
            left_expr, right_expr = join_condition.split('=')
            left_expr = left_expr.strip()
            right_expr = right_expr.strip()
            all_columns.add(left_expr)
            all_columns.add(right_expr)
            involved_tables.add(left_expr.split('.')[0])
            involved_tables.add(right_expr.split('.')[0])
            union(left_expr, right_expr)  # 合并等价类
        # 构建 SELECT 和 FROM 子句
        select_clause = "SELECT COUNT(*) FROM "
        # 从连接条件中提取涉及的表

        from_items = [
            f"{rev_alias_map[table]} AS {table}"
            for table in involved_tables
        ]
        from_clause = ', '.join(from_items)

        # 初始化 WHERE 条件列表
        join_clause = selected_join_condition

        num_filters = int(random.choices(num_filters_distribute[0], weights=num_filters_distribute[1])[0])
        
        available_filters = list(filter_conditions)
        random.shuffle(available_filters)

        filter_count = 0

        used_equivalence_classes = set()  # 跟踪已经添加过滤条件的等价类
        select_filter_conditions = []
        for filter_condition in available_filters:
            if filter_count >= num_filters:
                break  # 已达到所需的过滤条件数量

            # 从 filter_condition 中提取表和列
            try:
                table, column = filter_condition.split('.')
            except ValueError:
                print(f"Invalid filter_condition format: {filter_condition}")
                continue  # 跳过格式错误的过滤条件

            if table not in involved_tables:
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
                    select_filter_conditions.append(condition)
                    filter_count += 1
                    used_equivalence_classes.add(col_parent)
                elif all(isinstance(v, str) for v in m_value):
                    # 字符串型列
                    operator = random.choice(["=", "!="])
                    value = escape_single_quotes(random.choice(m_value))
                    condition = f"{table}.{column} {operator} '{value}'"
                    select_filter_conditions.append(condition)
                    filter_count += 1
                    used_equivalence_classes.add(col_parent)
                else:
                    print(f"Unsupported data type in M[{table}][{column}]")
                    continue
            else:
                print(f"M[{table}][{column}] is not a list")
                continue

        filter_clause = ' AND '.join(select_filter_conditions)
        # 构建完整的 SQL 查询
        sql_query = select_clause + from_clause

        if join_clause or filter_clause:
            sql_query += ' Where '
            if join_conditions:
                sql_query += join_clause
                if filter_clause:
                    sql_query += ' AND ' + filter_clause
                sql_query += ';'
            else:
                sql_query += filter_clause + ';'
        else:
            sql_query += ';'

        sql_queries.append(sql_query)
        # print(sql_query)
    return sql_queries

if __name__ == '__main__':
    database = 'stats'
    sql_list = read_query(f'data/test/{database}_test_sql.txt')
    join_conditions, filter_conditions, num_filters_distribute = extract_conditions(sql_list)
    with open(f'infos/{database}/rev_alias_map', 'r') as f:
        rev_alias_map = json.load(f)
    with open(f'infos/{database}/range_dict', 'r') as f:
        M = json.load(f)


    sql_queries = generate_random_sql(join_conditions,
                        filter_conditions, 
                        num_filters_distribute, 
                        rev_alias_map, 
                        M, 
                        num_sql=40000)
    lines = [f"{i}#####{sql}\n" for i, sql in enumerate(sql_queries)]
    with open(f'data/unlabeled_train_data/{database}_train_pool.txt', 'w') as f:
        f.writelines(lines)
    pass