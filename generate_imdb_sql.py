import sqlglot
from sqlglot import parse_one
import random
import psycopg2
from psycopg2 import OperationalError, sql
from collections import defaultdict
import json

def extract_tables_and_aliases(sql_query):
    """
    从单个SQL查询中提取表和别名的映射。

    返回两个字典：
    1. table_to_aliases: {table: [alias1, alias2, ...]}
    2. alias_to_table: {alias: table}
    """
    table_to_aliases = defaultdict(list)
    alias_to_table = {}

    parsed = sqlglot.parse(sql_query)
    if not parsed:
        return table_to_aliases, alias_to_table

    stmt = parsed[0]
    from_seen = False
    for token in stmt.tokens:
        if from_seen:
            if token.ttype is sqlglot.tokens.TokenType.KEYWORD:
                break
            if isinstance(token, sqlglot.expressions.IdentifierList):
                for identifier in token.get_identifiers():
                    table = identifier.this.name
                    alias = identifier.alias_or_name or table
                    table_to_aliases[table].append(alias)
                    alias_to_table[alias] = table
            elif isinstance(token, sqlglot.expressions.Identifier):
                table = token.this.name
                alias = token.alias_or_name or table
                table_to_aliases[table].append(alias)
                alias_to_table[alias] = table
            elif isinstance(token, sqlglot.expressions.Join):
                # 处理JOIN子句中的表
                join_table = token.this
                if isinstance(join_table, sqlglot.expressions.Identifier):
                    table = join_table.this.name
                    alias = join_table.alias_or_name or table
                    table_to_aliases[table].append(alias)
                    alias_to_table[alias] = table
        if token.ttype is sqlglot.tokens.TokenType.KEYWORD and token.text.upper() == 'FROM':
            from_seen = True
    return table_to_aliases, alias_to_table

def get_column_table(expr, alias_to_table):
    if isinstance(expr, sqlglot.expressions.Column):
        alias = expr.table
        if alias:
            return alias_to_table.get(alias, alias)
    return None

def separate_conditions(condition, alias_to_table):
    """
    分离JOIN条件和过滤条件。

    参数:
    - condition: sqlglot表达式的WHERE子句
    - alias_to_table: {alias: table}

    返回:
    - join_conditions: list
    - filter_conditions: list
    """
    join_conditions = []
    filter_conditions = []

    if not isinstance(condition, sqlglot.expressions.Expression):
        return join_conditions, filter_conditions

    if isinstance(condition, sqlglot.expressions.And):
        left = condition.left
        right = condition.right

        left_joins, left_filters = separate_conditions(left, alias_to_table)
        join_conditions.extend(left_joins)
        filter_conditions.extend(left_filters)

        right_joins, right_filters = separate_conditions(right, alias_to_table)
        join_conditions.extend(right_joins)
        filter_conditions.extend(right_filters)

    elif isinstance(condition, sqlglot.expressions.Or):
        # OR条件被视为过滤条件
        filter_conditions.append(condition)

    elif isinstance(condition, sqlglot.expressions.EQ):
        left = condition.left
        right = condition.right
        if is_column_from_different_tables(left, right, alias_to_table):
            join_conditions.append(condition)
        else:
            filter_conditions.append(condition)

    elif isinstance(condition, (sqlglot.expressions.Like, sqlglot.expressions.In, 
                                sqlglot.expressions.GT, sqlglot.expressions.LT,
                                sqlglot.expressions.GTE, sqlglot.expressions.LTE)):
        filter_conditions.append(condition)

    else:
        # 其他类型的条件被视为过滤条件
        filter_conditions.append(condition)

    return join_conditions, filter_conditions

def is_column_from_different_tables(left, right, alias_to_table):
    left_table = get_column_table(left, alias_to_table)
    right_table = get_column_table(right, alias_to_table)
    if left_table and right_table and left_table != right_table:
        return True
    return False

def parse_template_sql(template_sql):
    parsed = parse_one(template_sql)
    
    # 获取 SELECT 子句
    select_expr = parsed.select().expressions

    components = {
        'select': select_expr,
        'from_tables': [],
        'join_conditions': [],
        'filter_conditions': [],
        'columns': set(),
        'alias_map': {}
    }

    # 提取所有的表及其别名
    all_tables = []
    for table in parsed.find_all(sqlglot.expressions.Table):
        table_name = table.name
        alias = table.alias_or_name or table_name
        all_tables.append((table_name, alias))
    components['from_tables'] = all_tables


    # 提取表别名映射
    alias_map = {alias: name for name, alias in all_tables}
    components['alias_map'] = alias_map

    # 提取所有的连接条件和过滤条件
    where_clause = parsed.args.get('where')
    if where_clause:
        join_conditions, filter_conditions = separate_conditions(where_clause.this, alias_map)
        components['join_conditions'] = join_conditions
        components['filter_conditions'] = filter_conditions

    # 提取模板中涉及的列
    columns = set()
    for column in parsed.find_all(sqlglot.expressions.Column):
        if column.table:
            columns.add(f"{column.table}.{column.name}")
        else:
            columns.add(column.name)
    components['columns'] = columns

    return components


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

def generate_new_filters(components, alias_to_table, range_dict, filter_columns, num_conditions=2):
    """
    根据 filter_columns 和 range_dict 生成新的过滤条件。

    参数:
    - components (dict): 包含从 SQL 模板解析出的表、别名、列和其他信息。
    - alias_to_table (dict): {alias: table}
    - range_dict (dict): {alias: {column: range}}
    - filter_columns (dict): {table: [columns]}
    - num_conditions (int): 需要生成的过滤条件数量

    返回:
    - filters (list): 生成的过滤条件字符串
    """
    filters = []
    used_columns = set()

    # 1. 处理具有多个别名的表，判断表在 components['from_tables'] 中是否有多个别名
    table_aliases = {}  # 记录每个表的别名出现次数
    for table_name, alias in components['from_tables']:
        if table_name not in table_aliases:
            table_aliases[table_name] = []
        table_aliases[table_name].append(alias)

    tables_with_multiple_aliases = {table for table, aliases in table_aliases.items() if len(aliases) > 1}

    for table in tables_with_multiple_aliases:
        if len(filters) >= num_conditions:
            break

        aliases = table_aliases[table]
        shared_columns = list(set(filter_columns.get(table, [])))
        if not shared_columns:
            continue
        shared_column = random.choice(shared_columns)

        for alias in aliases:
            if len(filters) >= num_conditions:
                break
            column = f"{alias}.{shared_column}"
            if column in used_columns:
                continue
            col_info = range_dict.get(alias, {}).get(shared_column)
            if col_info is None:
                continue

            # 根据列类型生成条件
            if isinstance(col_info, (list, tuple)) and len(col_info) == 2 and all(isinstance(x, (int, float)) for x in col_info):
                # 数值型列
                min_value, max_value = col_info
                operator = random.choice(['=', '!=', '>', '<', '>=', '<='])
                if isinstance(min_value, int) and isinstance(max_value, int) and min_value != max_value:
                    value = random.randint(int(min_value), int(max_value))
                else:
                    value = random.uniform(min_value, max_value)
                condition = f"{column} {operator} {value}"
            elif isinstance(col_info, list):
                # 字符串型列
                operator = random.choices(['=', '!=', 'LIKE', 'IN'], weights=[1, 13, 13, 13])[0]
                if operator == 'IN':
                    if len(col_info) <= 2:
                        operator = '='
                        value = escape_single_quotes(random.choice(col_info))
                        condition = f"{column} {operator} '{value}'"
                    else:
                        num_values = random.randint(2, min(7, len(col_info)-1))
                        selected_values = random.sample(col_info, num_values)
                        escaped_values = [f"'{escape_single_quotes(v)}'" for v in selected_values]
                        values_str = ', '.join(escaped_values)
                        condition = f"{column} IN ({values_str})"
                elif operator == 'LIKE':
                    value = random.choice(col_info)
                    sliced_value = value[:2]  # 取前两个字符
                    escaped_sliced = escape_single_quotes(sliced_value)
                    pattern = f"'%{escaped_sliced}%'"
                    condition = f"{column} {operator} {pattern}"
                else:
                    value = escape_single_quotes(random.choice(col_info))
                    condition = f"{column} {operator} '{value}'"
            else:
                continue

            filters.append(condition)
            used_columns.add(column)

    # 2. 生成剩余的随机过滤条件
    remaining_conditions = num_conditions - len(filters)
    if remaining_conditions > 0:
        # 获取未使用的列
        available_columns = list(components['columns'] - used_columns)
        random.shuffle(available_columns)

        for column in available_columns:
            if len(filters) >= num_conditions:
                break
            table_alias, col_name = column.split('.', 1)
            col_info = range_dict.get(table_alias, {}).get(col_name)
            table = alias_to_table.get(table_alias)

            if col_info is None or col_name not in filter_columns.get(table, []):
                continue

            # 根据列类型生成条件
            if isinstance(col_info, (list, tuple)) and len(col_info) == 2 and all(isinstance(x, (int, float)) for x in col_info):
                # 数值型列
                min_value, max_value = col_info
                operator = random.choice(['=', '!=', '>', '<', '>=', '<='])
                if isinstance(min_value, int) and isinstance(max_value, int) and min_value != max_value:
                    value = random.randint(int(min_value), int(max_value))
                else:
                    value = random.uniform(min_value, max_value)
                condition = f"{column} {operator} {value}"
            elif isinstance(col_info, list):
                # 字符串型列
                operator = random.choices(['=', '!=', 'LIKE', 'IN'], weights=[1, 13, 13, 13])[0]
                if operator == 'IN':
                    num_values = random.randint(2, min(7, len(col_info)))
                    selected_values = random.sample(col_info, num_values)
                    escaped_values = [f"'{escape_single_quotes(v)}'" for v in selected_values]
                    values_str = ', '.join(escaped_values)
                    condition = f"{column} IN ({values_str})"
                elif operator == 'LIKE':
                    value = random.choice(col_info)
                    sliced_value = value[:2]  # 取前两个字符
                    escaped_sliced = escape_single_quotes(sliced_value)
                    pattern = f"'%{escaped_sliced}%'"
                    condition = f"{column} {operator} {pattern}"
                else:
                    value = escape_single_quotes(random.choice(col_info))
                    condition = f"{column} {operator} '{value}'"
            else:
                continue

            filters.append(condition)
            used_columns.add(column)

    return filters

from sqlglot import expressions

def generate_sql_from_template_info(template_info, range_dict, filter_columns, table_to_aliases, alias_to_table):
    """
    根据模板信息和 range_dict 生成新的 SQL 查询。

    参数:
    - template_info (dict): 解析后的模板信息。
    - range_dict (dict): {alias: {column: range, ...}, ...}。
    - filter_columns (dict): {table: [columns]}。
    - table_to_aliases (dict): {table: [aliases]}。
    - alias_to_table (dict): {alias: table}。
    - num_conditions (int): 新生成的过滤条件数量。

    返回:
    - new_sql (str): 生成的新 SQL 查询字符串。
    """
    num_conditions = random.choices(list(range(2, 7)), weights=[1,3,3,2,1], k=1)[0]
    components = template_info

    # 生成新的过滤条件
    new_filter_conditions = generate_new_filters(
        components,
        alias_to_table,
        range_dict,
        filter_columns,
        num_conditions
    )

    # 收集所有条件的字符串形式
    condition_strings = []

    # 添加连接条件的字符串形式
    for cond in components['join_conditions']:
        condition_strings.append(cond.sql())

    # 添加新过滤条件的字符串形式
    for cond in new_filter_conditions:
        condition_strings.append(cond)

    # 构建平坦的 WHERE 子句字符串
    where_clause_str = " AND ".join(condition_strings)

    # 解析 WHERE 子句
    if where_clause_str:
        try:
            where_clause_expr = sqlglot.parse_one(where_clause_str, read='postgres')
            where_clause = expressions.Where(this=where_clause_expr)
        except Exception as e:
            print(f"Failed to parse WHERE clause '{where_clause_str}': {e}")
            where_clause = None
    else:
        where_clause = None

    # 构建 SELECT 子句
    select_clause = expressions.Select(expressions=components['select'])

    # 手动构建 FROM 子句字符串
    from_clause_str = ", ".join(
        [f"{table_name} AS {table_alias}" if table_alias != table_name else table_name
         for table_name, table_alias in components['from_tables']]
    )



    # 构建最终的 SQL 查询字符串
    query_sql = f"{select_clause.sql()} FROM {from_clause_str}"
    if where_clause_str:
        query_sql += f" WHERE {where_clause_str}"

    return query_sql

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
# 示例使用
if __name__ == "__main__":
    import json

    database = 'imdb'
    db_name = database
    db_user = "postgres"
    db_password = "li6545991360"
    db_host = "127.0.0.1"  # 本地主机，对于远程数据库，请使用IP地址或域名
    db_port = "5432"  # PostgreSQL默认端口是5432

    connection = create_connection(db_name, db_user, db_password, db_host, db_port)

    with open(f'data/test/{database}_test_sql.txt', 'r') as f:
        lines = f.readlines()
    sql_queries = [line.split('#####')[1].strip() for line in lines]


    with open(f'infos/{database}/alias_map', 'r') as f:
        table_to_aliases = json.load(f)
    with open(f'infos/{database}/rev_alias_map', 'r') as f:
        alias_to_table =  json.load(f)


    with open(f'infos/{database}/filter_columns', 'r') as f:    
        filter_columns = json.load(f)

    with open(f'infos/{database}/range_dict', 'r') as f:    
            range_dict = json.load(f)

    # 解析模板SQL
    template_infos = []
    for template_sql in sql_queries:
        template_info = parse_template_sql(template_sql)
        template_infos.append(template_info)

    new_sqls = []
    # 生成新的SQL
    for _ in range(20000):
        template_info = random.choice(template_infos)
        new_sql = generate_sql_from_template_info(
            template_info,
            range_dict,
            filter_columns,
            table_to_aliases,
            alias_to_table,
        )
        new_sqls.append(new_sql)
        # 关闭数据库连接
    connection.close()
    with open(f'data/unlabeled_train_data/{database}_train_pool.txt', 'w') as f:
        lines = [f'{i}#####{sql}\n' for i, sql in enumerate(new_sqls)]
        f.writelines(lines)
