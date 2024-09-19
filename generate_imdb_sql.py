import sqlglot
from sqlglot import parse_one
import random

def extract_tables_from_from_clause(node):
    tables = []
    if isinstance(node, sqlglot.expressions.Table):
        table_name = node.name
        table_alias = node.alias_or_name
        tables.append((table_name, table_alias))
    elif isinstance(node, sqlglot.expressions.CrossJoin):
        tables.extend(extract_tables_from_from_clause(node.this))
        tables.extend(extract_tables_from_from_clause(node.expression))
    else:
        print(f"Unhandled node in FROM clause: {type(node)}")
    return tables

def parse_template_sql(template_sql):
    parsed = parse_one(template_sql)
    
    # 获取 SELECT 子句
    select_expr = parsed.select() if callable(parsed.select) else parsed.select

    components = {
        'select': select_expr.expressions,
        'from_tables': [],
        'join_conditions': [],
        'filter_conditions': [],
        'columns': set(),
    }

    # 提取所有的表
    all_tables = []
    for table in parsed.find_all(sqlglot.expressions.Table):
        table_name = table.name
        table_alias = table.alias_or_name
        all_tables.append((table_name, table_alias))
    components['from_tables'] = all_tables

    # 提取所有的连接条件和过滤条件
    where_clause = parsed.args.get('where')
    if where_clause:
        join_conditions, filter_conditions = separate_conditions(where_clause.this)
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

    # 提取表别名映射
    alias_map = {alias: name for name, alias in components['from_tables']}
    components['alias_map'] = alias_map

    return components


def separate_conditions(condition):
    join_conditions = []
    filter_conditions = []

    if not isinstance(condition, sqlglot.expressions.Expression):
        return join_conditions, filter_conditions

    if isinstance(condition, sqlglot.expressions.And):
        left = condition.this
        right = condition.expression

        left_joins, left_filters = separate_conditions(left)
        join_conditions.extend(left_joins)
        filter_conditions.extend(left_filters)

        right_joins, right_filters = separate_conditions(right)
        join_conditions.extend(right_joins)
        filter_conditions.extend(right_filters)

    elif isinstance(condition, sqlglot.expressions.Or):
        # OR conditions are considered filter conditions
        filter_conditions.append(condition)

    elif isinstance(condition, sqlglot.expressions.EQ):
        left = condition.this
        right = condition.expression
        if is_column_from_different_tables(left, right):
            join_conditions.append(condition)
        else:
            filter_conditions.append(condition)

    elif isinstance(condition, (sqlglot.expressions.Like, sqlglot.expressions.In, sqlglot.expressions.GT, sqlglot.expressions.LT,
                                sqlglot.expressions.GTE, sqlglot.expressions.LTE)):
        filter_conditions.append(condition)

    else:
        # Other types of conditions are considered filter conditions
        filter_conditions.append(condition)

    return join_conditions, filter_conditions

def is_column_from_different_tables(left, right):
    left_table = get_column_table(left)
    right_table = get_column_table(right)
    if left_table and right_table and left_table != right_table:
        return True
    return False

def get_column_table(expr):
    if isinstance(expr, sqlglot.expressions.Column):
        return expr.table
    elif isinstance(expr, sqlglot.expressions.Identifier):
        return None
    elif hasattr(expr, 'this'):
        return get_column_table(expr.this)
    else:
        return None

def generate_new_filters(columns, range_dict, num_conditions=2):
    logical_operators = ['AND', 'OR']
    filters = []

    for _ in range(num_conditions):
        # Randomly select a column
        column = random.choice(list(columns))
        table_alias, col_name = column.split('.', 1)
        col_info = range_dict.get(table_alias, {}).get(col_name)

        if col_info is None:
            continue

        # Generate condition based on column info
        if isinstance(col_info, list) and len(col_info) == 2:
            # Numeric column
            min_value, max_value = col_info
            operator = random.choice(['=', '!=', '>', '<', '>=', '<='])
            value = random.randint(int(min_value), int(max_value)) if min_value != max_value else min_value
            condition = f"{column} {operator} {value}"
        elif isinstance(col_info, list) and col_info:
            # String column
            operator = random.choice(['=', '!=', 'LIKE'])
            value = random.choice(col_info)
            if operator == 'LIKE':
                pattern = f"'%{value[:2]}%'"
                condition = f"{column} {operator} {pattern}"
            else:
                condition = f"{column} {operator} '{value}'"
        else:
            continue

        filters.append(condition)

    return filters

def generate_sql_from_template_info(template_info, range_dict, num_conditions=2):
    components = template_info

    # 生成新的过滤条件
    new_filter_conditions = generate_new_filters(components['columns'], range_dict, num_conditions)
    # 将新的过滤条件解析为表达式
    new_filter_expressions = [sqlglot.parse_one(cond, read='mysql') for cond in new_filter_conditions]

    # 合并连接条件和新的过滤条件
    all_conditions = components['join_conditions'] + new_filter_expressions

    # 构建新的 WHERE 子句
    if all_conditions:
        where_expression = all_conditions[0]
        for cond in all_conditions[1:]:
            where_expression = sqlglot.expressions.And(this=where_expression, expression=cond)
        where_clause = sqlglot.expressions.Where(this=where_expression)
    else:
        where_clause = None

    # 构建 SELECT 子句
    select_clause = sqlglot.expressions.Select(expressions=components['select'])

    # 构建 FROM 子句
    from_clause_sql = ", ".join([f"{table_name} AS {table_alias}" for table_name, table_alias in components['from_tables']])
    from_clause = sqlglot.parse_one(f"FROM {from_clause_sql}", into=sqlglot.expressions.From)

    # 构建查询
    query = select_clause
    query.set("from", from_clause)
    if where_clause:
        query.set("where", where_clause)

    # 转换为 SQL 字符串
    new_sql = query.sql()

    return new_sql

# Assuming range_dict is already defined
# Example usage
if __name__ == "__main__":
    import json
    database = 'imdb'
    template_sql = """SELECT     MIN(mi.info) AS movie_budget,     MIN(mi_idx.info) AS movie_votes,     MIN(n.name) AS writer,     MIN(t.title) AS violent_liongate_movie FROM     cast_info AS ci,     company_name AS cn,     info_type AS it1,     info_type AS it2,     keyword AS k,     movie_companies AS mc,     movie_info AS mi,     movie_info_idx AS mi_idx,     movie_keyword AS mk,     name AS n,     title AS t WHERE     ci.note in (         '(writer)',         '(head writer)',         '(written by)',         '(story)',         '(story editor)'     )     AND cn.name like 'Lionsgate%'     AND it1.info = 'genres'     AND it2.info = 'votes'     AND k.keyword in (         'murder',         'violence',         'blood',         'gore',         'death',         'female-nudity',         'hospital'     )     AND mi.info in (         'Horror',         'Action',         'Sci-Fi',         'Thriller',         'Crime',         'War'     )     AND t.id = mi.movie_id     AND t.id = mi_idx.movie_id     AND t.id = ci.movie_id     AND t.id = mk.movie_id     AND t.id = mc.movie_id     AND ci.movie_id = mi.movie_id     AND ci.movie_id = mi_idx.movie_id     AND ci.movie_id = mk.movie_id     AND ci.movie_id = mc.movie_id     AND mi.movie_id = mi_idx.movie_id     AND mi.movie_id = mk.movie_id     AND mi.movie_id = mc.movie_id     AND mi_idx.movie_id = mk.movie_id     AND mi_idx.movie_id = mc.movie_id     AND mk.movie_id = mc.movie_id     AND n.id = ci.person_id     AND it1.id = mi.info_type_id     AND it2.id = mi_idx.info_type_id     AND k.id = mk.keyword_id     AND cn.id = mc.company_id;
"""
    with open(f'infos/{database}/range_dict', 'r') as f:
        range_dict = json.load(f)

    template_info = parse_template_sql(template_sql)
    new_sql = generate_sql_from_template_info(template_info, range_dict, num_conditions=3)
    print("Generated SQL:")
    print(new_sql)

    