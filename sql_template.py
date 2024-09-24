import sqlglot
from sqlglot import parse_one
from sqlglot.errors import ParseError

def extract_query_components(sql_query):
    """
    解析 SQL 查询并提取组件：
    - 使用的表
    - 连接条件
    - 过滤条件（WHERE 子句中的非连接条件）
    - SELECT 的列
    - 过滤条件中涉及的列
    """
    try:
        parsed = parse_one(sql_query)
    except ParseError as e:
        print(f"解析 SQL 时出错: {e}")
        return None

    components = {
        'tables': set(),
        'joins': [],
        'filters': [],
        'filter_columns': set(),
        'select': []
    }

    # 提取 SELECT 列
    select_exprs = parsed.expressions
    for expr in select_exprs:
        components['select'].append(expr.sql())

    # 提取 FROM 中的表
    from_clause = parsed.args.get('from')
    if from_clause:
        for table in from_clause.find_all(sqlglot.exp.Table):
            components['tables'].add(table.alias_or_name)

    # 提取 WHERE 条件，区分连接条件和过滤条件
    where = parsed.args.get('where')
    if where:
        # 传递 where.this 而不是 where
        join_conditions, filter_conditions = separate_conditions(where.this)
        components['joins'].extend([cond.sql() for cond in join_conditions])
        components['filters'].extend([cond.sql() for cond in filter_conditions])

        # 提取过滤条件中涉及的列
        filter_columns = set()
        for cond in filter_conditions:
            cols = extract_columns_from_condition(cond)
            filter_columns.update(cols)
        components['filter_columns'] = filter_columns

    return components

def separate_conditions(condition):
    join_conditions = []
    filter_conditions = []

    if not isinstance(condition, sqlglot.exp.Expression):
        return join_conditions, filter_conditions

    if isinstance(condition, sqlglot.exp.And):
        # 递归处理 AND 条件
        left = condition.this
        right = condition.expression

        left_joins, left_filters = separate_conditions(left)
        join_conditions.extend(left_joins)
        filter_conditions.extend(left_filters)

        right_joins, right_filters = separate_conditions(right)
        join_conditions.extend(right_joins)
        filter_conditions.extend(right_filters)

    elif isinstance(condition, sqlglot.exp.Or):
        # OR 条件视为过滤条件
        filter_conditions.append(condition)

    elif isinstance(condition, sqlglot.exp.EQ):
        left = condition.this
        right = condition.expression
        if is_column_from_different_tables(left, right):
            join_conditions.append(condition)
        else:
            filter_conditions.append(condition)

    elif isinstance(condition, (sqlglot.exp.Like, sqlglot.exp.GT, sqlglot.exp.LT,
                                sqlglot.exp.GTE, sqlglot.exp.LTE, sqlglot.exp.In,
                                sqlglot.exp.Between)):
        filter_conditions.append(condition)

    elif isinstance(condition, sqlglot.exp.Is):
        # 处理 IS NULL 条件
        filter_conditions.append(condition)

    elif isinstance(condition, sqlglot.exp.Not):
        # 处理 IS NOT NULL 条件，Not(this=Is(...))
        if isinstance(condition.this, sqlglot.exp.Is):
            filter_conditions.append(condition)
        else:
            # 其他类型的 Not 表达式，递归处理
            nested_joins, nested_filters = separate_conditions(condition.this)
            join_conditions.extend(nested_joins)
            filter_conditions.extend(nested_filters)

    else:
        # 其他类型的条件，视为过滤条件
        filter_conditions.append(condition)

    return join_conditions, filter_conditions

def is_column_from_different_tables(left, right):
    """
    检查左侧和右侧是否是来自不同表的列。
    """
    left_table = get_column_table(left)
    right_table = get_column_table(right)
    if left_table and right_table and left_table != right_table:
        return True
    return False

def get_column_table(expr):
    """
    获取列的表名。
    """
    if isinstance(expr, sqlglot.exp.Column):
        return expr.table
    elif isinstance(expr, sqlglot.exp.Identifier):
        # Identifier 可能是没有表名的列
        return None
    elif isinstance(expr, sqlglot.exp.Literal):
        # 字面量没有表名
        return None
    elif hasattr(expr, 'this'):
        return get_column_table(expr.this)
    else:
        return None

def extract_columns_from_condition(condition):
    """
    从条件表达式中提取所有涉及的列名（包含表别名）。
    返回列的集合，格式为 "表别名.列名" 或 "列名"（如果没有表别名）。
    """
    columns = set()
    for column in condition.find_all(sqlglot.exp.Column):
        if column.table:
            columns.add(f"{column.table}.{column.name}")
        else:
            columns.add(column.name)
    return columns

def compare_queries(template_components, query_components):
    """
    比较模板和查询的组件，判断是否匹配。
    """
    # 比较使用的表
    if template_components['tables'] != query_components['tables']:
        return False, False

    # 比较 SELECT 列
    if template_components['select'] != query_components['select']:
        return False, False

    # 比较 JOIN 条件（无视顺序）
    if set(template_components['joins']) != set(query_components['joins']):
        return False, False

    # 比较过滤条件涉及的列
    if template_components['filter_columns'] != query_components['filter_columns']:
        return True, False
    
    return True, True



def find_matching_template(query_sql, templates):
    """
    在模板列表中查找与查询匹配的模板。
    """
    query_components = extract_query_components(query_sql)
    if query_components is None:
        return None

    for idx, template_sql in enumerate(templates):
        template_components = extract_query_components(template_sql)
        if template_components is None:
            continue
        
        flag0, flag1 = compare_queries(template_components, query_components)
        if flag0:
            return (idx, flag1)  # 返回匹配的模板索引

    return None



if __name__ == "__main__":
    database = 'stats'
    # 定义模板列表
    templates = []
    templates_file = 'data/test/{database}_test_sql.txt'
    with open(templates_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        templates.append(line.split('#####')[1])

    # 定义需要检查的 SQL 查询列表
    sql_queries = []
    sql_file = f'data/labeled_train_data/{database}_train_sql.txt'
    with open(sql_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        sql_queries.append(line.split('#####')[1])

    # 检查每个查询是否匹配模板
    for i, sql_query in enumerate(sql_queries):
        matching_template_idx = find_matching_template(sql_query, templates)
        if matching_template_idx is not None:
            print(f"查询 {i} 匹配模板 {matching_template_idx[0]}")
            if not matching_template_idx[1]:
                print(f"sql:{extract_query_components(sql_query)['filters']}")
                print(f"template:{extract_query_components(templates[matching_template_idx[0]])['filters']}")
                pass
        else:
            print(f"查询 {i} 未匹配任何模板")
