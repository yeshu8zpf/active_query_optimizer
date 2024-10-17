import sqlparse
from sqlparse.sql import Where, Identifier, IdentifierList, Token, Function, Parenthesis, Comparison
from sqlparse.tokens import Keyword, DML, Whitespace, Operator, Comparison as ComparisonToken

def parse_conditions(token_list):
    """递归解析 WHERE 子句，提取条件列表"""
    conditions = []
    tokens = list(token_list.flatten())
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]

        if token.ttype is Whitespace:
            idx += 1
            continue

        # 检查是否为比较操作符
        if token.ttype == Operator.Comparison:
            # 提取比较条件
            # 左侧操作数在 token.parent 的 left 属性
            if hasattr(token.parent, 'left'):
                left = str(token.parent.left).strip().lower()
            else:
                left = ''
            op = str(token).strip()
            # 右侧操作数在 token.parent 的 right 属性
            if hasattr(token.parent, 'right'):
                right = str(token.parent.right).strip().lower()
            else:
                right = ''
            conditions.append((left, op, right))
            idx += 1  # 继续下一个 token
        elif token.is_group:
            # 递归解析嵌套条件
            conditions.extend(parse_conditions(token))
            idx += 1
        else:
            idx += 1

    return conditions

def extract_join_and_filter_lists(sql_queries):
    """
    提取所有 SQL 查询中的连接条件、过滤列和别名映射，并将它们转换为小写。
    :param sql_queries: SQL 查询字符串的列表
    :return: join_list（连接条件列表），filter_list（过滤列列表），alias_map（别名映射字典）
    """
    join_set = set()
    filter_set = set()
    global_alias_map = {}

    for sql_query in sql_queries:
        parsed = sqlparse.parse(sql_query)[0]
        alias_map = {}
        from_seen = False

        # 提取表和别名
        tokens = parsed.tokens
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.ttype is Whitespace:
                idx += 1
                continue
            if token.ttype is DML and token.value.upper() == 'SELECT':
                idx += 1
                continue
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                idx += 1
                continue
            if from_seen:
                if isinstance(token, IdentifierList):
                    identifiers = token.get_identifiers()
                elif isinstance(token, Identifier):
                    identifiers = [token]
                else:
                    identifiers = []
                for idf in identifiers:
                    # 提取表名和别名，并转换为小写
                    table_name = idf.get_real_name().lower()
                    alias = (idf.get_alias() or table_name).lower()
                    alias_map[alias] = table_name
                    global_alias_map[alias] = table_name  # 累积到全局别名映射
                from_seen = False
            if isinstance(token, Where):
                # 解析 WHERE 条件
                conditions = parse_conditions(token)
                for left, op, right in conditions:
                    # 左右操作数已在 parse_conditions 中转换为小写
                    op = op.strip()
                    # 检查是否为连接条件
                    if op == '=' and '.' in left and '.' in right:
                        left_table_alias = left.split('.')[0]
                        right_table_alias = right.split('.')[0]
                        left_col = alias_map.get(left_table_alias, left_table_alias) + '.' + left.split('.')[1]
                        right_col = alias_map.get(right_table_alias, right_table_alias) + '.' + right.split('.')[1]
                        join_condition = frozenset([left_col.lower(), right_col.lower()])
                        join_set.add(join_condition)
                    else:
                        # 处理过滤条件
                        if '.' in left:
                            table_alias = left.split('.')[0]
                            column_name = left.split('.')[1]
                            col_full_name = alias_map.get(table_alias, table_alias) + '.' + column_name.lower()
                            filter_set.add(col_full_name.lower())
                break  # 假设只有一个 WHERE 子句
            idx += 1

    # 将集合转换为列表
    join_list = list(join_set)
    filter_list = list(filter_set)

    return join_list, filter_list, global_alias_map

# 示例使用
if __name__ == "__main__":
    sql_queries = [
        """
        SELECT *
        FROM a AS A_Alias
        JOIN b ON A_Alias.a1 = b.b1
        JOIN c ON b.b1 = c.c1
        WHERE A_Alias.a1 > 10 AND b.b2 = 20 AND c.c2 <= 100
        """,
        """
        SELECT *
        FROM X
        JOIN Y ON X.x1 = Y.y1
        WHERE X.x2 < 50 AND Y.y2 >= 30
        """,
        """
        SELECT *
        FROM M AS m_alias, N AS n_alias
        WHERE m_alias.m1 = n_alias.n1 AND m_alias.m2 = 5
        """
    ]

    join_list, filter_list, alias_map = extract_join_and_filter_lists(sql_queries)
    print("Join List:")
    for join in join_list:
        print(join)
    print("\nFilter List:")
    for filt in filter_list:
        print(filt)
    print("\nAlias Map:")
    for alias, table in alias_map.items():
        print(f"{alias} -> {table}")

# 示例使用
if __name__ == "__main__":
    with open("data/test/stats_test_sql.txt", 'r') as f:
        lines = f.readlines()
    sql_queries = [line.split('#####')[1].strip() for line in lines]

    join_list, filter_list, alias_map = extract_join_and_filter_lists(sql_queries)
    print("Join List:")
    for join in join_list:
        print(join)
    print("\nFilter List:")
    for filt in filter_list:
        print(filt)
    print(alias_map)
