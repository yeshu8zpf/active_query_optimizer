
from collections import defaultdict
import datetime
import psycopg2, re, json, os, time
from psycopg2 import OperationalError, sql
import sqlglot
from sqlglot import expressions as exp
from collections import defaultdict

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


def convert_time_to_int(date_string):

    # 指定日期字符串的格式
    date_format = '%Y-%m-%d %H:%M:%S'

    # 使用 datetime.strptime 将字符串转换为 datetime 对象
    date_obj = datetime.strptime(date_string, date_format)

    # 使用 time.mktime 将 datetime 对象转换为 UNIX 时间戳
    unix_timestamp = int(time.mktime(date_obj.timetuple()))
    return unix_timestamp

def find_timestamp_columns(connection):
    cursor = connection.cursor()
    query = """
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE data_type IN ('timestamp without time zone', 'timestamp with time exchange')
    AND table_schema = 'public';
    """
    try:
        cursor.execute(query)
        timestamp_columns = cursor.fetchall()
        print("Timestamp columns found:", timestamp_columns)
        return timestamp_columns
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_timestamp_to_int(connection, table_name, column_name):
    cursor = connection.cursor()
    alter_query = f"""
    ALTER TABLE {table_name}
    ALTER COLUMN {column_name} SET DATA TYPE bigint
    USING EXTRACT(EPOCH FROM {column_name});
    """
    try:
        cursor.execute(alter_query)
        connection.commit()
        print(f"Column {column_name} in table {table_name} converted to INT successfully.")
    except Exception as e:
        print(f"An error occurred while converting {column_name} in table {table_name}: {e}")




# 获取数据库中所有表的名称
def get_table_names(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

# 获取指定表的所有列名
def get_column_names(table_name, connection):
    cursor = connection.cursor()
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'")
    columns = cursor.fetchall()
    cursor.close()
    return [column[0] for column in columns]

def get_range_dict(table_names, columns, alias_map, connection):
    range_dict = {}
    cursor = connection.cursor()
    for table in table_names:
        table_alias = alias_map[table]
        range_dict[table_alias] = {}
        # 获取该表中列的数据类型
        # 构建列名到数据类型的映射字典
        data_types = {}
        for c in columns[table_alias]:
            cursor.execute("""
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s;
                """, (table, c))
            res = cursor.fetchone()
            if res:
                data_types[c] = res[0]
            else:
                data_types[c] = None

        for c in columns[table_alias]:
            data_type = data_types.get(c)
            if data_type is None:
                continue  # 跳过未知数据类型的列
            if data_type in ('integer', 'bigint', 'smallint', 'decimal', 'numeric', 'real', 'double precision'):
                # 数值型列
                q = sql.SQL('SELECT MIN({col}), MAX({col}) FROM {table}').format(
                    col=sql.Identifier(c),
                    table=sql.Identifier(table)
                )
                cursor.execute(q)
                res = cursor.fetchone()
                range_dict[table_alias][c] = res  # (min, max)
            elif data_type in ('character varying', 'varchar', 'text', 'character', 'char'):
                # 字符串型列
                # 首先获取不同值的数量
                q = sql.SQL('SELECT COUNT(DISTINCT {col}) FROM {table} WHERE {col} IS NOT NULL').format(
                    col=sql.Identifier(c),
                    table=sql.Identifier(table)
                )
                cursor.execute(q)
                res = cursor.fetchone()
                distinct_count = res[0]
                if distinct_count <= 1000:  # 设置合理的限制
                    q = sql.SQL('SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL').format(
                        col=sql.Identifier(c),
                        table=sql.Identifier(table)
                    )
                    cursor.execute(q)
                    res = cursor.fetchall()
                    values = [row[0] for row in res]
                    range_dict[table_alias][c] = values
                else:
                    # 获取频率最高的 1000 个值
                    q = sql.SQL('''
                        SELECT {col}
                        FROM {table}
                        WHERE {col} IS NOT NULL
                        GROUP BY {col}
                        ORDER BY COUNT(*) DESC
                        LIMIT 1000
                        ''').format(
                            col=sql.Identifier(c),
                            table=sql.Identifier(table)
                        )
                    cursor.execute(q)
                    res = cursor.fetchall()
                    values = [row[0] for row in res]
                    range_dict[table_alias][c] = values
            else:
                print(f"数据类型 '{data_type}' 未定义 ")
                # 可以根据需要处理其他数据类型
                continue
    cursor.close()
    return range_dict



def get_columns(table_names, alias_map):
    columns = {}
    for table in table_names:
        print(f"Table: {table}")
        column = get_column_names(table)
        print("Columns:", column)
        columns[alias_map[table]] = column
    return columns

def time2int(connection):
# 查找时间戳列
    timestamp_columns = find_timestamp_columns(connection)

    # 转换列
    for table_name, column_name in timestamp_columns:
        convert_timestamp_to_int(connection, table_name, column_name)


import sqlglot
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

def extract_tables_and_aliases(sql):
    """
    从单个SQL查询中提取表和别名的映射。
    
    返回两个字典：
    1. table_to_aliases: {table: [alias1, alias2, ...]}
    2. alias_to_table: {alias: table}
    """
    table_to_aliases = {}
    alias_to_table = {}
    
    parsed = sqlparse.parse(sql)
    if not parsed:
        return table_to_aliases, alias_to_table
    
    stmt = parsed[0]
    from_seen = False
    for token in stmt.tokens:
        if from_seen:
            if token.ttype is Keyword:
                break
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    table = identifier.get_real_name()
                    alias = identifier.get_alias() or table
                    # 更新 table_to_aliases
                    if table not in table_to_aliases:
                        table_to_aliases[table] = []
                    if alias not in table_to_aliases[table]:
                        table_to_aliases[table].append(alias)
                    # 更新 alias_to_table
                    alias_to_table[alias] = table
            elif isinstance(token, Identifier):
                table = token.get_real_name()
                alias = token.get_alias() or table
                # 更新 table_to_aliases
                if table not in table_to_aliases:
                    table_to_aliases[table] = []
                if alias not in table_to_aliases[table]:
                    table_to_aliases[table].append(alias)
                # 更新 alias_to_table
                alias_to_table[alias] = table
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True
    return table_to_aliases, alias_to_table

def get_tables_aliases(sql_list):
    """
    处理SQL列表，返回两个字典：
    1. table_to_aliases: {table: [alias1, alias2, ...]}
    2. alias_to_table: {alias: table}
    """
    table_to_aliases = {}
    alias_to_table = {}
    
    for sql in sql_list:
        tbl_to_al, al_to_tbl = extract_tables_and_aliases(sql)
        # 合并 table_to_aliases
        for table, aliases in tbl_to_al.items():
            if table not in table_to_aliases:
                table_to_aliases[table] = set()
            table_to_aliases[table].update(aliases)
        # 合并 alias_to_table
        for alias, table in al_to_tbl.items():
            alias_to_table[alias] = table
    
    # 将 set 转换为 list
    table_to_aliases = {table: list(aliases) for table, aliases in table_to_aliases.items()}
    
    return table_to_aliases, alias_to_table

def is_column_from_different_tables(left, right, alias_to_table):
    """
    判断两个表达式是否来自不同的表。
    """
    def get_table(expr):
        if isinstance(expr, exp.Column):
            table = expr.table
            return alias_to_table.get(table, table)
        return None

    left_table = get_table(left)
    right_table = get_table(right)
    return left_table and right_table and left_table != right_table

def separate_conditions(condition, alias_to_table):
    """
    分离JOIN条件和过滤条件。
    """
    join_conditions = []
    filter_conditions = []

    if not isinstance(condition, exp.Expression):
        return join_conditions, filter_conditions

    if isinstance(condition, exp.And):
        left = condition.args.get("this")
        right = condition.args.get("expression")

        left_joins, left_filters = separate_conditions(left, alias_to_table)
        join_conditions.extend(left_joins)
        filter_conditions.extend(left_filters)

        right_joins, right_filters = separate_conditions(right, alias_to_table)
        join_conditions.extend(right_joins)
        filter_conditions.extend(right_filters)

    elif isinstance(condition, exp.Or):
        # OR条件被视为过滤条件
        filter_conditions.append(condition)

    elif isinstance(condition, exp.EQ):
        left = condition.args.get("this")
        right = condition.args.get("expression")
        if is_column_from_different_tables(left, right, alias_to_table):
            join_conditions.append(condition)
        else:
            filter_conditions.append(condition)

    elif isinstance(condition, (exp.Like, exp.In, exp.GT, exp.LT, exp.GTE, exp.LTE)):
        filter_conditions.append(condition)

    else:
        # 其他类型的条件被视为过滤条件
        filter_conditions.append(condition)

    return join_conditions, filter_conditions

def extract_filter_columns(sql, table_to_aliases, alias_to_table):
    """
    从单个SQL查询中提取每个表在过滤条件中使用的列。
    
    返回一个字典：{table: set(columns)}
    """
    filter_columns = defaultdict(set)
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception as e:
        print(f"Failed to parse SQL: {e}")
        return filter_columns

    where = parsed.args.get("where")
    if not where:
        return filter_columns

    # 分离JOIN条件和过滤条件
    join_conds, filter_conds = separate_conditions(where.this, alias_to_table)

    # 处理过滤条件
    for cond in filter_conds:
        columns = cond.find_all(exp.Column)
        for col in columns:
            alias = col.table
            column_name = col.name
            table = alias_to_table.get(alias, alias)
            if table:
                filter_columns[table].add(column_name)

    return filter_columns

def get_filter_columns(sql_list, table_to_aliases, alias_to_table):
    """
    处理SQL查询列表，提取每个表在过滤条件中使用的列。
    
    返回一个字典：{table: set(columns)}
    """
    overall_filter_columns = defaultdict(set)

    for sql in sql_list:
        filter_columns = extract_filter_columns(sql, table_to_aliases, alias_to_table)
        for table, columns in filter_columns.items():
            overall_filter_columns[table].update(columns)

    # 转换为普通字典
    overall_filter_columns = {table: list(columns) for table, columns in overall_filter_columns.items()}
    return overall_filter_columns

def get_filter_columns_range(filter_columns, alias_to_table, connection):
    """
    根据已获得的 filter_columns 和 table_to_aliases，获取每个别名在过滤条件中使用的列的取值范围。

    参数:
    - filter_columns (dict): {table: [columns]}
    - alias_to_table (dict): {alias: table}
    - connection: 数据库连接对象

    返回:
    - range_dict (dict): {alias: {column: range, ...}, ...}
      其中 range 为数值型时为 tuple (min, max)，字符串型时为列表
    """
    range_dict = defaultdict(dict)
    cursor = connection.cursor()

    for alias, table in alias_to_table.items():
        columns = filter_columns.get(table, [])
        if not columns:
            continue  # 该表在过滤条件中没有使用任何列

        # 获取列的数据类型
        data_types = {}
        for c in columns:
            try:
                cursor.execute("""
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s;
                """, (table, c))
                res = cursor.fetchone()
                if res:
                    data_types[c] = res[0]
                else:
                    data_types[c] = None
            except Exception as e:
                print(f"Error fetching data type for {table}.{c}: {e}")
                data_types[c] = None

        # 获取每个列的取值范围
        for c in columns:
            data_type = data_types.get(c)
            if not data_type:
                continue  # 跳过未知数据类型的列

            try:
                if data_type in ('integer', 'bigint', 'smallint', 'decimal', 'numeric', 'real', 'double precision'):
                    # 数值型列
                    q = sql.SQL('SELECT MIN({col}), MAX({col}) FROM {table}').format(
                        col=sql.Identifier(c),
                        table=sql.Identifier(table)  # 使用表名而非别名
                    )
                    cursor.execute(q)
                    res = cursor.fetchone()
                    range_dict[alias][c] = tuple(res)  # (min, max)

                elif data_type in ('character varying', 'varchar', 'text', 'character', 'char'):
                    # 字符串型列
                    # 首先获取不同值的数量
                    q_count = sql.SQL('SELECT COUNT(DISTINCT {col}) FROM {table} WHERE {col} IS NOT NULL').format(
                        col=sql.Identifier(c),
                        table=sql.Identifier(table)  # 使用表名而非别名
                    )
                    cursor.execute(q_count)
                    res_count = cursor.fetchone()
                    distinct_count = res_count[0] if res_count else 0

                    if distinct_count <= 1000:
                        # 获取所有不同的值
                        q_values = sql.SQL('SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL').format(
                            col=sql.Identifier(c),
                            table=sql.Identifier(table)  # 使用表名而非别名
                        )
                        cursor.execute(q_values)
                        res_values = cursor.fetchall()
                        values = [row[0] for row in res_values]
                        range_dict[alias][c] = values
                    else:
                        # 获取频率最高的 1000 个值
                        q_top = sql.SQL('''
                            SELECT {col}
                            FROM {table}
                            WHERE {col} IS NOT NULL
                            GROUP BY {col}
                            ORDER BY COUNT(*) DESC
                            LIMIT 1000
                        ''').format(
                            col=sql.Identifier(c),
                            table=sql.Identifier(table)  # 使用表名而非别名
                        )
                        cursor.execute(q_top)
                        res_top = cursor.fetchall()
                        values = [row[0] for row in res_top]
                        range_dict[alias][c] = values

                else:
                    print(f"数据类型 '{data_type}' 未定义，跳过 {table}.{c}")
                    continue

            except Exception as e:
                print(f"Error processing {table}.{c} with alias {alias}: {e}")
                continue

    cursor.close()
    return dict(range_dict)

if __name__ == '__main__':
    database = 'imdb'
    with open(f'data/test/{database}_test_sql.txt', 'r') as f:
        lines = f.readlines()
    sql_list = [line.strip().split('#####')[1] for line in lines]
    if not os.path.exists(f'infos/{database}/alias_map'):
        alias_map, rev_alias_map = get_tables_aliases(sql_list)
        with open(f'infos/{database}/alias_map', 'w') as f:
            json.dump(alias_map, f, indent=4)
        with open(f'infos/{database}/rev_alias_map', 'w') as f:
            json.dump(rev_alias_map, f, indent=4)
    else:
        with open(f'infos/{database}/alias_map', 'r') as f:
            alias_map = json.load(f)
        with open(f'infos/{database}/rev_alias_map', 'r') as f:
            rev_alias_map =  json.load(f)
    
    if not os.path.exists(f'infos/{database}/filter_columns'):
        filter_columns = get_filter_columns(sql_list, alias_map, rev_alias_map)
        with open(f'infos/{database}/filter_columns', 'w') as f:
            json.dump(filter_columns, f, indent=4)
    else:
        with open(f'infos/{database}/filter_columns', 'r') as f:    
            filter_columns = json.load(f)

    if not os.path.exists(f'infos/{database}/range_dict'):
        db_name = database
        db_user = "postgres"
        db_password = "li6545991360"
        db_host = "127.0.0.1"  # 本地主机，对于远程数据库，请使用IP地址或域名
        db_port = "5432"  # PostgreSQL默认端口是5432

        connection = create_connection(db_name, db_user, db_password, db_host, db_port)
        range_dict = get_filter_columns_range(filter_columns, rev_alias_map, connection)
        connection.close()
        with open(f'infos/{database}/range_dict', 'w') as f:
            json.dump(range_dict, f, indent=4)
    else:
        with open(f'infos/{database}/range_dict', 'r') as f:    
            range_dict = json.load(f)



    # 关闭数据库连接
    if connection:
        connection.close()


