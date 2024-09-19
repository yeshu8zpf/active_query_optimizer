
import datetime
import psycopg2, re, json, os, time
from psycopg2 import OperationalError, sql

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
db_name = "imdb"
db_user = "postgres"
db_password = "li6545991360"
db_host = "127.0.0.1"  # 本地主机，对于远程数据库，请使用IP地址或域名
db_port = "5432"  # PostgreSQL默认端口是5432

connection = create_connection(db_name, db_user, db_password, db_host, db_port)


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


def get_alias_map(file='data/test_query/stats.txt'):
    with open(file, 'r') as f:
        lines = f.readlines()
    alias_map = {}
    rev_alias_map = {}
    for line in lines:
        q = line.split('#####')[1]
        t_part = re.findall(r'\b(\w+)\s+AS\s+(\w+)', q, re.IGNORECASE)
        for table, alias in t_part:
            alias_map[table] = alias
            rev_alias_map[alias] = table

    return alias_map, rev_alias_map




# 获取数据库中所有表的名称
def get_table_names():
    cursor = connection.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

# 获取指定表的所有列名
def get_column_names(table_name):
    cursor = connection.cursor()
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'")
    columns = cursor.fetchall()
    cursor.close()
    return [column[0] for column in columns]

def get_range_dict(table_names, columns, alias_map):
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


if __name__ == '__main__':
    database = 'imdb'
    if not os.path.exists(f'infos/{database}/alias_map'):
        alias_map, rev_alias_map = get_alias_map('data/test/imdb_test_sql.txt')
        with open(f'infos/{database}/alias_map', 'w') as f:
            json.dump(alias_map, f, indent=4)
        with open(f'infos/{database}/rev_alias_map', 'w') as f:
            json.dump(rev_alias_map, f, indent=4)
    with open(f'infos/{database}/alias_map', 'r') as f:
        alias_map = json.load(f)
    with open(f'infos/{database}/rev_alias_map', 'r') as f:
        rev_alias_map = json.load(f)

    table_names = get_table_names()
    if not os.path.exists(f'infos/{database}/columns'):
        columns = get_columns(table_names, alias_map)
        with open(f'infos/{database}/columns', 'w') as f:
            json.dump(columns, f, indent=4)
    with open(f'infos/{database}/columns', 'r') as f:
        columns = json.load(f)


    if not os.path.exists(f'infos/{database}/range_dict'):
        range_dict = get_range_dict(table_names, columns, alias_map)
        with open(f'infos/{database}/range_dict', 'w') as f:
            json.dump(range_dict, f, indent=4)
    with open(f'infos/{database}/range_dict', 'r') as f:
        range_dict = json.load(f)



    # 关闭数据库连接
    if connection:
        connection.close()


