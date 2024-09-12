
import psycopg2, re, json, os
from psycopg2 import OperationalError

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
db_name = "stats"
db_user = "postgres"
db_password = "li6545991360"
db_host = "127.0.0.1"  # 本地主机，对于远程数据库，请使用IP地址或域名
db_port = "5432"  # PostgreSQL默认端口是5432

connection = create_connection(db_name, db_user, db_password, db_host, db_port)

# 获取数据库中所有表的名称
def get_table_names():
    cursor = connection.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

# 获取指定表的所有列名
def get_column_names(table_name):
    cursor = connection.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'")
    columns = cursor.fetchall()
    return [column[0] for column in columns]

def get_range_dict(table_names, columns, aliasmap):
    range_dict = {}
    cursor = connection.cursor()
    for table in table_names:
        range_dict[alias_map[table]] = {}
        for c in columns[alias_map[table]]:    
            q = 'select min(%s), max(%s) from %s' % (c, c, table)
            cursor.execute(q)
            res = cursor.fetchall()
            range_dict[alias_map[table]][c] = res[0]
    return range_dict

def get_columns(table_names, alias_map):
    columns = {}
    table_names = get_table_names()
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

if not os.path.exists('infos/alias_map'):
    alias_map, rev_alias_map = get_alias_map()
    with open('infos/alias_map', 'w') as f:
        json.dump(alias_map, f, indent=4)
    with open('infos/rev_alias_map', 'w') as f:
        json.dump(rev_alias_map, f, indent=4)
with open('infos/alias_map', 'r') as f:
    alias_map = json.load(f)
with open('infos/rev_alias_map', 'r') as f:
    rev_alias_map = json.load(f)

table_names = get_table_names()
if not os.path.exists('infos/columns'):
    columns = get_columns(table_names, alias_map)
    with open('infos/columns', 'w') as f:
        json.dump(columns, f, indent=4)
with open('infos/columns', 'r') as f:
    columns = json.load(f)

# time2int(connection)


if not os.path.exists('infos/range_dict'):
    range_dict = get_range_dict(table_names, columns, alias_map)
    with open('infos/range_dict', 'w') as f:
        json.dump(range_dict, f, indent=4)
with open('infos/range_dict', 'r') as f:
    range_dict = json.load(f)



# 关闭数据库连接
if connection:
    connection.close()


