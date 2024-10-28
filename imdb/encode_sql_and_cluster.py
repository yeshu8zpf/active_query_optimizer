import time
import numpy as np
import sqlparse
import json
from collections import defaultdict
from sqlparse.sql import Where, Identifier, IdentifierList, Function, Comparison, Parenthesis
from sqlparse.tokens import Keyword, DML, Whitespace, Operator, Comparison as ComparisonToken
from sklearn_extra.cluster import KMedoids
import psycopg2  # For PostgreSQL queries
import cupy as cp  # For GPU acceleration
from extract_join_and_filter_lists_imdb import extract_join_and_filter_lists, extract_from_clause, is_column, parse_conditions, extract_select_columns




# Update compute_distance_matrix_vectorized_gpu function
def compute_distance_matrix_vectorized_gpu(encodings, join_list_length, filter_list_length, select_list_length,
                                           d_max=2, w_join=0.5, w_filter=0.5, w_select=0.4):
    """
    Compute the distance matrix using GPU acceleration.

    :param encodings: List of encodings, shape (N, D)
    :param join_list_length: Length of join encoding
    :param filter_list_length: Number of filter columns
    :param select_list_length: Length of select encoding
    :param d_max: Maximum distance for filters
    :param w_join: Weight for join distance
    :param w_filter: Weight for filter distance
    :param w_select: Weight for select distance
    :return: Distance matrix, shape (N, N)
    """
    # Transfer data to GPU
    encodings_gpu = cp.asarray(encodings)

    N = encodings_gpu.shape[0]
    idx = 0
    # Extract join encodings
    join_encodings = encodings_gpu[:, idx:idx + join_list_length]  # Shape: (N, join_list_length)
    idx += join_list_length
    # Extract filter encodings
    filter_encodings = encodings_gpu[:, idx:idx + filter_list_length * 2]  # Each filter column has 2 values
    filter_encodings = filter_encodings.reshape(N, filter_list_length, 2)  # Shape: (N, filter_list_length, 2)
    idx += filter_list_length * 2
    # Extract select encodings
    select_encodings = encodings_gpu[:, idx:idx + select_list_length]  # Shape: (N, select_list_length)

    # Compute normalized join distance
    join_diff = join_encodings[:, cp.newaxis, :] != join_encodings[cp.newaxis, :, :]  # Shape: (N, N, join_list_length)
    D_join = cp.sum(join_diff, axis=2)  # Shape: (N, N)
    D_join_norm = D_join / join_list_length if join_list_length > 0 else 0

    # Compute filter distance
    total_filter_distance_matrix = cp.zeros((N, N))
    max_filter_distance = filter_list_length * d_max

    filter_presence = filter_encodings[:, :, 0]  # Shape: (N, filter_list_length)
    filter_frequency = filter_encodings[:, :, 1]  # Shape: (N, filter_list_length)

    for i in range(filter_list_length):
        f1_present = filter_presence[:, i][:, cp.newaxis]  # Shape: (N, 1)
        f2_present = filter_presence[:, i][cp.newaxis, :]  # Shape: (1, N)

        # Both filters present
        f_present_both = (f1_present == 1) & (f2_present == 1)
        # Only one filter present
        f_present_diff = (f1_present != f2_present)

        # Compute frequency difference
        freq_diff = cp.abs(filter_frequency[:, i][:, cp.newaxis] - filter_frequency[:, i][cp.newaxis, :])

        # Initialize distance matrix for this filter
        d_i = cp.zeros((N, N))
        d_i[f_present_both] = freq_diff[f_present_both]
        d_i[f_present_diff] = d_max
        # When both filters are absent, distance is zero (default)

        total_filter_distance_matrix += d_i

    D_filter_norm = total_filter_distance_matrix / max_filter_distance if max_filter_distance > 0 else 0

    # Compute select distance
    select_diff = select_encodings[:, cp.newaxis, :] != select_encodings[cp.newaxis, :, :]  # Shape: (N, N, select_list_length)
    D_select = cp.sum(select_diff, axis=2)  # Shape: (N, N)
    D_select_norm = D_select / select_list_length if select_list_length > 0 else 0

    # Combine distances
    D_total = w_join * D_join_norm + w_filter * D_filter_norm + w_select * D_select_norm

    # Transfer result back to CPU
    D_total_cpu = cp.asnumpy(D_total)
    return D_total_cpu

def encode_sql_query(sql_query, alias_map, join_list, filter_list, select_list, conn, frequency_cache=None):
    """
    Encode SQL query conditions into a vector representation, including join conditions, filter conditions, and select objects.
    - For filters, compute the frequency (percentage of rows returned over total rows).
    - Each filter column is represented by two values: [presence (0/1), frequency (0-1)].
    - Select objects are encoded as 0/1 indicating presence or absence.

    :param sql_query: SQL query string
    :param alias_map: Alias mapping dictionary, e.g., {'a_alias': 'a', ...}
    :param join_list: List of all possible join conditions
    :param filter_list: List of filter columns, e.g., ['a_alias.a1', 'b_alias.b2', ...]
    :param select_list: List of select columns, e.g., ['a_alias.a1', 'b_alias.b2', ...]
    :param conn: psycopg2 connection object to PostgreSQL
    :param frequency_cache: Dictionary to cache frequency computations
    :return: Encoded vector including join conditions, filter conditions, and select objects
    """
    # Initialize frequency_cache if not provided
    if frequency_cache is None:
        frequency_cache = {}

    # 1. Parse the SQL query
    parsed = sqlparse.parse(sql_query)[0]
    actual_joins = []
    actual_filters = []
    actual_selects = []
    local_alias_map = {}  # Local alias mapping

    # 2. Extract tables and aliases
    tables = extract_from_clause(parsed)
    for table_name, alias in tables:
        # Correct alias_map: alias -> table_name
        local_alias_map[alias] = table_name

    # 3. Extract select columns
    select_columns = extract_select_columns(sql_query)
    actual_selects = [col.lower() for col in select_columns]

    # 4. Parse WHERE conditions
    where_clause = None
    for token in parsed.tokens:
        if isinstance(token, Where):
            where_clause = token
            break
    if where_clause:
        conditions = parse_conditions(where_clause)
        for left, op, right in conditions:
            left = left.strip().lower()
            op = op.strip()
            right = right.strip().lower()
            # Check if it's a join condition
            is_left_column = is_column(left, local_alias_map)
            is_right_column = is_column(right, local_alias_map)
            if op == '=' and is_left_column and is_right_column:
                # Join condition
                left_col = left
                right_col = right
                join_condition = frozenset([left_col.lower(), right_col.lower()])
                actual_joins.append(join_condition)
            else:
                # Filter condition
                if is_left_column:
                    table_alias = left.split('.')[0]
                    column_name = left.split('.')[1]
                    col_full_name = f"{table_alias}.{column_name}"
                    actual_filters.append((col_full_name.lower(), op, right))
                elif is_right_column:
                    table_alias = right.split('.')[0]
                    column_name = right.split('.')[1]
                    col_full_name = f"{table_alias}.{column_name}"
                    actual_filters.append((col_full_name.lower(), op, left))

    # 5. Build index mappings
    join_list_lower = [frozenset([col.lower() for col in join]) for join in join_list]
    filter_list_lower = [col.lower() for col in filter_list]
    select_list_lower = [col.lower() for col in select_list]

    join_index_map = {join: idx for idx, join in enumerate(join_list_lower)}
    filter_index_map = {col_name: idx for idx, col_name in enumerate(filter_list_lower)}
    select_index_map = {col_name: idx for idx, col_name in enumerate(select_list_lower)}

    total_filter_columns = len(filter_list_lower)
    total_select_columns = len(select_list_lower)

    # 6. Initialize encoding vectors
    join_encoding = np.zeros(len(join_list_lower), dtype=int)
    filter_encoding = np.full((total_filter_columns, 2), -1.0)  # Initialize with -1.0
    filter_encoding[:, 0] = 0.0  # Presence flag initialized to 0
    select_encoding = np.zeros(len(select_list_lower), dtype=int)

    # 7. Encode join conditions
    actual_joins_set = set(actual_joins)
    for join in actual_joins_set:
        if join in join_index_map:
            idx = join_index_map[join]
            join_encoding[idx] = 1

    # 8. Encode filter conditions
    for col_full_name, op, right in actual_filters:
        if col_full_name in filter_index_map:
            col_index = filter_index_map[col_full_name]
            filter_encoding[col_index][0] = 1  # Presence flag

            # Compute frequency using PostgreSQL or use cache
            filter_key = (col_full_name, op, right)
            if filter_key in frequency_cache:
                frequency = frequency_cache[filter_key]
            else:
                # Split the column into alias and column name
                try:
                    table_alias, column_name = col_full_name.split('.')
                except ValueError:
                    # Invalid column format; skip this filter
                    frequency = 0.0
                    frequency_cache[filter_key] = frequency
                    filter_encoding[col_index][1] = frequency
                    continue

                table_name = local_alias_map.get(table_alias)
                if not table_name:
                    # Alias not found; skip frequency computation
                    frequency = 0.0
                    frequency_cache[filter_key] = frequency
                else:
                    try:
                        # Build the WHERE clause condition
                        condition = f"{col_full_name} {op} {right}"
                        # Query total number of rows in the table
                        total_rows_query = f"SELECT COUNT(*) FROM {table_name}"
                        # Query number of rows matching the condition
                        condition_rows_query = f"SELECT COUNT(*) FROM {table_name} WHERE {condition}"

                        with conn.cursor() as cur:
                            cur.execute(total_rows_query)
                            total_rows = cur.fetchone()[0]
                            if total_rows == 0:
                                frequency = 0.0
                            else:
                                cur.execute(condition_rows_query)
                                condition_rows = cur.fetchone()[0]
                                frequency = condition_rows / total_rows
                        # Store the frequency in the cache
                        frequency_cache[filter_key] = frequency
                    except Exception as e:
                        # If any error occurs, set frequency to 0.0 and cache it
                        frequency = 0.0
                        frequency_cache[filter_key] = frequency
                filter_encoding[col_index][1] = frequency
        else:
            continue  # Column not in filter list, skip

    # 9. Encode select objects
    for col_name in actual_selects:
        if col_name in select_index_map:
            idx = select_index_map[col_name]
            select_encoding[idx] = 1

    # 10. Combine encoding vectors
    encoding_vector = np.concatenate([join_encoding, filter_encoding.flatten(), select_encoding])

    return encoding_vector

# Example usage
if __name__ == "__main__":
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="imdb",  # Replace with your database name
            user="postgres",   # Replace with your username
            password="your_password"  # Replace with your password
        )
    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        exit(1)

    try:
        # Load test SQL queries
        with open("data/test/imdb_test_sql.txt", 'r') as f:
            lines = f.readlines()
        sql_queries_test = [line.split('#####')[1].strip() for line in lines if '#####' in line]

        # Extract join_list, filter_list, alias_map, and select_list from test queries
        join_list, filter_list, alias_map, select_list = extract_join_and_filter_lists(sql_queries_test)

        # Load training SQL queries
        with open('data/unlabeled_train_data/imdb_train_pool.txt', 'r') as f:
            lines = f.readlines()[:10000]
        sql_queries_train = [line.split('#####')[1].strip() for line in lines if '#####' in line]

        # Initialize frequency_cache
        frequency_cache = {}

        # Initialize list to store encodings
        encodings = []

        # Iterate over training SQL queries and encode them
        for idx, sql_query in enumerate(sql_queries_train):
            encoding = encode_sql_query(
                sql_query, alias_map, join_list, filter_list, select_list, conn, frequency_cache=frequency_cache
            )
            encodings.append(encoding)
            if (idx + 1) % 100 == 0:
                print(f"Encoded {idx + 1} / {len(sql_queries_train)} queries")

        # Convert encodings to a NumPy array and save to disk
        encodings = np.stack(encodings)
        np.save('data/tmp/imdb_train_sql_encodings.npy', encodings)
        print("Encoding of training SQL queries completed and saved.")

        # Optionally, load encodings if needed
        # encodings = np.load('data/tmp/imdb_train_sql_encodings.npy')

        # Compute distance matrix
        t1 = time.time()
        distance_matrix = compute_distance_matrix_vectorized_gpu(
            encodings,
            len(join_list),
            len(filter_list),
            len(select_list),
            w_join=0.8,
            w_filter=0.2,
            w_select=0.2
        )
        print(f"Distance matrix computed in {time.time() - t1:.2f} seconds.")

        # Perform KMedoids clustering
        num_groups = 100
        kmedoids = KMedoids(n_clusters=num_groups, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(distance_matrix)
        print("KMedoids clustering completed.")

        # Grouping the indices based on cluster labels
        groups = [[] for _ in range(num_groups)]
        for idx, label in enumerate(labels):
            groups[label].append(idx)
        non_empty_cluster_count = 0
        for group_num, group in enumerate(groups):
            print(f"Group {group_num + 1}: {group}")
            if group:
                non_empty_cluster_count += 1
        print(f'num of non-empty cluster: {non_empty_cluster_count}')
    except Exception as e:
        print("An error occurred during processing:", e)
    finally:
        # Close the database connection
        conn.close()
        print("Database connection closed.")

