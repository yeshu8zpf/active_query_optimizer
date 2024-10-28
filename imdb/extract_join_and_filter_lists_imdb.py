import sqlparse
from sqlparse.sql import (
    Where,
    Identifier,
    IdentifierList,
    Function,
    Comparison,
    Parenthesis,
    TokenList,
    Token,
)
from sqlparse.tokens import Keyword, DML, Whitespace, Operator, Comparison as ComparisonToken


def parse_conditions(token_list):
    """Recursively parse WHERE clause to extract conditions, including IN, LIKE, BETWEEN."""
    conditions = []
    tokens = token_list.tokens
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        if isinstance(token, sqlparse.sql.Where):
            # Recurse into WHERE clause
            conditions.extend(parse_conditions(token))
            idx += 1
        elif isinstance(token, sqlparse.sql.Comparison):
            # Handle standard comparisons
            left = str(token.left).strip()
            op = ''
            # Find the operator token
            for t in token.tokens:
                if t.ttype in [Operator.Comparison, Operator, Keyword]:
                    op = t.value.upper().strip()
                    break
            if not op:
                op = '='
            right = str(token.right).strip()
            conditions.append((left, op, right))
            idx += 1
        elif token.ttype is Keyword and token.value.upper() in ('IN', 'LIKE', 'BETWEEN'):
            # Handle IN, LIKE, BETWEEN operators not encapsulated in Comparison tokens
            op = token.value.upper()
            # Look for the left-hand side (column name)
            left_idx = idx - 1
            while left_idx >= 0 and tokens[left_idx].ttype is Whitespace:
                left_idx -= 1
            if left_idx >= 0:
                left_token = tokens[left_idx]
                if isinstance(left_token, Identifier):
                    left = str(left_token).strip()
                else:
                    left = None
            else:
                left = None
            # Move idx to the token after 'IN', 'LIKE', or 'BETWEEN'
            idx += 1
            # Skip whitespace tokens to find the right-hand side
            while idx < len(tokens) and tokens[idx].ttype is Whitespace:
                idx += 1
            if idx < len(tokens):
                right_token = tokens[idx]
                if isinstance(right_token, Parenthesis):
                    right = str(right_token).strip()
                else:
                    right = str(right_token).strip()
                idx += 1
            else:
                right = ''
            if left:
                conditions.append((left, op, right))
        elif token.is_group:
            # Recurse into group tokens
            conditions.extend(parse_conditions(token))
            idx += 1
        else:
            idx += 1
    return conditions


def extract_column_name(identifier):
    """
    Extract column name from an Identifier.

    Returns:
    - column_name: e.g., 't1.title' or 'title'
    """
    # Check if the identifier contains a function
    if isinstance(identifier.tokens[0], Function):
        # Extract the column from the function
        return extract_column_from_function(identifier.tokens[0])
    else:
        if identifier.get_parent_name():
            parent_name = identifier.get_parent_name().lower()
            column_name = identifier.get_real_name().lower()
            return f"{parent_name}.{column_name}"
        else:
            # No parent name, just return the real name
            return identifier.get_real_name().lower()


def extract_column_from_function(function):
    """
    Extract column name from a Function token.

    Returns:
    - column_name: string, e.g., 't1.title'.
    """
    # The tokens inside the function are typically:
    # [Function Name (e.g., 'MIN'), Parenthesis]
    for token in function.tokens:
        if isinstance(token, Parenthesis):
            # Remove parentheses and parse the content
            content = token.value[1:-1].strip()
            sub_parsed = sqlparse.parse(content)[0]
            for sub_token in sub_parsed.tokens:
                if isinstance(sub_token, Identifier):
                    return extract_column_name(sub_token)
                elif isinstance(sub_token, Function):
                    # Handle nested functions
                    return extract_column_from_function(sub_token)
                elif sub_token.ttype not in (Whitespace, sqlparse.tokens.Punctuation):
                    # Handle cases like "table.column" without alias
                    return sub_token.value.lower()
            break
    return None


def extract_select_columns(sql_query):
    """
    Extract the columns inside the SELECT clause, including objects inside functions.

    Returns:
    - select_columns (List[str]): Sorted list of selected columns with aliases.
    """
    parsed = sqlparse.parse(sql_query)[0]
    tokens = parsed.tokens
    select_columns = set()  # Use a set to deduplicate
    in_select = False

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.ttype is Whitespace:
            idx += 1
            continue
        if token.ttype is DML and token.value.upper() == 'SELECT':
            in_select = True
            idx += 1
            continue
        if in_select:
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                break
            if isinstance(token, IdentifierList):
                for idf in token.get_identifiers():
                    column_name = extract_column_name(idf)
                    if column_name:
                        select_columns.add(column_name)
            elif isinstance(token, Identifier):
                column_name = extract_column_name(token)
                if column_name:
                    select_columns.add(column_name)
            elif isinstance(token, Function):
                column_name = extract_column_from_function(token)
                if column_name:
                    select_columns.add(column_name)
            idx += 1
        else:
            idx += 1
    return sorted(select_columns)


def extract_from_clause(parsed):
    """
    Extract tables and aliases from the FROM clause.
    Returns a list of (table_name, alias) tuples.
    """
    from_seen = False
    from_tokens = []
    for token in parsed.tokens:
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True
            continue
        if from_seen:
            if token.ttype is Keyword:
                # Reached another clause (e.g., WHERE)
                break
            from_tokens.append(token)

    # Now process from_tokens to extract tables and aliases
    tables = []
    for token in from_tokens:
        if token.ttype in (Whitespace, sqlparse.tokens.Punctuation):
            continue
        if isinstance(token, IdentifierList):
            for idf in token.get_identifiers():
                table_name = idf.get_real_name().lower()
                alias = (idf.get_alias() or table_name).lower()
                tables.append((table_name, alias))
        elif isinstance(token, Identifier):
            table_name = token.get_real_name().lower()
            alias = (token.get_alias() or table_name).lower()
            tables.append((table_name, alias))
        elif isinstance(token, Function):
            # Handle cases where a table is defined via a function (e.g., subqueries)
            table_name = token.get_name().lower()
            alias = (token.get_alias() or table_name).lower()
            tables.append((table_name, alias))
    return tables


def is_column(name, alias_map):
    """
    Determine if the given name represents a column based on the alias_map.

    Args:
    - name (str): The identifier to check (e.g., 't1.title').
    - alias_map (dict): Mapping of aliases to table names.

    Returns:
    - bool: True if name is a column, False otherwise.
    """
    if '.' in name:
        alias, col = name.split('.', 1)
        return alias in alias_map
    return False


def extract_join_and_filter_lists(sql_queries):
    """
    Extract join conditions, filter columns, alias mapping, and select_list from SQL queries.
    :param sql_queries: List of SQL query strings
    :return: join_list, filter_list, alias_map, select_list
    """
    join_set = set()
    filter_set = set()
    global_alias_map = {}
    global_select_set = set()

    for sql_query in sql_queries:
        parsed = sqlparse.parse(sql_query)[0]
        alias_map = {}
        select_list = extract_select_columns(sql_query)
        global_select_set.update(select_list)

        # Extract tables and aliases from FROM clause
        tables = extract_from_clause(parsed)
        for table_name, alias in tables:
            alias_map[alias] = table_name
            global_alias_map[alias] = table_name

        # Parse WHERE clause
        where_clause = None
        for token in parsed.tokens:
            if isinstance(token, Where):
                where_clause = token
                break
        if where_clause:
            conditions = parse_conditions(where_clause)
            idx = 0
            while idx < len(conditions):
                condition = conditions[idx]
                if isinstance(condition, tuple) and len(condition) == 3:
                    left, op, right = condition
                    op = op.strip()
                    # Check if it's a join condition
                    is_left_column = is_column(left, alias_map)
                    is_right_column = is_column(right, alias_map)

                    if op == '=' and is_left_column and is_right_column:
                        # Join condition
                        left_col = left.lower()
                        right_col = right.lower()
                        join_condition = frozenset([left_col, right_col])
                        join_set.add(join_condition)
                    else:
                        # Handle filter condition
                        if is_left_column:
                            table_alias = left.split('.')[0]
                            column_name = left.split('.')[1]
                            col_full_name = f"{table_alias}.{column_name.lower()}"
                            filter_set.add(col_full_name.lower())
                        if is_right_column and op in ('=', 'LIKE', 'IN', 'BETWEEN', '>', '<', '>=', '<='):
                            # Right side could also be a column in some cases
                            table_alias = right.split('.')[0]
                            column_name = right.split('.')[1]
                            col_full_name = f"{table_alias}.{column_name.lower()}"
                            filter_set.add(col_full_name.lower())
                    idx += 1
                elif condition in ('AND', 'OR'):
                    # Logical operators; no action needed for now
                    idx += 1
                else:
                    # Unhandled condition type
                    idx += 1

    join_list = list(join_set)
    filter_list = list(filter_set)
    select_list = list(global_select_set)

    return join_list, filter_list, global_alias_map, select_list


# Example usage
if __name__ == "__main__":
    sql_queries = [
        """
        SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS mainstream_movie
        FROM company_name AS cn, company_type AS ct, info_type AS it1, info_type AS it2, movie_companies AS mc,
             movie_info AS mi, movie_info_idx AS mi_idx, title AS t
        WHERE cn.country_code = '[us]'
            AND ct.kind = 'production companies'
            AND it1.info = 'genres'
            AND it2.info = 'rating'
            AND mi.info in ('Drama', 'Horror', 'Western', 'Family')
            AND mi_idx.info > '7.0'
            AND t.production_year between 2000 and 2010
            AND t.id = mi.movie_id
            AND t.id = mi_idx.movie_id
            AND mi.info_type_id = it1.id
            AND mi_idx.info_type_id = it2.id
            AND t.id = mc.movie_id
            AND ct.id = mc.company_type_id
            AND cn.id = mc.company_id
            AND mc.movie_id = mi.movie_id
            AND mc.movie_id = mi_idx.movie_id
            AND mi.movie_id = mi_idx.movie_id;
        """
    ]

    join_list, filter_list, alias_map, select_list = extract_join_and_filter_lists(sql_queries)
    print("Join List:")
    for join in join_list:
        print(join)
    print("\nFilter List:")
    for filt in filter_list:
        print(filt)
    print("\nAlias Map:")
    for alias, table in alias_map.items():
        print(f"{alias} -> {table}")
    print("\nSelect List:")
    for select in select_list:
        print(select)
