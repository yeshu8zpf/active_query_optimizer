import sqlparse
from sqlparse.sql import Where, Identifier, IdentifierList, Function, Comparison, Parenthesis
from sqlparse.tokens import Keyword, DML, Whitespace, Operator, Token
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import random

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
                if t.ttype in [Operator.Comparison, Token.Operator, Keyword]:
                    op = t.value.upper().strip()
                    break
            if not op:
                op = '='
            right = str(token.right).strip()
            conditions.append((left, op, right))
            idx += 1
        elif token.ttype is Keyword and token.value.upper() in ('IN', 'LIKE'):
            # Handle IN and LIKE operators not encapsulated in Comparison tokens
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
            # Move idx to the token after 'IN' or 'LIKE'
            idx += 1
            # Skip whitespace tokens to find the right-hand side
            while idx < len(tokens) and tokens[idx].ttype is Whitespace:
                idx += 1
            if idx < len(tokens):
                right_token = tokens[idx]
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


def extract_select_columns(sql_query):
    """
    Extract the columns inside the SELECT clause, including objects inside functions.

    Returns:
    - select_columns (List[str]): Sorted list of selected columns with aliases.
    """
    parsed = sqlparse.parse(sql_query)[0]
    tokens = parsed.tokens
    select_columns = []
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
                        select_columns.append(column_name)
            elif isinstance(token, Identifier):
                column_name = extract_column_name(token)
                if column_name:
                    select_columns.append(column_name)
            elif isinstance(token, Function):
                column_name = extract_column_from_function(token)
                if column_name:
                    select_columns.append(column_name)
            idx += 1
        else:
            idx += 1
    return sorted(select_columns)

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
            break
    return None
def extract_join_and_filter_lists(sql_query):
    """
    Extract join and filter conditions from SQL query, handling aliases.

    Returns:
    - join_conditions (List[Tuple[str, str]]): List of join column pairs.
    - filter_conditions (List[str]): List of filter columns.
    """
    parsed = sqlparse.parse(sql_query)[0]
    tokens = parsed.tokens
    joins = []
    filters = []
    from_seen = False

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
            # Skip over FROM clause processing since aliases are used as is
            from_seen = False
        if isinstance(token, Where):
            conditions = parse_conditions(token)
            for left, op, right in conditions:
                if op == '=' and '.' in left and '.' in right:
                    joins.append((left.lower(), right.lower()))
                else:
                    # Determine which side is the column and add to filters
                    if '.' in left:
                        filters.append(left.lower())
                    elif '.' in right and isinstance(right, str) and '.' in right:
                        filters.append(right.lower())
            break
        idx += 1

    return joins, filters

def generate_template_id(selected_columns: List[str], 
                         joins: List[Tuple[str, str]], 
                         filters: List[str]) -> str:
    """
    Generate a unique template identifier.

    Returns:
    - template_id (str): Unique template identifier.
    """
    selected_columns_str = '|'.join(sorted(selected_columns))
    sorted_joins = sorted(['_'.join(sorted([left, right])) for left, right in joins])
    joins_str = '|'.join(sorted_joins)
    filters_str = '|'.join(sorted(filters))
    template_id = f"SELECT:{selected_columns_str}|JOINS:{joins_str}"
    # template_id = f"SELECT:{selected_columns_str}|JOINS:{joins_str}|FILTERS:{filters_str}"
    return template_id

def group_sql_by_template(sql_queries: List[str]) -> Tuple[Dict[str, List[int]], List[str]]:
    """
    Group SQL queries by their templates.

    Returns:
    - template_to_sql_indices (Dict[str, List[int]]): Mapping from template ID to list of SQL indices.
    - templates (List[str]): List of all template IDs.
    """
    template_to_sql_indices = defaultdict(list)

    for idx, sql_query in enumerate(sql_queries):
        if not ('t1' in sql_query and 't2' in sql_query):
            continue
        joins, filters = extract_join_and_filter_lists(sql_query)
        selected_columns = extract_select_columns(sql_query)
        template_id = generate_template_id(selected_columns, joins, filters)
        template_to_sql_indices[template_id].append(idx)

    templates = list(template_to_sql_indices.keys())
    return template_to_sql_indices, templates

def sample_sql_queries(template_to_sql_indices: Dict[str, List[int]], 
                       proportions: np.ndarray, 
                       total_samples: int) -> List[int]:
    """
    Sample SQL queries based on template proportions.

    Returns:
    - sampled_sql_indices: List of sampled SQL query indices.
    """
    templates = list(template_to_sql_indices.keys())
    num_templates = len(templates)

    # Calculate the number of samples per template
    samples_per_template = (proportions * total_samples).astype(int)

    # Adjust for any rounding errors
    remainder = total_samples - np.sum(samples_per_template)
    if remainder > 0:
        for _ in range(remainder):
            idx = np.random.choice(num_templates)
            samples_per_template[idx] += 1
    elif remainder < 0:
        for _ in range(-remainder):
            idx = np.random.choice(num_templates)
            if samples_per_template[idx] > 0:
                samples_per_template[idx] -= 1

    # Sample queries from each template
    sampled_sql_indices = []
    for i, template in enumerate(templates):
        sql_indices = template_to_sql_indices[template]
        num_samples = samples_per_template[i]
        if num_samples >= len(sql_indices):
            sampled_sql_indices.extend(sql_indices)
        else:
            sampled_sql_indices.extend(random.sample(sql_indices, num_samples))

    return sampled_sql_indices

def generate_dirichlet_proportions(alpha, num_templates):
    """
    Generate sampling proportions using a Dirichlet distribution.

    Returns:
    - proportions: Sampling proportions for each template.
    """
    return np.random.dirichlet([alpha] * num_templates)

def create_imbalanced_sql_subset(sql_queries: List[str], 
                                 alpha: float, 
                                 total_samples: int) -> List[str]:
    """
    Create an imbalanced SQL subset using Dirichlet sampling.

    Returns:
    - subset_sql_queries: List of sampled SQL queries.
    """
    # Group SQL queries by template
    template_to_sql_indices, templates = group_sql_by_template(sql_queries)
    num_templates = len(templates)

    # Generate sampling proportions
    proportions = generate_dirichlet_proportions(alpha, num_templates)

    # Sample SQL queries
    sampled_sql_indices = sample_sql_queries(template_to_sql_indices, proportions, total_samples)

    # Retrieve the sampled SQL queries
    subset_sql_queries = [sql_queries[idx] for idx in sampled_sql_indices]

    return subset_sql_queries

if __name__ == '__main__':
    with open("data/unlabeled_train_data/imdb_train_pool.txt", 'r') as f:
        lines = f.readlines()
    sql_queries = [line.strip() for line in lines]

    total_samples = 1000
    alpha = 0.3
    imbalanced_sql_subset = create_imbalanced_sql_subset(sql_queries, alpha, total_samples)

    # Save the sampled SQL queries
    with open("data/unlabeled_train_data/imbalanced_imdb_train_subset.txt", 'w') as f:
        for sql in imbalanced_sql_subset:
            f.write(sql + '\n')
