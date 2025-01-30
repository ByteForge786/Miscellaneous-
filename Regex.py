regex_pattern = r'(?i)CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?TABLE\s+(?:(?:[a-zA-Z0-9_]+\.)?[a-zA-Z0-9_]+)\s*\([^)]*\)\s*;?'

# Let's break down each part and handle edge cases:
# (?i) - Case insensitive flag
# CREATE\s+ - 'CREATE' followed by any whitespace
# (?:OR\s+REPLACE\s+)? - Optional 'OR REPLACE' with any whitespace
# (?:TEMPORARY\s+)? - Optional 'TEMPORARY' with any whitespace
# TABLE\s+ - 'TABLE' with any whitespace
# (?: - Start non-capturing group for schema.table
#   (?:[a-zA-Z0-9_]+\.)? - Optional schema name with dot
#   [a-zA-Z0-9_]+ - Table name
# )\s* - End group, any whitespace
# \( - Opening parenthesis
# [^)]* - Any characters except closing parenthesis (column definitions)
# \) - Closing parenthesis
# \s* - Any whitespace after parenthesis
# ;? - Optional semicolon

# Test cases:
test_cases = [
    """CREATE TABLE t1(a int);""",
    """CREATE OR REPLACE TABLE t1(a int)""",
    """CREATE  TABLE 
    schema.table_name  ( 
        col1    int,
        col2    varchar
    );""",
    """create TEMPORARY TABLE t1(
        id int
    )  ;""",
    """CREATE OR   REPLACE    TABLE    t1    (col1 int,
    col2 varchar
    )""",
    """CREATE TABLE "SCHEMA"."TABLE"(a int);""",
    """CREATE table T1(
        /* comments here */
        col1 int -- line comment
    );"""
]

for test in test_cases:
    match = re.search(regex_pattern, test, re.MULTILINE | re.DOTALL)
    if match:
        print("Matched:", match.group(0))
        print("---")
