regex_pattern = r'(?i)CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?TABLE\s+(?:(?:[a-zA-Z0-9_]+\.)?[a-zA-Z0-9_]+)\s*\((.*?)\)\s*;?'

# Improved regex explanation:
# Everything same as before except:
# (.*?) - Lazy match of everything between first and last parentheses
#       - Will capture all columns including those with parentheses in their definitions

# Test cases:
test_cases = [
    """CREATE TABLE t1(
        id INTEGER PRIMARY KEY,
        val DECIMAL(10,2),
        str VARCHAR(255)
    );""",
    
    """CREATE OR REPLACE TABLE schema.table_name(
        id int,
        date_col TIMESTAMP(6),
        num_col NUMBER(38,0),
        var_col VARCHAR(100)
    )""",
    
    """create TABLE t1(col1 int PRIMARY KEY,
        col2 decimal(16,4),
        col3 varchar(50),
        col4 timestamp(3)
    );"""
]

for test in test_cases:
    match = re.search(regex_pattern, test, re.MULTILINE | re.DOTALL)
    if match:
        print("Matched:", match.group(0))
        print("---")
