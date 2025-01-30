# Most flexible regex that will work for any column definition
regex_pattern = r'(?si)CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?TABLE\s+(?:[a-zA-Z0-9_"]+\.)?[a-zA-Z0-9_"]+\s*\((.*?)\)\s*;?'

# Test cases
test_cases = [
    """
    CREATE OR REPLACE TABLE schema.test_table (
        sr_id,
        -- Dummy Samples: 1001, 1002, 1003, 1004, 1005
        customer_name,
        -- Dummy Samples: John Smith, Jane Doe, Bob Wilson
        amount NUMBER(10,2),
        -- Dummy Samples: 1500.50, 2000.75, 3000.25
        created_at
        -- Dummy Samples: 2024-01-01 10:00:00, 2024-01-02 15:30:00
    );
    """,
    """
    CREATE TABLE test (
        sr_id NUMBER(38,0),
        -- Dummy Samples: 1, 2, 3
        col2
        -- Dummy Sample: Some long text here
    );
    """,
    """
    CREATE OR REPLACE TABLE t1 (column1);
    """
]

for test in test_cases:
    match = re.search(regex_pattern, test, re.MULTILINE | re.DOTALL)
    if match:
        print("Matched:")
        print(match.group(0))
        print("---")
