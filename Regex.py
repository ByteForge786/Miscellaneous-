regex_pattern = r'(?si)CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?(?:TRANSIENT\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[a-zA-Z0-9_"\.\$\{?\}?]+\.)?[a-zA-Z0-9_"\.\$\{?\}?]+\s*\((.*?)\)\s*(?:;|\Z)'

# Test cases covering edge cases:
test_cases = [
    # Your current patterns
    """
    create or replace TABLE AUDIT_20250130 (
        SR_ID NUMBER(38,0),
        -- Dummy Samples: 1, 2, 3, 4, 5
        ISSUEID NUMBER(38,0)
    );
    """,
    
    # Different variations
    """
    CREATE TABLE my_table (col1);
    """,
    
    """
    CREATE TEMPORARY TABLE IF NOT EXISTS schema.table (
        col1 NUMBER(38,0),
        -- Dummy Samples: 1, 2, 3
        col2 VARCHAR
    );
    """,
    
    """
    CREATE OR REPLACE TRANSIENT TABLE ${database}.${schema}.table_name(
        id number(38,0),
        -- Dummy Samples: 1, 2, 3
        name varchar
    )
    """,
    
    """
    create table "SCHEMA"."TABLE"(
        "COLUMN1" number,
        -- Dummy Samples: 1, 2, 3
        "COLUMN2" varchar
    );
    """,
    
    """
    CREATE OR REPLACE TABLE database.schema.table (
        col1,
        -- Dummy Samples: val1, val2, val3
        col2 NUMBER,
        col3 VARCHAR(100)
    ) AS SELECT * FROM other_table;
    """,
    
    """
    CREATE TABLE IF NOT EXISTS MyTable(id int);
    """,
    
    # With comments and complex formatting
    """
    CREATE OR REPLACE 
    /* Multiple line
       comments */
    TABLE my_table ( -- inline comment
        id NUMBER(38,0), -- another comment
        -- Dummy Samples: 1, 2, 3
        name VARCHAR -- end comment
    );
    """
]

for test in test_cases:
    match = re.search(regex_pattern, test, re.MULTILINE | re.DOTALL)
    if match:
        print("Matched:")
        print(match.group(0))
        print("---")
