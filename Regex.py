regex_pattern = r'(?si)CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?(?:TABLE|VIEW)\s+(?:[a-zA-Z0-9_"\.\$\{\}]+\.)?[a-zA-Z0-9_"\.\$\{\}]+\s*\((.*?)\)\s*(?:as\b|\);|\)|\Z)'

# This pattern needs to handle:
# 1. Both TABLE and VIEW
# 2. Column lists without data types (like your VIEW example)
# 3. Statements ending with 'AS' clauses
# 4. Multiple line dummy samples
# 5. Complex subqueries and replacements

test_cases = [
    # Table with dummy samples
    """
    create or replace TABLE AUDIT_20250130 (
        SR_ID NUMBER(38,0),
        -- Dummy Samples: 1, 2, 3, 4, 5
        ISSUEID NUMBER(38,0)
    );
    """,
    
    # View with column list
    """
    create or replace view ATTRIBUTE_MASTER_VIEW(
        ATTRIBUTE_ID,
        ATTRIBUTE_NAME,
        ATTRIBUTE_DESC,
        ENTITY_ID,
        IS_KEY
    ) as 
    select * from ATTRIBUTE_MASTER;
    """,
    
    # Complex case with subqueries
    """
    create or replace table SOME_TABLE (
        CLOSURECOB,
        SUBCATEGORY,
        -- Dummy Samples: Clean Set, Clean Set, Clean Set
        FIXDATE,
        EXCEPTIONITEMID,
        -- Dummy Sample: TRADEACCOUNT_COUNTERPARTY_INTERNAL|400237|1164068
        COBDATE
    ) as
    select distinct RMD_ID.ne.* from
    (select coalesce((replace(DATAVALUE:ORG_RDM_ID,'','')),(replace(DATAVALUE
    -- Dummy Samples: 0, 0, 0, 0, 0
    from target_clone.exception_attribute
    where sk_owner_id=12) ea;
    """,
    
    # View with long column list
    """
    create or replace view MY_VIEW(
        col1,
        col2,
        -- Dummy Samples: val1, val2, val3
        col3,
        col4
    ) as select * from base_table;
    """
]

for test in test_cases:
    match = re.search(regex_pattern, test, re.MULTILINE | re.DOTALL)
    if match:
        print("Matched:")
        print(match.group(0))
        print("---")
