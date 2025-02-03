def parse_ddl_for_columns(ddl: str) -> List[str]:
    # Regex to match column definitions
    # Matches: column_name TYPE(optional_parameters), or column_name TYPE,
    column_pattern = r'^\s*(\w+)\s+(?:(?:\w+\s*(?:\([^)]+\))?\s*,?)|(?:\w+\s*,?))'
    
    columns = []
    for line in ddl.split('\n'):
        # Skip comments, create statement, empty lines and closing bracket
        if not line.strip() or line.strip().startswith('--') or 'create' in line.lower() or ');' in line:
            continue
            
        match = re.match(column_pattern, line)
        if match:
            columns.append(match.group(1))
            
    return columns

def parse_ddl_tags(ddl: str) -> Dict[str, str]:
    # Regex to match TAG lines with DATA_SENSITIVITY
    tag_pattern = r'WITH\s+TAG\s+\([^)]*DATA_SENSITIVITY-\'(\w+)\'\)'
    
    tagged_columns = {}
    lines = ddl.split('\n')
    
    for i, line in enumerate(lines):
        if 'WITH TAG' in line and 'DATA_SENSITIVITY' in line:
            match = re.search(tag_pattern, line)
            if match and i > 0:
                # Get column name from previous line
                prev_line = lines[i-1]
                col_match = re.match(r'^\s*(\w+)', prev_line)
                if col_match:
                    sensitivity = match.group(1)
                    # Map the abbreviated sensitivity to full form
                    sensitivity_map = {
                        'CIF': 'Confidential Information',
                        'NSPII': 'Non-sensitive PII',
                        'SPII': 'Sensitive PII'
                    }
                    tagged_columns[col_match.group(1)] = sensitivity_map.get(sensitivity, sensitivity)
                    
    return tagged_columns
