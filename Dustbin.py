def parse_ddl_for_columns(ddl: str) -> List[str]:
    # More flexible regex to handle various DDL formats
    column_pattern = r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:(?:NUMBER|VARCHAR|DATE|TIMESTAMP_NTZ|TIMESTAMP|BOOLEAN|INTEGER|FLOAT|TIME|CHAR|TEXT|DATETIME)?(?:\s*\([^)]+\))?)?\s*,?'
    
    columns = []
    lines = ddl.split('\n')
    
    for i, line in enumerate(lines):
        # Skip comments, create statement, empty lines and closing bracket
        if (not line.strip() or 
            line.strip().startswith('--') or 
            'create' in line.lower() or 
            ');' in line or
            'WITH TAG' in line):  # Skip tag lines
            continue
            
        match = re.match(column_pattern, line)
        if match:
            col_name = match.group(1).strip()
            if col_name and col_name not in columns:  # Avoid duplicates
                columns.append(col_name)
                
    return columns

def parse_ddl_tags(ddl: str) -> pd.DataFrame:
    """Parse DDL to extract column names and their sensitivity tags"""
    # Regex to match TAG lines with DATA_SENSITIVITY
    tag_pattern = r'WITH\s+TAG\s*\([^)]*DATA_SENSITIVITY-\'(\w+)\'\)'
    
    tagged_columns = []
    lines = ddl.split('\n')
    
    for i, line in enumerate(lines):
        if 'WITH TAG' in line and 'DATA_SENSITIVITY' in line:
            match = re.search(tag_pattern, line)
            if match and i > 0:
                # Look at previous non-comment lines to find column name
                prev_idx = i - 1
                while prev_idx >= 0:
                    prev_line = lines[prev_idx].strip()
                    if prev_line and not prev_line.startswith('--'):
                        col_match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)', prev_line)
                        if col_match:
                            column_name = col_match.group(1)
                            sensitivity = match.group(1)
                            # Map the abbreviated sensitivity to full form
                            sensitivity_map = {
                                'CIF': 'Confidential Information',
                                'NSPII': 'Non-sensitive PII',
                                'SPII': 'Sensitive PII'
                            }
                            tagged_columns.append({
                                'Column Name': column_name,  
                                'Explanation': '',
                                'Data Sensitivity': sensitivity_map.get(sensitivity, sensitivity)
                            })
                            break
                    prev_idx -= 1
                    
    return pd.DataFrame(tagged_columns) if tagged_columns else None
