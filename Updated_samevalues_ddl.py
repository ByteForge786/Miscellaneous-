import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import time
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ddl_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def get_schema_list() -> List[str]:
    """Fetches list of schemas - cached globally for 1 hour"""
    logger.info("Fetching schema list")
    try:
        query = """
        SELECT SCHEMA_NAME as name 
        FROM information_schema.schemata 
        WHERE SCHEMA_NAME NOT LIKE 'INFORMATION_SCHEMA%'
        ORDER BY SCHEMA_NAME
        """
        df = get_data_sf(query)
        schemas = df['name'].tolist()
        logger.info(f"Cached {len(schemas)} schemas globally")
        return schemas
    except Exception as e:
        logger.error(f"Error fetching schemas: {str(e)}")
        raise

@st.cache_data(ttl=3600)
def get_schema_objects(schema: str) -> Dict[str, List[str]]:
    """Fetches objects for a schema - cached globally for 1 hour"""
    logger.info(f"Fetching objects for schema: {schema}")
    try:
        # Get tables
        tables_query = f"""
        SELECT TABLE_NAME as name
        FROM information_schema.tables
        WHERE TABLE_SCHEMA = '{schema}'
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        tables_df = get_data_sf(tables_query)
        
        # Get views
        views_query = f"""
        SELECT TABLE_NAME as name
        FROM information_schema.views
        WHERE TABLE_SCHEMA = '{schema}'
        ORDER BY TABLE_NAME
        """
        views_df = get_data_sf(views_query)
        
        result = {
            "tables": tables_df['name'].tolist(),
            "views": views_df['name'].tolist()
        }
        logger.info(f"Cached objects for {schema}: {len(result['tables'])} tables, {len(result['views'])} views")
        return result
    except Exception as e:
        logger.error(f"Error fetching objects for {schema}: {str(e)}")
        raise

def get_ddl_and_samples(schema: str, object_name: str, object_type: str) -> Tuple[str, Dict[str, List[str]]]:
    """Fetches DDL and optionally sample values for specific object"""
    logger.info(f"Fetching DDL and samples for {object_type} {schema}.{object_name}")
    
    # First get column information - we need this for samples
    try:
        columns_query = f"""
        SELECT column_name
        FROM information_schema.columns 
        WHERE table_schema = '{schema}' 
        AND table_name = '{object_name}'
        ORDER BY ordinal_position
        """
        columns_df = get_data_sf(columns_query)
        if columns_df.empty:
            logger.warning(f"No columns found for {schema}.{object_name}")
            return ddl, {}
            
        columns = columns_df['column_name'].tolist()
    except Exception as e:
        logger.error(f"Error fetching column information: {str(e)}")
        raise

    # Get DDL
    try:
        ddl_query = f"SELECT GET_DDL('{object_type}', '{schema}.{object_name}')"
        ddl_df = get_data_sf(ddl_query)
        ddl = ddl_df.iloc[0, 0]
    except Exception as e:
        logger.error(f"Error fetching DDL: {str(e)}")
        raise  # DDL is essential, so we raise the error
        
    # Then try to get samples - this is optional
    try:
        # Single query to get samples
        non_null_conditions = [f"{col} IS NOT NULL" for col in columns]
        sample_query = f"""
        WITH sample_data AS (
            SELECT {', '.join(columns)}
            FROM {schema}.{object_name}
            WHERE {' OR '.join(non_null_conditions)}
        )
        SELECT DISTINCT {', '.join(columns)}
        FROM sample_data
        QUALIFY ROW_NUMBER() OVER (PARTITION BY 
            {', '.join(f'CASE WHEN {col} IS NOT NULL THEN {col} END' for col in columns)}
        ) <= 5
        """
        samples_df = get_data_sf(sample_query)
        
        # Process samples for each column
        samples_dict = {}
        ddl_lines = ddl.split('\n')
        processed_ddl = []
        current_line_index = 0
        
        for column in columns:
            if column in samples_df.columns:  # Check if column exists in samples
                values = samples_df[samples_df[column].notnull()][column].astype(str).unique().tolist()
                if values and not all(v.lower() in ['nan', 'none', 'null', ''] for v in values):
                    samples_dict[column] = values[:5]  # Ensure we take max 5 unique values
        
        # Process DDL with samples
        while current_line_index < len(ddl_lines):
            line = ddl_lines[current_line_index]
            processed_ddl.append(line)
            
            # Check if this line contains a column definition
            for column in samples_dict:
                if column in line and (',' in line or ')' in line):  # Column definition line
                    values = samples_dict[column]
                    # If any value > 50 chars, keep only one sample
                    if any(len(str(v)) > 50 for v in values):
                        sample_str = f"    -- Dummy Sample: {values[0]}"
                    else:
                        sample_str = f"    -- Dummy Samples: {', '.join(str(v) for v in values)}"
                    processed_ddl.append(sample_str)
                    break
                    
            current_line_index += 1
                    
        modified_ddl = '\n'.join(processed_ddl)
        logger.info(f"Successfully fetched samples for {schema}.{object_name}")
        return modified_ddl, samples_dict
        
    except Exception as e:
        logger.warning(f"Failed to fetch samples for {schema}.{object_name}: {str(e)}")
        # Return original DDL without samples if sample fetching fails
        return ddl, {}

def main():
    # Page config
    st.set_page_config(
        page_title="DDL Analyzer",
        page_icon="🔍",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Initialize session state for analysis if not done
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Sidebar for selections
    with st.sidebar:
        st.header("Object Selection")
        
        try:
            # 1. Schema Selection
            available_schemas = get_schema_list()
            if not available_schemas:
                st.error("No schemas available")
                return
                
            selected_schema = st.selectbox(
                "1. Select Schema",
                options=available_schemas,
                help="Choose a schema to explore",
                key="schema_selector"
            )
            
            # Reset analysis state when schema changes
            if 'previous_schema' not in st.session_state:
                st.session_state.previous_schema = selected_schema
            elif st.session_state.previous_schema != selected_schema:
                st.session_state.analysis_complete = False
                if 'df' in st.session_state:
                    del st.session_state.df
                st.session_state.previous_schema = selected_schema

            if selected_schema:
                # 2. Object Type Selection
                object_type = st.radio(
                    "2. Select Object Type",
                    options=["TABLE", "VIEW"],
                    help="Choose the type of object to analyze",
                    key="object_type"
                )
                
                # Reset analysis state when object type changes
                if 'previous_type' not in st.session_state:
                    st.session_state.previous_type = object_type
                elif st.session_state.previous_type != object_type:
                    st.session_state.analysis_complete = False
                    if 'df' in st.session_state:
                        del st.session_state.df
                    st.session_state.previous_type = object_type

                # 3. Object Selection
                schema_objects = get_schema_objects(selected_schema)
                object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
                
                if not object_list:
                    st.warning(f"No {object_type.lower()}s found in {selected_schema}")
                    selected_object = None
                else:
                    selected_object = st.selectbox(
                        f"3. Select {object_type}",
                        options=object_list,
                        help=f"Choose a {object_type.lower()} to analyze",
                        key="object_selector"
                    )
                    
                    # Reset analysis state when table selection changes
                    if 'previous_object' not in st.session_state:
                        st.session_state.previous_object = selected_object
                    elif st.session_state.previous_object != selected_object:
                        st.session_state.analysis_complete = False
                        if 'df' in st.session_state:
                            del st.session_state.df
                        st.session_state.previous_object = selected_object

        except Exception as e:
            st.error("Error loading options. Please check your connection.")
            logger.error(f"Error in selection options: {str(e)}")
            return

    # Main content area
    if 'selected_schema' in locals() and 'selected_object' in locals() and selected_object:
        try:
            # Get DDL and samples
            ddl, samples = get_ddl_and_samples(selected_schema, selected_object, object_type)
            
            # Display DDL with samples if available
            st.subheader("📝 DDL Statement")
            with st.expander("View DDL", expanded=True):
                st.code(ddl, language='sql')
                if samples:  # Only show samples section if samples were successfully fetched
                    st.subheader("Dummy Sample Values (from actual data)")
                    for column, values in samples.items():
                        st.write(f"**{column}**: {', '.join(str(v) for v in values)}")
            
            # Analyze button
            if not st.session_state.analysis_complete and st.button("🔍 Analyze Structure"):
                with st.spinner("Analyzing structure..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Generate prompt for LLM
                    prompt = f"""Analyze this DDL statement and provide a detailed classification of each column:
                    {ddl}
                    For each column, provide:
                    1. Classification (choose from: Identifier, Descriptive, Temporal, Numerical, Categorical, Status)
                    2. Brief explanation of the classification
                    3. Suggested data quality checks
                    Format as JSON: {{"column_name": {{"classification": "category", "explanation": "brief explanation", "data_quality_checks": ["check1", "check2"]}}}}"""

                    # Get LLM analysis
                    analysis = eval(get_llm_response(prompt))
                    
                    # Add sample values to analysis
                    for column in analysis:
                        if column in samples:
                            analysis[column]["sample_values"] = samples[column]
                    
                    # Convert to DataFrame
                    results_data = [
                        {
                            "Column": col,
                            "Classification": details["classification"],
                            "Explanation": details["explanation"],
                            "Sample Values": ", ".join(str(v) for v in details.get("sample_values", [])),
                            "Data Quality Checks": "\n".join(details["data_quality_checks"])
                        }
                        for col, details in analysis.items()
                    ]
                    
                    st.session_state.df = pd.DataFrame(results_data)
                    st.session_state.analysis_complete = True
                    st.rerun()

            # Show results if analysis is complete
            if st.session_state.analysis_complete:
                st.subheader("📊 Analysis Results")
                
                tab1, tab2 = st.tabs(["📋 Details", "📊 Summary"])
                
                with tab1:
                    edited_df = st.data_editor(
                        st.session_state.df,
                        use_container_width=True,
                        num_rows="fixed",
                        disabled=False,
                        key="editor"
                    )
                    st.session_state.df = edited_df
                
                with tab2:
                    st.subheader("Classification Distribution")
                    st.bar_chart(edited_df["Classification"].value_counts())
                
                st.info("💡 You can edit the values in the table above. The downloaded file will include your changes.")
                
                # Download option with edited dataframe
                st.download_button(
                    "📥 Download Analysis",
                    edited_df.to_csv(index=False),
                    f"ddl_analysis_{selected_schema}_{selected_object}_{datetime.now():%Y%m%d_%H%M}.csv",
                    "text/csv",
                    key="download"
                )

        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
