import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import time

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

# Global cache for schema list
@st.cache_data(ttl=3600, experimental_allow_widgets=True)
def get_schema_list(conn) -> list:
    """
    Fetches and caches list of schemas for all users
    Query runs once per hour globally
    """
    logger.info("Fetching schema list")
    try:
        query = """
        SELECT SCHEMA_NAME as name 
        FROM information_schema.schemata 
        WHERE SCHEMA_NAME NOT LIKE 'INFORMATION_SCHEMA%'
        ORDER BY SCHEMA_NAME
        """
        df = pd.read_sql(query, conn)
        schemas = df['name'].tolist()
        logger.info(f"Cached {len(schemas)} schemas for global use")
        return schemas
    except Exception as e:
        logger.error(f"Error fetching schemas: {str(e)}")
        raise

# Global cache for object types in a schema
@st.cache_data(ttl=3600, experimental_allow_widgets=True)
def get_schema_objects(conn, schema: str) -> dict:
    """
    Fetches and caches tables and views for given schema for all users
    Query runs once per hour per schema globally
    """
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
        tables_df = pd.read_sql(tables_query, conn)
        
        # Get views
        views_query = f"""
        SELECT TABLE_NAME as name
        FROM information_schema.views
        WHERE TABLE_SCHEMA = '{schema}'
        ORDER BY TABLE_NAME
        """
        views_df = pd.read_sql(views_query, conn)
        
        result = {
            "tables": tables_df['name'].tolist(),
            "views": views_df['name'].tolist()
        }
        logger.info(f"Cached {len(result['tables'])} tables and {len(result['views'])} views for {schema}")
        return result
    except Exception as e:
        logger.error(f"Error fetching objects for {schema}: {str(e)}")
        raise

def get_ddl(conn, schema: str, object_name: str, object_type: str) -> str:
    """
    Fetches DDL for specific object - not cached as it might change frequently
    """
    logger.info(f"Fetching DDL for {object_type} {schema}.{object_name}")
    try:
        query = f"SELECT GET_DDL('{object_type}', '{schema}.{object_name}')"
        df = pd.read_sql(query, conn)
        ddl = df.iloc[0, 0]
        return ddl
    except Exception as e:
        logger.error(f"Error fetching DDL: {str(e)}")
        raise

def create_ui(conn):
    st.set_page_config(
        page_title="DDL Analyzer",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Sidebar for selections
    with st.sidebar:
        st.header("Object Selection")
        
        # 1. Schema Selection
        try:
            available_schemas = get_schema_list(conn)
            selected_schema = st.selectbox(
                "1. Select Schema",
                options=available_schemas,
                help="Choose a schema to explore"
            )
        except Exception as e:
            st.error("Error loading schemas. Please check connection.")
            logger.error(f"Schema loading error: {str(e)}")
            return

        # 2. Object Type Selection
        if selected_schema:
            object_type = st.radio(
                "2. Select Object Type",
                options=["TABLE", "VIEW"],
                help="Choose the type of object to analyze"
            )

            # 3. Object Selection
            try:
                schema_objects = get_schema_objects(conn, selected_schema)
                object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
                
                if not object_list:
                    st.warning(f"No {object_type.lower()}s found in {selected_schema}")
                    selected_object = None
                else:
                    selected_object = st.selectbox(
                        f"3. Select {object_type}",
                        options=object_list,
                        help=f"Choose a {object_type.lower()} to analyze"
                    )
            except Exception as e:
                st.error(f"Error loading {object_type.lower()}s")
                logger.error(f"Object loading error: {str(e)}")
                return

    # Main content area
    if selected_schema and selected_object:
        try:
            # Get DDL
            ddl = get_ddl(conn, selected_schema, selected_object, object_type)
            
            # Display DDL
            st.subheader("📝 DDL Statement")
            with st.expander("View DDL", expanded=True):
                st.code(ddl, language='sql')
            
            # Analyze button
            if st.button("🔍 Analyze Structure"):
                with st.spinner("Analyzing structure..."):
                    # Progress bar for visual feedback
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Generate prompt
                    prompt = f"""Analyze this DDL statement and provide a detailed classification of each column:

{ddl}

For each column, provide:
1. Classification (choose from: Identifier, Descriptive, Temporal, Numerical, Categorical, Status)
2. Brief explanation of the classification
3. Suggested data quality checks

Format as JSON: {{"column_name": {{"classification": "category", "explanation": "brief explanation", "data_quality_checks": ["check1", "check2"]}}}}"""

                    # Get LLM response
                    analysis = eval(get_llm_response(prompt))
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Convert to DataFrame
                    results_data = [
                        {
                            "Column": col,
                            "Classification": details["classification"],
                            "Explanation": details["explanation"],
                            "Data Quality Checks": "\n".join(details["data_quality_checks"])
                        }
                        for col, details in analysis.items()
                    ]
                    
                    df = pd.DataFrame(results_data)
                    
                    # Display in tabs
                    tab1, tab2 = st.tabs(["📋 Details", "📊 Summary"])
                    
                    with tab1:
                        st.dataframe(df, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Classification Distribution")
                        st.bar_chart(df["Classification"].value_counts())
                    
                    # Download option
                    st.download_button(
                        "📥 Download Analysis",
                        df.to_csv(index=False),
                        f"ddl_analysis_{selected_schema}_{selected_object}_{datetime.now():%Y%m%d_%H%M}.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    # Assuming conn is passed from your existing connection
    create_ui(conn)
