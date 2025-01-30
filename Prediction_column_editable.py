import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import time
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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

# Initialize model and tokenizer globally with caching
@st.cache_resource
def get_model_and_tokenizer():
    """Loads and caches the model and tokenizer globally"""
    logger.info("Loading model and tokenizer")
    MODEL_ID = "/data/ntracedevpkg/dev/scripts/nhancebot/flant5_sensitivity/AutoModelForSequenceClassification/flant5"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

def create_classification_prompt(column_name: str, explanation: str) -> str:
    """Create a prompt with reasoning for classification."""
    return (
        f"Classify this data attribute into one of these categories:\n"
        f"- Sensitive PII: user data that if made public can harm user through fraud or theft\n"
        f"- Non-sensitive PII: user data that can be safely made public without harm\n"
        f"- Non-person data: internal company data not related to personal information\n\n"
        f"Attribute Name: {column_name}\n"
        f"Description: {explanation}\n"
        f"Consider the privacy impact and potential for misuse. Classify this as:"
    )

def classify_sensitivity(texts_to_classify: List[str]) -> List[str]:
    """Function to classify a list of texts using the model in batch"""
    logger.info(f"Classifying {len(texts_to_classify)} columns for sensitivity")
    model, tokenizer = get_model_and_tokenizer()
    
    # Process all texts in a single batch
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidences, predicted_classes = torch.max(probs, dim=1)
    
    predicted_classes = predicted_classes.cpu().numpy()
    
    id2label = {0: "Sensitive PII", 1: "Non-sensitive PII", 2: "Non-person data"}
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]
    
    return predicted_labels

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

def get_ddl(schema: str, object_name: str, object_type: str) -> str:
    """Fetches DDL for specific object"""
    logger.info(f"Fetching DDL for {object_type} {schema}.{object_name}")
    
    try:
        ddl_query = f"SELECT GET_DDL('{object_type}', '{schema}.{object_name}')"
        ddl_df = get_data_sf(ddl_query)
        return ddl_df.iloc[0, 0]
    except Exception as e:
        logger.error(f"Error fetching DDL: {str(e)}")
        raise

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

            if selected_schema:
                # 2. Object Type Selection
                object_type = st.radio(
                    "2. Select Object Type",
                    options=["TABLE", "VIEW"],
                    help="Choose the type of object to analyze",
                    key="object_type"
                )

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

        except Exception as e:
            st.error("Error loading options. Please check your connection.")
            logger.error(f"Error in selection options: {str(e)}")
            return

    # Main content area
    if 'selected_schema' in locals() and 'selected_object' in locals() and selected_object:
        try:
            # Get DDL
            ddl = get_ddl(selected_schema, selected_object, object_type)
            
            # Display DDL
            st.subheader("📝 DDL Statement")
            with st.expander("View DDL", expanded=True):
                st.code(ddl, language='sql')
            
            # Analyze button
            if not st.session_state.analysis_complete and st.button("🔍 Analyze Structure"):
                with st.spinner("Analyzing structure and predicting sensitivity..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Generate prompt for LLM
                    prompt = f"""Analyze this DDL statement and provide an explanation for each column:
                    {ddl}
                    For each column, provide a clear, concise explanation of what the column represents.
                    Format as JSON: {{"column_name": "explanation of what this column represents"}}"""

                    # Get LLM analysis
                    analysis = eval(get_llm_response(prompt))
                    
                    # Create classification prompts for sensitivity prediction
                    classification_prompts = [
                        create_classification_prompt(col, explanation) 
                        for col, explanation in analysis.items()
                    ]
                    
                    # Get sensitivity predictions in batch
                    sensitivity_predictions = classify_sensitivity(classification_prompts)
                    
                    # Transform "Non-person data" to "Confidential Information"
                    transformed_predictions = [
                        "Confidential Information" if pred == "Non-person data" else pred 
                        for pred in sensitivity_predictions
                    ]
                    
                    # Convert to DataFrame
                    results_data = [
                        {
                            "Column Name": col,
                            "Explanation": explanation,
                            "Data Sensitivity": sensitivity
                        }
                        for (col, explanation), sensitivity in zip(analysis.items(), transformed_predictions)
                    ]
                    
                    st.session_state.df = pd.DataFrame(results_data)
                    st.session_state.analysis_complete = True
                    st.rerun()

            # Show results if analysis is complete
            if st.session_state.analysis_complete:
                st.subheader("📊 Analysis Results")
                
                # Predefined sensitivity options
                SENSITIVITY_OPTIONS = [
                    "Sensitive PII",
                    "Non-sensitive PII",
                    "Confidential Information",
                    "Licensed Data"
                ]
                
                edited_df = st.data_editor(
                    st.session_state.df,
                    use_container_width=True,
                    column_config={
                        "Column Name": st.column_config.TextColumn("Column Name", width="medium"),
                        "Explanation": st.column_config.TextColumn("Explanation", width="large"),
                        "Data Sensitivity": st.column_config.SelectboxColumn(
                            "Data Sensitivity",
                            width="medium",
                            help="Predicted data sensitivity classification",
                            options=SENSITIVITY_OPTIONS,
                            required=True
                        )
                    },
                    num_rows="fixed",
                    disabled=False,
                    key="editor"
                )
                
                # Standardize sensitivity values in the DataFrame
                if edited_df is not None:
                    edited_df['Data Sensitivity'] = edited_df['Data Sensitivity'].apply(
                        lambda x: next(
                            (option for option in SENSITIVITY_OPTIONS if option.lower() == str(x).lower()),
                            x  # Keep original if no match found
                        )
                    )
                st.session_state.df = edited_df
                
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
