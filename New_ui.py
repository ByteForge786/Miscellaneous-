import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import time
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

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

def load_custom_css():
    """Load professional UI styling"""
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        
        /* Header styling */
        .stTitle {
            color: #1e3a8a;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f1f5f9;
            padding: 1.5rem 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 0.375rem;
            background-color: #2563eb;
            color: white;
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #1d4ed8;
            transform: translateY(-1px);
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.875rem;
            border-collapse: collapse;
            width: 100%;
        }
        
        .dataframe th {
            background-color: #f8fafc;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* Card styling */
        .css-1r6slb0 {
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #3b82f6;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state with proper typing"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete: bool = False
    if 'current_config' not in st.session_state:
        st.session_state.current_config: Optional[str] = None
    if 'edited_data' not in st.session_state:
        st.session_state.edited_data: Optional[pd.DataFrame] = None
    if 'ddl_content' not in st.session_state:
        st.session_state.ddl_content: Optional[str] = None

# Keep original model initialization
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
        return result
    except Exception as e:
        logger.error(f"Error fetching objects: {str(e)}")
        raise

def get_ddl_and_samples(schema: str, object_name: str, object_type: str) -> Tuple[str, Dict[str, List[str]]]:
    """Fetches DDL and sample values for specific object"""
    logger.info(f"Fetching DDL and samples for {object_type} {schema}.{object_name}")
    
    # First get DDL - this is required
    try:
        ddl_query = f"SELECT GET_DDL('{object_type}', '{schema}.{object_name}')"
        ddl_df = get_data_sf(ddl_query)
        ddl = ddl_df.iloc[0, 0]
    except Exception as e:
        logger.error(f"Error fetching DDL: {str(e)}")
        raise
        
    # Then try to get samples - this is optional
    try:
        sample_query = f"SELECT * FROM {schema}.{object_name} LIMIT 5"
        samples_df = get_data_sf(sample_query)
        
        # Process samples for each column
        samples_dict = {}
        ddl_lines = ddl.split('\n')
        processed_ddl = []
        current_line_index = 0
        
        while current_line_index < len(ddl_lines):
            line = ddl_lines[current_line_index]
            is_column_line = False
            
            for column in samples_df.columns:
                if column in line and (',' in line or ')' in line):
                    is_column_line = True
                    values = samples_df[column].tolist()
                    valid_values = [str(v) for v in values if pd.notna(v) and v is not None 
                            and not pd.isna(v) and not pd.isnull(v)
                            and str(v).strip().lower() not in ['none', 'nan', 'nat', '']
                            and str(v).strip() != '[]']
                    
                    processed_ddl.append(line)
                    
                    if valid_values:
                        if any(len(str(v)) > 50 for v in valid_values):
                            samples_dict[column] = [valid_values[0]]
                            sample_str = f"    -- Sample: {valid_values[0]}"
                            processed_ddl.append(sample_str)
                        else:
                            samples_dict[column] = valid_values[:5]
                            if valid_values[:5]:
                                sample_str = f"    -- Samples: {', '.join(valid_values[:5])}"
                                processed_ddl.append(sample_str)
                    break
            
            if not is_column_line:
                processed_ddl.append(line)
                
            current_line_index += 1
                    
        modified_ddl = '\n'.join(processed_ddl)
        return modified_ddl, samples_dict
        
    except Exception as e:
        logger.warning(f"Failed to fetch samples: {str(e)}")
        return ddl, {}

def save_user_feedback(data: pd.DataFrame, schema: str, object_name: str):
    """Save/update user feedback with configuration tracking"""
    try:
        feedback_file = "ddl_analyzer_feedback.csv"
        timestamp = datetime.now()
        
        # Add metadata
        data = data.copy()
        data['schema'] = schema
        data['object'] = object_name
        data['config_key'] = f"{schema}.{object_name}"
        data['last_updated'] = timestamp
        
        # Try to load existing feedback
        try:
            existing_feedback = pd.read_csv(feedback_file)
            # Remove old entries for same configuration
            existing_feedback = existing_feedback[
                existing_feedback['config_key'] != f"{schema}.{object_name}"
            ]
            # Append new data
            updated_feedback = pd.concat([existing_feedback, data])
        except FileNotFoundError:
            updated_feedback = data
            
        # Save updated feedback
        updated_feedback.to_csv(feedback_file, index=False)
        logger.info(f"Updated feedback for {schema}.{object_name}")
        
        return feedback_file
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise

def load_existing_feedback(schema: str, object_name: str) -> Optional[pd.DataFrame]:
    """Load existing feedback for schema.object if it exists"""
    try:
        feedback_file = "ddl_analyzer_feedback.csv"
        if os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)
            config_data = feedback_df[
                feedback_df['config_key'] == f"{schema}.{object_name}"
            ]
            if not config_data.empty:
                return config_data
    except Exception as e:
        logger.warning(f"Error loading existing feedback: {str(e)}")
    return None

def show_loader(text: str):
    """Display professional loading animation"""
    with st.spinner(f'🔄 {text}'):
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
            time.sleep(0.01)

def custom_data_editor(data: pd.DataFrame) -> pd.DataFrame:
    """Custom data editor with persistent state"""
    try:
        return st.data_editor(
            data,
            use_container_width=True,
            column_config={
                "Column Name": st.column_config.TextColumn(
                    "Column Name",
                    width="medium",
                    required=True,
                    disabled=True
                ),
                "Explanation": st.column_config.TextColumn(
                    "Explanation",
                    width="large"
                ),
                "Data Sensitivity": st.column_config.SelectboxColumn(
                    "Data Sensitivity",
                    width="medium",
                    options=[
                        "Sensitive PII",
                        "Non-sensitive PII",
                        "Confidential Information",
                        "Licensed Data"
                    ]
                ),
                "Sample Values": st.column_config.TextColumn(
                    "Sample Values",
                    width="large",
                    disabled=True
                )
            },
            hide_index=True,
            num_rows="fixed"
        )
    except Exception as e:
        logger.error(f"Error in data editor: {str(e)}")
        raise

def generate_alter_statements(data: pd.DataFrame, schema: str, object_name: str) -> List[str]:
    """Generate ALTER statements based on sensitivity classifications"""
    try:
        alter_statements = []
        for _, row in data.iterrows():
            comment = f"Sensitivity: {row['Data Sensitivity']} | Explanation: {row['Explanation']}"
            stmt = f"ALTER TABLE {schema}.{object_name} MODIFY COLUMN {row['Column Name']} COMMENT '{comment}';"
            alter_statements.append(stmt)
        return alter_statements
    except Exception as e:
        logger.error(f"Error generating ALTER statements: {str(e)}")
        raise

def main():
    try:
        # Page configuration
        st.set_page_config(
            page_title="DDL Analyzer Pro",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load custom CSS
        load_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        st.title("DDL Analyzer Pro")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            try:
                # Schema Selection
                available_schemas = get_schema_list()
                if not available_schemas:
                    st.error("No schemas available")
                    return
                    
                selected_schema = st.selectbox(
                    "Select Schema",
                    options=available_schemas,
                    help="Choose a schema to analyze"
                )
                
                # Reset state if schema changes
                current_config = f"{selected_schema}" if selected_schema else None
                if current_config != st.session_state.current_config:
                    st.session_state.current_config = current_config
                    st.session_state.analysis_complete = False
                    st.session_state.edited_data = None
                    st.session_state.ddl_content = None
                
                if selected_schema:
                    # Object Type Selection
                    object_type = st.radio(
                        "Select Object Type",
                        options=["TABLE", "VIEW"]
                    )
                    
                    # Object Selection
                    objects = get_schema_objects(selected_schema)
                    object_list = objects["tables"] if object_type == "TABLE" else objects["views"]
                    
                    if not object_list:
                        st.warning(f"No {object_type.lower()}s found in {selected_schema}")
                        selected_object = None
                    else:
                        selected_object = st.selectbox(
                            f"Select {object_type}",
                            options=object_list
                        )
                        
                        # Update config and reset if object changes
                        new_config = f"{selected_schema}.{selected_object}"
                        if new_config != st.session_state.current_config:
                            st.session_state.current_config = new_config
                            st.session_state.analysis_complete = False
                            st.session_state.edited_data = None
                            st.session_state.ddl_content = None
                
            except Exception as e:
                st.error("Error loading options. Please check your connection.")
                logger.error(f"Error in selection options: {str(e)}")
                return

        # Main content area
        if selected_schema and selected_object:
            try:
                # Check for existing feedback first
                existing_feedback = load_existing_feedback(selected_schema, selected_object)
                
                if existing_feedback is not None and not st.session_state.analysis_complete:
                    st.info(f"Found existing analysis for {selected_schema}.{selected_object}")
                    if st.button("Load Previous Analysis"):
                        st.session_state.edited_data = existing_feedback
                        st.session_state.analysis_complete = True
                        st.rerun()
                
                # Get DDL and samples
                ddl, samples = get_ddl_and_samples(selected_schema, selected_object, object_type)
                st.session_state.ddl_content = ddl
                
                # Display DDL
                with st.expander("📝 DDL Statement", expanded=True):
                    st.code(ddl, language='sql')
                    if samples:
                        st.subheader("Sample Values")
                        for column, values in samples.items():
                            st.write(f"**{column}**: {', '.join(str(v) for v in values)}")
                
                # Analysis section
                if not st.session_state.analysis_complete:
                    if st.button("Analyze Structure", use_container_width=True):
                        show_loader("Analyzing structure and predicting sensitivity...")
                        
                        # Generate prompt for analysis
                        prompt = f"""Analyze this DDL statement and provide an explanation for each column:
                        {ddl}
                        For each column, provide a clear, concise explanation of what the column represents.
                        Format as JSON: {{"column_name": "explanation of what this column represents"}}"""

                        # Get analysis
                        analysis = eval(get_llm_response(prompt))
                        
                        # Create classification prompts
                        classification_prompts = [
                            create_classification_prompt(col, explanation) 
                            for col, explanation in analysis.items()
                        ]
                        
                        # Get sensitivity predictions
                        sensitivity_predictions = classify_sensitivity(classification_prompts)
                        
                        # Transform predictions
                        transformed_predictions = [
                            "Confidential Information" if pred == "Non-person data" else pred 
                            for pred in sensitivity_predictions
                        ]
                        
                        # Prepare results data
                        results_data = [
                            {
                                "Column Name": col,
                                "Explanation": explanation,
                                "Data Sensitivity": sensitivity,
                                "Sample Values": ", ".join(str(v) for v in samples.get(col, []))
                                                if col in samples and samples[col] else ""
                            }
                            for (col, explanation), sensitivity in zip(analysis.items(), transformed_predictions)
                        ]
                        
                        st.session_state.edited_data = pd.DataFrame(results_data)
                        st.session_state.analysis_complete = True
                        st.rerun()
                
                # Results section
                if st.session_state.analysis_complete and st.session_state.edited_data is not None:
                    st.header("Analysis Results")
                    
                    # Info message
                    st.info("💡 Edit explanations and sensitivity classifications below, then click 'Execute' to generate ALTER statements and save changes.")
                    
                    # Editable table
                    edited_df = custom_data_editor(st.session_state.edited_data)
                    st.session_state.edited_data = edited_df
                    
                    # Execute button
                    if st.button("Execute Changes", use_container_width=True):
                        show_loader("Processing changes...")
                        
                        try:
                            # Generate ALTER statements
                            alter_statements = generate_alter_statements(
                                edited_df,
                                selected_schema,
                                selected_object
                            )
                            
                            # Save user feedback
                            feedback_file = save_user_feedback(
                                edited_df,
                                selected_schema,
                                selected_object
                            )
                            
                            # Display results
                            st.markdown("### Generated ALTER Statements")
                            st.code("\n".join(alter_statements), language="sql")
                            st.success("✅ Changes saved successfully!")
                            
                        except Exception as e:
                            st.error("Failed to process changes. Please try again.")
                            logger.error(f"Error executing changes: {str(e)}")
                    
                    # Visualization
                    if not edited_df.empty:
                        st.subheader("Data Sensitivity Distribution")
                        
                        # Distribution chart
                        sensitivity_counts = edited_df["Data Sensitivity"].value_counts()
                        st.bar_chart(sensitivity_counts)
                        
                        # Metrics
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Total Columns", len(edited_df))
                        with cols[1]:
                            sensitive_count = len(edited_df[edited_df['Data Sensitivity'] == 'Sensitive PII'])
                            st.metric("Sensitive PII", sensitive_count)
                        with cols[2]:
                            nonsensitive_count = len(edited_df[edited_df['Data Sensitivity'] == 'Non-sensitive PII'])
                            st.metric("Non-sensitive PII", nonsensitive_count)
                        with cols[3]:
                            confidential_count = len(edited_df[edited_df['Data Sensitivity'] == 'Confidential Information'])
                            st.metric("Confidential", confidential_count)
            
            except Exception as e:
                st.error("Error during analysis. Please try again.")
                logger.error(f"Analysis error: {str(e)}")
                if st.checkbox("Show error details"):
                    st.code(str(e))

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again or contact support.")

if __name__ == "__main__":
    main()
