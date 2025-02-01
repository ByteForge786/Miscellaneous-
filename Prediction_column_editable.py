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

def get_ddl_and_samples(schema: str, object_name: str, object_type: str) -> Tuple[str, Dict[str, List[str]]]:
    """Fetches DDL and optionally sample values for specific object"""
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
        sample_query = f"""
        SELECT *
        FROM {schema}.{object_name}
        LIMIT 5
        """
        samples_df = get_data_sf(sample_query)
        
        # Process samples for each column
        samples_dict = {}
        ddl_lines = ddl.split('\n')
        processed_ddl = []
        current_line_index = 0
        
        # Process each line of DDL
        while current_line_index < len(ddl_lines):
            line = ddl_lines[current_line_index]
            is_column_line = False
            
            # Check if this line contains a column definition
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
        logger.info(f"Successfully fetched samples for {schema}.{object_name}")
        return modified_ddl, samples_dict
        
    except Exception as e:
        logger.warning(f"Failed to fetch samples for {schema}.{object_name}: {str(e)}")
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

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'editor_key' not in st.session_state:
        st.session_state.editor_key = 0

    # Sidebar for selections
    with st.sidebar:
        st.header("Object Selection")
        
        try:
            # Schema Selection
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
                if 'temp_df' in st.session_state:
                    del st.session_state.temp_df
                st.session_state.previous_schema = selected_schema

            if selected_schema:
                # Object Type Selection
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
                    if 'temp_df' in st.session_state:
                        del st.session_state.temp_df
                    st.session_state.previous_type = object_type

                # Object Selection
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
                    
                    # Reset analysis state when object changes
                    if 'previous_object' not in st.session_state:
                        st.session_state.previous_object = selected_object
                    elif st.session_state.previous_object != selected_object:
                        st.session_state.analysis_complete = False
                        if 'df' in st.session_state:
                            del st.session_state.df
                        if 'temp_df' in st.session_state:
                            del st.session_state.temp_df
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
            
            # Display DDL
            st.subheader("📝 DDL Statement")
            with st.expander("View DDL", expanded=True):
                st.code(ddl, language='sql')
                if samples:
                    st.subheader("Sample Values")
                    for column, values in samples.items():
                        st.write(f"**{column}**: {', '.join(str(v) for v in values)}")
            
            # Analyze button
            if not st.session_state.analysis_complete and st.button("🔍 Analyze Structure"):
                with st.spinner("Analyzing structure and predicting sensitivity..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
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
                    
                    st.session_state.df = pd.DataFrame(results_data)
                    st.session_state.analysis_complete = True
                    st.rerun()

            # Show results if analysis is complete
            if st.session_state.analysis_complete:
                st.subheader("📊 Analysis Results")
                
                # Add a submit button for changes
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("💡 Make all your changes in the table below, then click 'Save Changes' to update.")
                
                with col2:
                    submit_changes = st.button("💾 Save Changes", key="save_changes")
                
                # Predefined sensitivity options
                SENSITIVITY_OPTIONS = [
                    "Sensitive PII",
                    "Non-sensitive PII",
                    "Confidential Information",
                    "Licensed Data"
                ]
                
                # Initialize temporary state for edits if not exists
                if 'temp_df' not in st.session_state:
                    st.session_state.temp_df = st.session_state.df.copy()
                
                # Create an editable dataframe with batch saves
                edited_df = st.data_editor(
                    st.session_state.temp_df,  # Use temporary DataFrame for edits
                    use_container_width=True,
                    column_config={
                        "Column Name": st.column_config.TextColumn(
                            "Column Name",
                            width="medium",
                            required=True
                        ),
                        "Explanation": st.column_config.TextColumn(
                            "Explanation",
                            width="large"
                        ),
                        "Data Sensitivity": st.column_config.SelectboxColumn(
                            "Data Sensitivity",
                            width="medium",
                            help="Data sensitivity classification",
                            options=SENSITIVITY_OPTIONS,
                            required=True
                        ),
                        "Sample Values": st.column_config.TextColumn(
                            "Sample Values",
                            width="large",
                            help="Sample values from the data"
                        )
                    },
                    num_rows="fixed",
                    disabled=False,
                    key="editor"
                )
                
                # Update temporary state with current edits
                st.session_state.temp_df = edited_df
                
                # Only update final state if save button is clicked
                if submit_changes:
                    # Standardize sensitivity values in the DataFrame
                    if edited_df is not None:
                        edited_df['Data Sensitivity'] = edited_df['Data Sensitivity'].apply(
                            lambda x: next(
                                (option for option in SENSITIVITY_OPTIONS if option.lower() == str(x).lower()),
                                x  # Keep original if no match found
                            )
                        )
                    st.session_state.df = edited_df.copy()  # Save to final state
                    st.success("✅ Changes saved successfully!")
                    st.balloons()
                
                # Add visualization of sensitivity distribution using the temporary state
                st.subheader("Data Sensitivity Distribution")
                sensitivity_counts = st.session_state.temp_df["Data Sensitivity"].value_counts()
                st.bar_chart(sensitivity_counts)
                
                # Download option with saved state
                st.download_button(
                    "📥 Download Analysis",
                    st.session_state.df.to_csv(index=False),  # Use saved state for download
                    f"ddl_analysis_{selected_schema}_{selected_object}_{datetime.now():%Y%m%d_%H%M}.csv",
                    "text/csv",
                    key="download"
                )

        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
