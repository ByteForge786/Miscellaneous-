import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import time
from typing import List, Dict, Tuple
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

# Constants
FEEDBACK_FILE = 'user_feedback.csv'
SENSITIVITY_OPTIONS = [
    "Sensitive PII",
    "Non-sensitive PII",
    "Confidential Information",
    "Licensed Data"
]

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_schema = None
    st.session_state.current_object_type = None
    st.session_state.current_object = None
    st.session_state.analysis_df = None
    st.session_state.tagged_df = None
    st.session_state.analysis_complete = False

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

def save_feedback(schema: str, table: str, feedback_df: pd.DataFrame):
    """Save or update feedback in CSV file"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_records = []
    
    for _, row in feedback_df.iterrows():
        new_records.append({
            'schema': schema,
            'table': table,
            'column_name': row['Column Name'],
            'explanation': row['Explanation'],
            'sensitivity': row['Data Sensitivity'],
            'timestamp': now
        })
    
    new_df = pd.DataFrame(new_records)
    
    if os.path.exists(FEEDBACK_FILE):
        existing_df = pd.read_csv(FEEDBACK_FILE)
        # Remove old entries for this schema/table
        existing_df = existing_df[~((existing_df['schema'] == schema) & 
                                  (existing_df['table'] == table))]
        # Append new records
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.to_csv(FEEDBACK_FILE, index=False)

def parse_ddl_tags(ddl: str) -> pd.DataFrame:
    """Parse DDL to extract column names and their sensitivity tags"""
    tagged_columns = []
    lines = ddl.split('\n')
    
    for line in lines:
        if 'WITH TAG' in line and 'DATA_SENSITIVITY' in line:
            # Extract column name from previous line
            prev_line = lines[lines.index(line) - 1]
            column_name = prev_line.strip().split(',')[0].strip()
            
            # Extract sensitivity value
            sensitivity = line.split('DATA_SENSITIVITY-')[-1].strip("'").strip(')').strip(',')
            if sensitivity == 'CIF':
                sensitivity = "Confidential Information"
            elif sensitivity == 'NSPII':
                sensitivity = "Non-sensitive PII"
            elif sensitivity == 'SPII':
                sensitivity = "Sensitive PII"
            
            tagged_columns.append({
                'Column Name': column_name,
                'Data Sensitivity': sensitivity,
                'Explanation': ''  # Empty explanation for tagged columns
            })
    
    return pd.DataFrame(tagged_columns) if tagged_columns else None

def display_editable_tables():
    """Display both tagged and analyzed tables with appropriate editing controls"""
    if st.session_state.analysis_df is None:
        return

    # Add download button with custom styling
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Combine both dataframes for download
        download_df = st.session_state.analysis_df.copy()
        if 'tagged_df' in st.session_state and st.session_state.tagged_df is not None:
            download_df = pd.concat([st.session_state.tagged_df, download_df], ignore_index=True)
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Results as CSV",
            data=csv,
            file_name=f"ddl_analysis_{st.session_state.current_schema}_{st.session_state.current_object}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Display tagged columns if they exist
    if 'tagged_df' in st.session_state and st.session_state.tagged_df is not None and not st.session_state.tagged_df.empty:
        st.markdown("### 📌 Tagged Columns (from DDL)")
        tagged_edited = st.data_editor(
            st.session_state.tagged_df,
            key=f"tagged_table_{st.session_state.current_schema}_{st.session_state.current_object}",
            column_config={
                "Column Name": st.column_config.TextColumn(
                    "Column Name",
                    width="medium",
                    disabled=True,
                ),
                "Data Sensitivity": st.column_config.TextColumn(
                    "Data Sensitivity",
                    width="medium",
                    disabled=True,
                ),
                "Explanation": st.column_config.TextColumn(
                    "Explanation",
                    width="large",
                ),
            },
            disabled=["Column Name", "Data Sensitivity"],  # Explicitly disable these columns
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
        )
        st.session_state.tagged_df = tagged_edited
        st.markdown("---")

    # Display analyzed columns
    st.markdown("### 🔍 Analyzed Columns (Model Predictions)")
    if not st.session_state.analysis_df.empty:
        analyzed_edited = st.data_editor(
            st.session_state.analysis_df,
            key=f"analyzed_table_{st.session_state.current_schema}_{st.session_state.current_object}",
            column_config={
                "Column Name": st.column_config.TextColumn(
                    "Column Name",
                    width="medium",
                    disabled=True,
                ),
                "Explanation": st.column_config.TextColumn(
                    "Explanation",
                    width="large",
                ),
                "Data Sensitivity": st.column_config.SelectboxColumn(
                    "Data Sensitivity",
                    width="medium",
                    options=SENSITIVITY_OPTIONS,
                )
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
        )
        st.session_state.analysis_df = analyzed_edited

    # Update the dataframe only if Execute is clicked
    if st.button("▶️ Execute", key="execute_button"):
        # Combine both dataframes for saving
        save_df = st.session_state.analysis_df.copy()
        if 'tagged_df' in st.session_state and st.session_state.tagged_df is not None:
            save_df = pd.concat([st.session_state.tagged_df, save_df], ignore_index=True)
        
        save_feedback(
            st.session_state.current_schema,
            st.session_state.current_object,
            save_df
        )
        st.success("✅ Feedback saved successfully!")
        st.balloons()

    # Display distribution chart with enhanced styling
    st.markdown("### 📊 Data Sensitivity Distribution")
    if st.session_state.analysis_df is not None or st.session_state.tagged_df is not None:
        # Combine both dataframes for distribution
        chart_df = st.session_state.analysis_df.copy()
        if 'tagged_df' in st.session_state and st.session_state.tagged_df is not None:
            chart_df = pd.concat([st.session_state.tagged_df, chart_df], ignore_index=True)
        
        sensitivity_counts = chart_df["Data Sensitivity"].value_counts()
        chart_data = pd.DataFrame({
            'Sensitivity': sensitivity_counts.index,
            'Count': sensitivity_counts.values
        })
        st.bar_chart(
            chart_data.set_index('Sensitivity'),
            use_container_width=True,
        )

def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    
    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Sidebar selections
    with st.sidebar:
        st.header("Object Selection")
        
        # Get schema list
        schemas = get_schema_list()  # Your existing function
        selected_schema = st.selectbox("1. Select Schema", schemas)
        
        # Check if schema changed
        if selected_schema != st.session_state.current_schema:
            st.session_state.current_schema = selected_schema
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
            st.session_state.tagged_df = None
        
        # Object type selection
        object_type = st.radio("2. Select Object Type", ["TABLE", "VIEW"])
        
        # Check if object type changed
        if object_type != st.session_state.current_object_type:
            st.session_state.current_object_type = object_type
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
            st.session_state.tagged_df = None
        
        # Get objects for selected schema
        schema_objects = get_schema_objects(selected_schema)  # Your existing function
        object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
        
        selected_object = st.selectbox(f"3. Select {object_type}", object_list)
        
        # Check if object changed
        if selected_object != st.session_state.current_object:
            st.session_state.current_object = selected_object
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
            st.session_state.tagged_df = None

    # Main content area
    if all([selected_schema, object_type, selected_object]):
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
            
            # Analysis section
            if not st.session_state.analysis_complete:
                if st.button("🔍 Analyze Structure"):
                    with st.spinner("Analyzing structure and predicting sensitivity..."):
                        # First parse DDL for tagged columns
                        st.session_state.tagged_df = parse_ddl_tags(ddl)
                        
                        # Get all column names from DDL for untagged analysis
                        all_columns = [line.strip().split(',')[0].strip() 
                                     for line in ddl.split('\n') 
                                     if line.strip() and not line.strip().startswith('--')
                                     and not line.strip().startswith(')')
                                     and '(' not in line]
                        
                        # Filter out tagged columns
                        tagged_columns = [] if st.session_state.tagged_df is None else st.session_state.tagged_df['Column Name'].tolist()
                        untagged_columns = [col for col in all_columns if col not in tagged_columns]
                        
                        # Only analyze untagged columns
                        if untagged_columns:
                            # Generate prompt for analysis
                            untagged_ddl = '\n'.join([line for line in ddl.split('\n') 
                                                     if any(col in line for col in untagged_columns)])
                            prompt = f"""Analyze this DDL statement and provide an explanation for each column:
                            {untagged_ddl}
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
                                    "Data Sensitivity": sensitivity
                                }
                                for (col, explanation), sensitivity in zip(analysis.items(), transformed_predictions)
                            ]
                            
                            st.session_state.analysis_df = pd.DataFrame(results_data)
                        else:
                            st.session_state.analysis_df = pd.DataFrame(columns=['Column Name', 'Explanation', 'Data Sensitivity'])
                        
                        st.session_state.analysis_complete = True
            
            # Display results and enable editing
            if st.session_state.analysis_complete:
                st.subheader("📊 Analysis Results")
                display_editable_tables()display_editable_table()

        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
