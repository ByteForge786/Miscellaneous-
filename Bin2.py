import streamlit as st
import pandas as pd
from datetime import datetime
import time
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ddl_analyzer.log'), logging.StreamHandler()]
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
    st.session_state.editing_enabled = False
    st.session_state.edited_cells = {}
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

def load_existing_feedback(schema, table):
    """Load existing feedback from CSV file"""
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        filtered = df[(df['schema'] == schema) & (df['table'] == table)]
        if not filtered.empty:
            return pd.DataFrame({
                'Column Name': filtered['column_name'],
                'Explanation': filtered['explanation'],
                'Data Sensitivity': filtered['sensitivity']
            })
    return None

def save_feedback(schema, table, feedback_df):
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

# Custom styling for the app
def apply_custom_css():
    st.markdown("""
        <style>
        /* Make the table more compact */
        .stDataEditor {
            font-size: 0.9rem;
        }
        .stDataEditor td {
            padding: 0.5rem !important;
        }
        /* Style the download button */
        .stDownloadButton {
            margin-bottom: 1rem;
        }
        /* Adjust column widths */
        [data-testid="column"] {
            width: calc(33.3333% - 1rem) !important;
            margin: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

def display_editable_table():
    """Display the analysis results in a compact table format with download option"""
    if st.session_state.analysis_df is None:
        return

    # Add download button
    df_download = st.session_state.analysis_df.copy()
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"ddl_analysis_{st.session_state.current_schema}_{st.session_state.current_object}_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    # Create a more compact table using st.data_editor
    edited_df = st.data_editor(
        st.session_state.analysis_df,
        key=f"table_{st.session_state.current_schema}_{st.session_state.current_object}",
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
        num_rows="fixed"
    )

    # Update the dataframe if changes were made
    if not edited_df.equals(st.session_state.analysis_df):
        st.session_state.analysis_df = edited_df.copy()

def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
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
            st.session_state.edited_cells = {}
        
        # Object type selection
        object_type = st.radio("2. Select Object Type", ["TABLE", "VIEW"])
        
        # Check if object type changed
        if object_type != st.session_state.current_object_type:
            st.session_state.current_object_type = object_type
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
            st.session_state.edited_cells = {}
        
        # Get objects for selected schema
        schema_objects = get_schema_objects(selected_schema)  # Your existing function
        object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
        
        selected_object = st.selectbox(f"3. Select {object_type}", object_list)
        
        # Check if object changed
        if selected_object != st.session_state.current_object:
            st.session_state.current_object = selected_object
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
            st.session_state.edited_cells = {}

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
                        # Check for existing analysis first
                        existing_analysis = load_existing_feedback(selected_schema, selected_object)
                        
                        if existing_analysis is not None:
                            st.session_state.analysis_df = existing_analysis
                        else:
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
                                    "Data Sensitivity": sensitivity
                                }
                                for (col, explanation), sensitivity in zip(analysis.items(), transformed_predictions)
                            ]
                            
                            st.session_state.analysis_df = pd.DataFrame(results_data)
                        
                        st.session_state.analysis_complete = True
            
            # Display results and enable editing
            if st.session_state.analysis_complete:
                st.subheader("📊 Analysis Results")
                
                # Display editable table
                display_editable_table()
                
                # Execute button
                if st.button("▶️ Execute"):
                    save_feedback(
                        selected_schema,
                        selected_object,
                        st.session_state.analysis_df
                    )
                    st.success("✅ Feedback saved successfully!")
                    st.balloons()
                
                # Display distribution chart
                st.subheader("Data Sensitivity Distribution")
                if st.session_state.analysis_df is not None:
                    sensitivity_counts = st.session_state.analysis_df["Data Sensitivity"].value_counts()
                    st.bar_chart(sensitivity_counts)

        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
