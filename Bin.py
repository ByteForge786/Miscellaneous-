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

def load_existing_feedback():
    """Load existing feedback from CSV file"""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=['schema', 'table', 'column_name', 'explanation', 'sensitivity', 'timestamp'])

def save_feedback(schema: str, table: str, feedback_df: pd.DataFrame):
    """Save or update feedback in CSV file"""
    existing_df = load_existing_feedback()
    
    # Create new feedback records
    new_records = []
    for _, row in feedback_df.iterrows():
        new_record = {
            'schema': schema,
            'table': table,
            'column_name': row['Column Name'],
            'explanation': row['Explanation'],
            'sensitivity': row['Data Sensitivity'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        new_records.append(new_record)
    
    # Remove existing entries for this schema/table combination
    existing_df = existing_df[~((existing_df['schema'] == schema) & 
                              (existing_df['table'] == table))]
    
    # Append new records
    updated_df = pd.concat([existing_df, pd.DataFrame(new_records)], ignore_index=True)
    updated_df.to_csv(FEEDBACK_FILE, index=False)

def get_existing_analysis(schema: str, table: str):
    """Get existing analysis for a schema/table combination"""
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        filtered_df = feedback_df[
            (feedback_df['schema'] == schema) & 
            (feedback_df['table'] == table)
        ]
        if not filtered_df.empty:
            return pd.DataFrame({
                'Column Name': filtered_df['column_name'],
                'Explanation': filtered_df['explanation'],
                'Data Sensitivity': filtered_df['sensitivity']
            })
    return None

[Previous model and tokenizer functions remain the same...]

def main():
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
    if 'current_selections' not in st.session_state:
        st.session_state.current_selections = {
            'schema': None,
            'object_type': None,
            'object': None
        }

    # Function to check if selections have changed
    def has_selections_changed(schema, obj_type, obj):
        current = st.session_state.current_selections
        if (current['schema'] != schema or 
            current['object_type'] != obj_type or 
            current['object'] != obj):
            return True
        return False

    def update_selections(schema, obj_type, obj):
        st.session_state.current_selections = {
            'schema': schema,
            'object_type': obj_type,
            'object': obj
        }
        # Only reset analysis state when selections change
        st.session_state.analysis_complete = False
        if 'df' in st.session_state:
            del st.session_state.df

    # Sidebar for selections
    with st.sidebar:
        [Previous schema and object selection code remains the same...]

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
            
            # Check for existing analysis
            existing_analysis = get_existing_analysis(selected_schema, selected_object)
            
            # Analyze button
            if not st.session_state.analysis_complete and st.button("🔍 Analyze Structure"):
                with st.spinner("Analyzing structure and predicting sensitivity..."):
                    if existing_analysis is not None:
                        st.session_state.df = existing_analysis
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
                        
                        st.session_state.df = pd.DataFrame(results_data)
                    
                    st.session_state.analysis_complete = True

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
                
                # Initialize editor data only once when analysis is complete
                if 'editor_data' not in st.session_state:
                    st.session_state.editor_data = st.session_state.df.copy()
                
                # Create an editable dataframe without callback
                edited_df = st.data_editor(
                    data=st.session_state.editor_data,
                    use_container_width=True,
                    column_config={
                        "Column Name": st.column_config.TextColumn(
                            "Column Name",
                            width="medium",
                            required=True,
                            disabled=True  # Make column name non-editable
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
                        )
                    },
                    hide_index=True,
                    num_rows="fixed",
                    disabled=False,
                    key=f"editor_{selected_schema}_{selected_object}"  # Unique key per table
                )
                
                # Update session state with edited data
                st.session_state.editor_data = edited_df
                
                # Execute button for saving feedback
                if st.button("▶️ Execute"):
                    save_feedback(selected_schema, selected_object, st.session_state.editor_data)
                    st.success("✅ Feedback saved successfully!")
                    st.balloons()
                
                # Add visualization of sensitivity distribution
                st.subheader("Data Sensitivity Distribution")
                sensitivity_counts = st.session_state.editor_data["Data Sensitivity"].value_counts()
                st.bar_chart(sensitivity_counts)
                
                # Execute button for saving feedback
                if st.button("▶️ Execute"):
                    save_feedback(selected_schema, selected_object, st.session_state.editor_data)
                    st.success("✅ Feedback saved successfully!")
                    st.balloons()
                
                # Add visualization of sensitivity distribution
                st.subheader("Data Sensitivity Distribution")
                sensitivity_counts = st.session_state.editor_data["Data Sensitivity"].value_counts()
                st.bar_chart(sensitivity_counts)

        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()




















import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
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

# Initialize model and tokenizer globally with caching
@st.cache_resource
def load_model_and_head():
    """Loads and caches the model and classifier head globally"""
    logger.info("Loading model and head")
    model_path = "/path/to/your/setfit/model"  # Update this path
    try:
        # Load sentence transformer
        model = SentenceTransformer(model_path)
        
        # Load the classifier head
        head_path = os.path.join(model_path, "model_head.pkl")
        with open(head_path, 'rb') as f:
            head = pickle.load(f)
            
        # Load label mapping
        metadata_path = os.path.join(model_path, "metadata.txt")
        with open(metadata_path, 'r') as f:
            for line in f:
                if "label_mapping" in line:
                    mapping_str = line.split(': ', 1)[1].strip()
                    label_mapping = eval(mapping_str)
                    break
        id_to_label = {v: k for k, v in label_mapping.items()}
        
        logger.info("Model, head and mappings loaded successfully")
        return model, head, id_to_label
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def create_classification_prompt(column_name: str, explanation: str) -> str:
    """Create a prompt with reasoning for classification."""
    return (
        f"Attribute Name: {column_name}\n"
        f"Description: {explanation}\n"
        f"Consider the privacy impact and potential for misuse."
    )

def predict_texts(model, head, texts, id_to_label):
    """Make predictions for texts"""
    # Get embeddings
    embeddings = model.encode(texts)
    
    # Get predictions and probabilities
    probabilities = head.predict_proba(embeddings)
    predictions = head.predict(embeddings)
    
    # Convert numeric predictions to labels
    predicted_labels = [id_to_label[pred] for pred in predictions]
    confidences = np.max(probabilities, axis=1)
    
    return predicted_labels, confidences

def classify_sensitivity(texts_to_classify: List[str]) -> List[str]:
    """Function to classify a list of texts using SetFit model"""
    logger.info(f"Classifying {len(texts_to_classify)} columns for sensitivity")
    model, head, id_to_label = load_model_and_head()
    
    # Use the original predict_texts function
    predicted_labels, _ = predict_texts(model, head, texts_to_classify, id_to_label)
    
    return predicted_labels

[... keep all your existing schema and DDL fetching functions unchanged ...]

def main():
    [... keep all your existing main function code until the analyze button ...]

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
                    
                    try:
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
                        
                        # Ensure predictions match allowed options
                        SENSITIVITY_OPTIONS = [
                            "Sensitive PII",
                            "Non-sensitive PII",
                            "Confidential Information",
                            "Licensed Data"
                        ]
                        
                        # Default to "Confidential Information" if prediction is not in allowed options
                        validated_predictions = [
                            pred if pred in SENSITIVITY_OPTIONS else "Confidential Information"
                            for pred in transformed_predictions
                        ]
                        
                        # Convert to DataFrame
                        results_data = [
                            {
                                "Column Name": col,
                                "Explanation": explanation,
                                "Data Sensitivity": sensitivity
                            }
                            for (col, explanation), sensitivity in zip(analysis.items(), validated_predictions)
                        ]
                        
                        st.session_state.df = pd.DataFrame(results_data)
                        st.session_state.analysis_complete = True
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Error in sensitivity prediction: {str(e)}")
                        # Fallback to safe default if prediction fails
                        results_data = [
                            {
                                "Column Name": col,
                                "Explanation": explanation,
                                "Data Sensitivity": "Confidential Information"  # Safe default
                            }
                            for col, explanation in analysis.items()
                        ]
                        st.session_state.df = pd.DataFrame(results_data)
                        st.session_state.analysis_complete = True
                        st.rerun()

            [... keep the rest of your code unchanged ...]

if __name__ == "__main__":
    main()
