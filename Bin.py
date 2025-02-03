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

# Initialize session state variables at the start of the app
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.analysis_complete = False
        st.session_state.current_selections = {
            'schema': None,
            'object_type': None,
            'object': None
        }
        st.session_state.analysis_results = None
        st.session_state.editing_data = None
        st.session_state.last_edited = None

def has_selection_changed(schema, obj_type, obj):
    if not all(v is not None for v in st.session_state.current_selections.values()):
        return True
    return (st.session_state.current_selections['schema'] != schema or
            st.session_state.current_selections['object_type'] != obj_type or
            st.session_state.current_selections['object'] != obj)

def update_selection_state(schema, obj_type, obj):
    if has_selection_changed(schema, obj_type, obj):
        st.session_state.current_selections = {
            'schema': schema,
            'object_type': obj_type,
            'object': obj
        }
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = None
        st.session_state.editing_data = None
        st.session_state.last_edited = None
        return True
    return False

def main():
    init_session_state()
    
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
                        st.session_state.analysis_results = existing_analysis
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
                        
                        st.session_state.analysis_results = pd.DataFrame(results_data)
                    
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

                if st.session_state.editing_data is None:
                    st.session_state.editing_data = st.session_state.analysis_results.copy()

                # Create the data editor with a key based on current state
                editor_key = f"editor_{st.session_state.current_selections['schema']}_{st.session_state.current_selections['object']}"
                
                # Create an editable dataframe
                edited_df = st.data_editor(
                    st.session_state.editing_data,
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
                            help="Data sensitivity classification",
                            options=SENSITIVITY_OPTIONS,
                            required=True
                        )
                    },
                    disabled=False,
                    hide_index=True,
                    key=editor_key
                )
                
                # Update editing data only if it has changed
                if not edited_df.equals(st.session_state.editing_data):
                    st.session_state.editing_data = edited_df.copy()
                    st.session_state.last_edited = datetime.now()
                
                # Execute button for saving feedback
                if st.button("▶️ Execute"):
                    save_feedback(
                        st.session_state.current_selections['schema'],
                        st.session_state.current_selections['object'],
                        st.session_state.editing_data
                    )
                    st.success("✅ Feedback saved successfully!")
                    st.balloons()
                
                # Add visualization of sensitivity distribution
                st.subheader("Data Sensitivity Distribution")
                sensitivity_counts = st.session_state.editing_data["Data Sensitivity"].value_counts()
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
