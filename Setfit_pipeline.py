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

# Initialize model and classifier head globally with caching
@st.cache_resource
def load_model_and_head():
    """Loads and caches the model, classifier head and label mappings globally"""
    logger.info("Loading SetFit model and classifier head")
    try:
        MODEL_PATH = "/path/to/your/setfit/model"  # Replace with your model path
        
        # Load sentence transformer
        model = SentenceTransformer(MODEL_PATH)
        
        # Load the classifier head
        head_path = os.path.join(MODEL_PATH, "model_head.pkl")
        with open(head_path, 'rb') as f:
            head = pickle.load(f)
        
        # Load label mapping
        metadata_path = os.path.join(MODEL_PATH, "metadata.txt")
        with open(metadata_path, 'r') as f:
            for line in f:
                if "label_mapping" in line:
                    mapping_str = line.split(': ', 1)[1].strip()
                    label_mapping = eval(mapping_str)
                    break
        
        id_to_label = {v: k for k, v in label_mapping.items()}
        
        return model, head, id_to_label
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def create_classification_prompt(column_name: str, explanation: str) -> str:
    """Create a prompt for the column classification."""
    return (
        f"Column Name: {column_name}\n"
        f"Description: {explanation}\n"
        f"Consider the privacy impact and potential for misuse."
    )

def classify_sensitivity(texts_to_classify: List[str], schema: str, object_name: str, column_details: List[Dict]) -> List[str]:
    """Function to classify a list of texts using the SetFit model"""
    logger.info(f"Classifying {len(texts_to_classify)} columns for sensitivity for {schema}.{object_name}")
    model, head, id_to_label = load_model_and_head()
    
    try:
        # Get embeddings
        embeddings = model.encode(texts_to_classify)
        
        # Get predictions and probabilities
        predictions = head.predict(embeddings)
        
        # Convert numeric predictions to labels
        predicted_labels = [id_to_label[pred] for pred in predictions]
        
        return predicted_labels
        
    except Exception as e:
        # Log detailed information about the failure
        logger.error(f"Error in sensitivity prediction for {schema}.{object_name}: {str(e)}")
        logger.error("Failed prediction details:")
        for idx, details in enumerate(column_details):
            logger.error(f"Column: {details['column_name']}")
            logger.error(f"Text to classify: {texts_to_classify[idx]}")
            logger.error(f"Column explanation: {details['explanation']}")
        
        # Return empty predictions
        return ["" for _ in texts_to_classify]

[... keep your existing schema and DDL fetching functions unchanged ...]

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

    [... keep your existing sidebar code unchanged ...]

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
                    
                    try:
                        # Prepare column details for logging
                        column_details = [
                            {
                                "column_name": col,
                                "explanation": explanation
                            }
                            for col, explanation in analysis.items()
                        ]
                        
                        # Get sensitivity predictions with context
                        sensitivity_predictions = classify_sensitivity(
                            classification_prompts, 
                            selected_schema, 
                            selected_object,
                            column_details
                        )
                        
                        # Transform "Non-person data" to "Confidential Information" only for successful predictions
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
                        
                    except Exception as e:
                        logger.error(f"Error in sensitivity prediction process for {selected_schema}.{selected_object}: {str(e)}")
                        # Create DataFrame with empty sensitivity values
                        results_data = [
                            {
                                "Column Name": col,
                                "Explanation": explanation,
                                "Data Sensitivity": ""
                            }
                            for col, explanation in analysis.items()
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
