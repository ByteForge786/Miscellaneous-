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
