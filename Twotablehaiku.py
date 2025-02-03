import streamlit as st
import pandas as pd
from datetime import datetime
import time
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import re

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
    st.session_state.tagged_columns = None
    st.session_state.untagged_columns = None
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

def classify_sensitivity(texts_to_classify: list[str]) -> list[str]:
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
            return {
                'tagged_columns': filtered[filtered['column_name'].isin(filtered['column_name'][filtered['sensitivity'].notnull()])].to_dict('records'),
                'untagged_columns': filtered[filtered['column_name'].isin(filtered['column_name'][filtered['sensitivity'].isnull()])].to_dict('records')
            }
    return {'tagged_columns': [], 'untagged_columns': []}

def save_feedback(schema, table, tagged_columns, untagged_columns):
    """Save or update feedback in CSV file"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_records = []
    
    for column in tagged_columns:
        new_records.append({
            'schema': schema,
            'table': table,
            'column_name': column['Column Name'],
            'sensitivity': column['Data Sensitivity'],
            'timestamp': now
        })
    
    for column in untagged_columns:
        new_records.append({
            'schema': schema,
            'table': table,
            'column_name': column['Column Name'],
            'explanation': column['Explanation'],
            'sensitivity': column['Data Sensitivity'],
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
        /* Modern table styling */
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .custom-table thead {
            background: #f8f9fa;
        }
        
        .custom-table th {
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #344767;
            border-bottom: 2px solid #eee;
        }
        
        .custom-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .custom-table tr:last-child td {
            border-bottom: none;
        }
        
        .custom-table tr:hover {
            background: #f8f9fa;
        }
        
        /* Editable cells */
        .editable {
            position: relative;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .editable:hover {
            background: #f1f3f6;
        }
        
        .editable input, .editable select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .editable input:focus, .editable select:focus {
            outline: none;
            border-color: #2196f3;
            box-shadow: 0 0 0 2px rgba(33,150,243,0.2);
        }
        
        /* Column Name styling */
        .column-name {
            font-weight: 500;
            color: #1a73e8;
        }
        
        /* Sensitivity badges */
        .sensitivity-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .sensitive-pii {
            background: #ffebee;
            color: #d32f2f;
        }
        
        .non-sensitive-pii {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .confidential {
            background: #fff3e0;
            color: #ef6c00;
        }
        
        .licensed {
            background: #e3f2fd;
            color: #1976d2;
        }

        /* Execute button styling */
        .execute-button {
            background: #1a73e8;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            margin-top: 1rem;
            transition: background 0.2s;
        }
        
        .execute-button:hover {
            background: #1557b0;
        }

        /* Download button styling */
        .download-button {
            background: #34a853;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            margin: 1rem 0;
            transition: background 0.2s;
        }
        
        .download-button:hover {
            background: #2d8644;
        }
        </style>
    """, unsafe_allow_html=True)

def display_tagged_columns_table(tagged_columns):
    st.subheader("📋 Tagged Columns")
    
    if tagged_columns:
        tagged_df = pd.DataFrame(tagged_columns)
        
        # Custom styling for the tagged columns table
        styled_table = (
            tagged_df.style
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': 'td', 'props': [('padding', '12px 15px')]}
            ])
            .format({
                'Data Sensitivity': lambda x: f'<div class="sensitivity-badge {x.lower().replace(" ", "-")}">{x}</div>'
            }, escape=False)
        )
        
        st.write(styled_table, unsafe_allow_html=True)
    else:
        st.write("No tagged columns found in the DDL.")

def display_untagged_columns_table(untagged_columns):
    st.subheader("📋 Untagged Columns")
    
    if untagged_columns:
        untagged_df = pd.DataFrame(untagged_columns)
        
        # Custom styling for the untagged columns table
        styled_table = (
            untagged_df.style
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': 'td', 'props': [('padding', '12px 15px')]}
            ])
            .format({
                'Data Sensitivity': lambda x: f'<div class="sensitivity-badge {x.lower().replace(" ", "-") if x else "non-sensitive-pii"}">{x or "Non-sensitive PII"}</div>'
            }, escape=False)
        )
        
        # Create an editable data editor for the untagged columns
        edited_untagged_df = st.data_editor(
            untagged_df,
            key=f"untagged_table_{st.session_state.current_schema}_{st.session_state.current_object}",
            column_config={
                "Column Name": st.column_config.TextColumn("Column Name", disabled=True),
                "Data Type": st.column_config.TextColumn("Data Type", disabled=True),
                "Explanation": st.column_config.TextColumn("Explanation"),
                "Data Sensitivity": st.column_config.SelectboxColumn("Data Sensitivity", options=SENSITIVITY_OPTIONS)
            },
            use_container_width=True,
            num_rows="fixed"
        )
        
        # Update the untagged columns with the edited data
        if st.button("▶️ Execute", key=f"execute_untagged_{st.session_state.current_schema}_{st.session_state.current_object}"):
            st.session_state[f"untagged_columns_{st.session_state.current_schema}_{st.session_state.current_object}"] = edited_untagged_df.to_dict('records')
            save_feedback(st.session_state.current_schema, st.session_state.current_object, st.session_state[f"tagged_columns_{st.session_state.current_schema}_{st.session_state.current_object}"], st.session_state[f"untagged_columns_{st.session_state.current_schema}_{st.session_state.current_object}"])
            st.success("✅ Feedback saved successfully!")
            st.balloons()

def extract_tagged_columns(ddl):
    tagged_columns = []
    for line in ddl.split('\n'):
        if 'NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_SENSITIVITY' in line:
            column_name = re.search(r'"(\w+)"', line)
            if column_name:
                column_name = column_name.group(1)
                sensitivity_tag = re.search(r'NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_SENSITIVITY=\'(\w+)\'', line).group(1)
                tagged_columns.append({
                    'Column Name': column_name,
                    'Data Sensitivity': sensitivity_tag
                })
    return tagged_columns

def extract_untagged_columns(ddl):
    untagged_columns = []
    for line in ddl.split('\n'):
        if 'CREATE OR REPLACE' in line and 'COLUMN' in line:
            column_name = re.search(r'"(\w+)"', line)
            if column_name:
                column_name = column_name.group(1)
                data_type = re.search(r'\w+\s+(\w+)', line).group(1)
                untagged_columns.append({
                    'Column Name': column_name,
                    'Data Type': data_type,
                    'Explanation': '',
                    'Data Sensitivity': ''
                })
    return untagged_columns

def handle_table_updates(selected_schema, selected_object):
    if st.session_state.get(f"tagged_columns_{selected_schema}_{selected_object}") is None or st.session_state.get(f"untagged_columns_{selected_schema}_{selected_object}") is None:
        existing_feedback = load_existing_feedback(selected_schema, selected_object)
        st.session_state[f"tagged_columns_{selected_schema}_{selected_object}"] = existing_feedback['tagged_columns']
        st.session_state[f"untagged_columns_{selected_schema}_{selected_object}"] = existing_feedback['untagged_columns']
    
    # Handle updates to the tagged columns table
    if st.button("▶️ Execute", key=f"execute_tagged_{selected_schema}_{selected_object}"):
        if st.session_state[f"tagged_columns_{selected_schema}_{selected_object}"] != st.session_state[f"tagged_columns_{selected_schema}_{selected_object}"]:
            save_feedback(selected_schema, selected_object, st.session_state[f"tagged_columns_{selected_schema}_{selected_object}"], st.session_state[f"untagged_columns_{selected_schema}_{selected_object}"])
            st.success("✅ Feedback saved successfully!")
            st.balloons()
    
    # Handle updates to the untagged columns table
    if st.session_state.get(f"untagged_columns_{selected_schema}_{selected_object}") != st.session_state[f"untagged_columns_{selected_schema}_{selected_object}"]:
        st.session_state[f"untagged_columns_{selected_schema}_{selected_object}"] = st.session_state[f"untagged_columns_{selected_schema}_{selected_object}"]
        save_feedback(selected_schema, selected_object, st.session_state[f"tagged_columns_{selected_schema}_{selected_object}"], st.session_state[f"untagged_columns_{selected_schema}_{selected_object}"])
        st.success("✅ Feedback saved successfully!")
        st.balloons()

def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Sidebar selections
    with st.sidebar:
        st.header("Object Selection")
        
        schemas = get_schema_list()
        selected_schema = st.selectbox("1. Select Schema", schemas)
        
        if selected_schema != st.session_state.current_schema:
            st.session_state.current_schema = selected_schema
            st.session_state.analysis_complete = False
            st.session_state.tagged_columns = None
            st.session_state.untagged_columns = None
        
        object_type = st.radio("2. Select Object Type", ["TABLE", "VIEW"])
        
        if object_type != st.session_state.current_object_type:
            st.session_state.current_object_type = object_type
            st.session_state.analysis_complete = False
            st.session_state.tagged_columns = None
            st.session_state.untagged_columns = None
        
        schema_objects = get_schema_objects(selected_schema)
        object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
        
        selected_object = st.selectbox(f"3. Select {object_type}", object_list)
        
        if selected_object != st.session_state.current_object:
            st.session_state.current_object = selected_object
            st.session_state.analysis_complete = False
            st.session_state.tagged_columns = None
            st.session_state.untagged_columns = None
    
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
            
            # Check if the "Analyze" button has been clicked
            if st.button("🔍 Analyze Structure"):
                # Extract tagged and untagged columns
                tagged_columns = extract_tagged_columns(ddl)
                untagged_columns = extract_untagged_columns(ddl)
                
                # Display tagged columns table
                display_tagged_columns_table(tagged_columns)
                
                # Display untagged columns table
                display_untagged_columns_table(untagged_columns)
                
                # Handle user interactions and updates for both tables
                handle_table_updates(selected_schema, selected_object)
            else:
                st.session_state.analysis_complete = False

        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
