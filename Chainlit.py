import chainlit as cl
import pandas as pd
from datetime import datetime
import time
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import logging

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

# Global state
class State:
    def __init__(self):
        self.current_schema = None
        self.current_object_type = None
        self.current_object = None
        self.analysis_df = None
        self.analysis_complete = False

state = State()

async def get_model_and_tokenizer():
    """Loads and caches the model and tokenizer"""
    logger.info("Loading model and tokenizer")
    MODEL_ID = "/data/ntracedevpkg/dev/scripts/nhancebot/flant5_sensitivity/AutoModelForSequenceClassification/flant5"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

def create_classification_prompt(column_name: str, explanation: str) -> str:
    """Create a prompt for classification"""
    return (
        f"Classify this data attribute into one of these categories:\n"
        f"- Sensitive PII: user data that if made public can harm user through fraud or theft\n"
        f"- Non-sensitive PII: user data that can be safely made public without harm\n"
        f"- Non-person data: internal company data not related to personal information\n\n"
        f"Attribute Name: {column_name}\n"
        f"Description: {explanation}\n"
        f"Consider the privacy impact and potential for misuse. Classify this as:"
    )

async def classify_sensitivity(texts_to_classify: List[str]) -> List[str]:
    """Classify texts using the model"""
    model, tokenizer = await get_model_and_tokenizer()
    inputs = tokenizer(texts_to_classify, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    _, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = predicted_classes.cpu().numpy()
    
    id2label = {0: "Sensitive PII", 1: "Non-sensitive PII", 2: "Non-person data"}
    return [id2label[class_id] for class_id in predicted_classes]

def save_feedback(schema: str, table: str, feedback_df: pd.DataFrame):
    """Save feedback to CSV"""
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
        existing_df = existing_df[~((existing_df['schema'] == schema) & 
                                  (existing_df['table'] == table))]
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.to_csv(FEEDBACK_FILE, index=False)

def create_table_html(df: pd.DataFrame) -> str:
    """Create HTML for editable table"""
    html = """
    <style>
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-family: Inter, system-ui, sans-serif;
        }
        .analysis-table th {
            background: #f8f9fa;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #344767;
            border-bottom: 2px solid #eee;
        }
        .analysis-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        .analysis-table tr:hover {
            background: #f8f9fa;
        }
        .analysis-table input, .analysis-table select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .analysis-table input:focus, .analysis-table select:focus {
            outline: none;
            border-color: #2196f3;
            box-shadow: 0 0 0 2px rgba(33,150,243,0.2);
        }
        .column-name {
            font-weight: 500;
            color: #1a73e8;
        }
        .sensitivity-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .sensitive-pii { background: #ffebee; color: #d32f2f; }
        .non-sensitive-pii { background: #e8f5e9; color: #2e7d32; }
        .confidential { background: #fff3e0; color: #ef6c00; }
        .licensed { background: #e3f2fd; color: #1976d2; }
    </style>
    <table class="analysis-table">
        <thead>
            <tr>
                <th>Column Name</th>
                <th>Explanation</th>
                <th>Data Sensitivity</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.iterrows():
        sensitivity_class = {
            "Sensitive PII": "sensitive-pii",
            "Non-sensitive PII": "non-sensitive-pii",
            "Confidential Information": "confidential",
            "Licensed Data": "licensed"
        }.get(row['Data Sensitivity'], "")
        
        html += f"""
            <tr>
                <td class="column-name">{row['Column Name']}</td>
                <td>
                    <input type="text" 
                           value="{row['Explanation']}"
                           data-row="{idx}"
                           data-col="Explanation"
                           onchange="handleEdit(this)">
                </td>
                <td>
                    <div class="sensitivity-badge {sensitivity_class}">
                        <select data-row="{idx}" 
                                data-col="Data Sensitivity"
                                onchange="handleEdit(this)">
                            {"".join(f'<option value="{opt}" {"selected" if opt == row["Data Sensitivity"] else ""}'
                                   f'>{opt}</option>' for opt in SENSITIVITY_OPTIONS)}
                        </select>
                    </div>
                </td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    <script>
        function handleEdit(element) {
            const row = element.dataset.row;
            const col = element.dataset.col;
            const value = element.value;
            window.chainlitState = window.chainlitState || {};
            window.chainlitState.edits = window.chainlitState.edits || {};
            window.chainlitState.edits[`${row}_${col}`] = value;
        }
    </script>
    """
    return html

@cl.on_chat_start
async def start():
    """Initialize the app"""
    await cl.Message(content="Welcome to DDL Analyzer 🔍").send()
    schemas = await get_schema_list()  # Your existing function
    
    schema_select = cl.Select(
        id="schema_select",
        label="Select Schema",
        values=schemas,
        initial_value=schemas[0] if schemas else None
    )
    await schema_select.send()

@cl.on_select
async def handle_select(select: cl.Select):
    """Handle selection changes"""
    if select.id == "schema_select":
        state.current_schema = select.value
        state.analysis_complete = False
        state.analysis_df = None
        
        # Show object type selection
        object_type_select = cl.Select(
            id="object_type_select",
            label="Select Object Type",
            values=["TABLE", "VIEW"],
            initial_value="TABLE"
        )
        await object_type_select.send()
    
    elif select.id == "object_type_select":
        state.current_object_type = select.value
        state.analysis_complete = False
        state.analysis_df = None
        
        # Get and show objects for selected schema/type
        objects = await get_schema_objects(state.current_schema)  # Your existing function
        object_list = objects["tables"] if select.value == "TABLE" else objects["views"]
        
        object_select = cl.Select(
            id="object_select",
            label=f"Select {select.value}",
            values=object_list,
            initial_value=object_list[0] if object_list else None
        )
        await object_select.send()
    
    elif select.id == "object_select":
        state.current_object = select.value
        state.analysis_complete = False
        state.analysis_df = None
        
        # Show DDL
        ddl, samples = await get_ddl_and_samples(
            state.current_schema, 
            state.current_object, 
            state.current_object_type
        )
        
        await cl.Message(content=f"```sql\n{ddl}\n```").send()
        
        if samples:
            samples_msg = "**Sample Values:**\n"
            for column, values in samples.items():
                samples_msg += f"- **{column}**: {', '.join(str(v) for v in values)}\n"
            await cl.Message(content=samples_msg).send()
        
        # Show analyze button
        analyze_action = cl.Action(
            name="analyze_structure",
            label="🔍 Analyze Structure",
            description="Analyze the DDL structure"
        )
        await cl.Message(content="Click to analyze the structure:", actions=[analyze_action]).send()

@cl.action_callback("analyze_structure")
async def analyze_structure(action: cl.Action):
    """Handle structure analysis"""
    async with cl.Step("Analyzing structure...") as step:
        ddl, _ = await get_ddl_and_samples(
            state.current_schema, 
            state.current_object, 
            state.current_object_type
        )
        
        # Generate prompt and get analysis
        prompt = f"""Analyze this DDL statement and provide an explanation for each column:
        {ddl}
        For each column, provide a clear, concise explanation of what the column represents.
        Format as JSON: {{"column_name": "explanation of what this column represents"}}"""
        
        analysis = eval(await get_llm_response(prompt))  # Your existing function
        
        # Create classification prompts
        classification_prompts = [
            create_classification_prompt(col, explanation) 
            for col, explanation in analysis.items()
        ]
        
        # Get sensitivity predictions
        sensitivity_predictions = await classify_sensitivity(classification_prompts)
        
        # Transform predictions
        transformed_predictions = [
            "Confidential Information" if pred == "Non-person data" else pred 
            for pred in sensitivity_predictions
        ]
        
        # Prepare results
        results_data = [
            {
                "Column Name": col,
                "Explanation": explanation,
                "Data Sensitivity": sensitivity
            }
            for (col, explanation), sensitivity in zip(analysis.items(), transformed_predictions)
        ]
        
        state.analysis_df = pd.DataFrame(results_data)
        state.analysis_complete = True
        
        # Show editable table
        table_html = create_table_html(state.analysis_df)
        await cl.Message(content=table_html).send()
        
        # Add Execute and Download buttons
        actions = [
            cl.Action(name="execute", label="▶️ Execute", description="Save changes"),
            cl.Action(name="download", label="📥 Download CSV", description="Download as CSV")
        ]
        await cl.Message(content="Actions:", actions=actions).send()
        
        # Show sensitivity distribution
        sensitivity_counts = state.analysis_df["Data Sensitivity"].value_counts()
        
        # Create chart using HTML/CSS
        chart_html = create_chart_html(sensitivity_counts)  # You'll need to implement this
        await cl.Message(content=chart_html).send()

@cl.action_callback("execute")
async def handle_execute(action: cl.Action):
    """Handle execute button click"""
    if state.analysis_df is not None:
        # Update DataFrame with edits
        edits = await cl.get_state("edits") or {}
        for key, value in edits.items():
            row, col = key.split('_')
            state.analysis_df.at[int(row), col] = value
        
        # Save feedback
        save_feedback(state.current_schema, state.current_object, state.analysis_df)
        
        await cl.Message(content="✅ Feedback saved successfully!").send()

@cl.action_callback("download")
async def handle_download(action: cl.Action):
    """Handle download button click"""
    if state.analysis_df is not None:
        csv_data = state.analysis_df.to_csv(index=False)
        filename = f"ddl_analysis_{state.current_schema}_{state.current_object}_{datetime.now():%Y%m%d_%H%M}.csv"
        
        await cl.Message(
            content="Here's your CSV file:",
            files=[cl.File(name=filename, content=csv_data.encode())]
        ).send()

def create_chart_html(sensitivity_counts: pd.Series) -> str:
    """Create HTML for sensitivity distribution chart"""
    max_count = sensitivity_counts.max()
    chart_html = """
    <style>
        .chart-container {
            margin: 1rem 0;
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .chart-bar {
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            font-family: Inter, system-ui, sans-serif;
        }
        .chart-label {
            width: 150px;
            font-weight: 500;
            color: #344767;
        }
        .chart-bar-fill {
            height: 24px;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        .chart-value {
            margin-left: 1rem;
            color: #666;
        }
        .sensitive-pii-bar { background: #ffebee; }
        .non-sensitive-pii-bar { background: #e8f5e9; }
        .confidential-bar { background: #fff3e0; }
        .licensed-bar { background: #e3f2fd; }
    </style>
    <div class="chart-container">
        <h3>Data Sensitivity Distribution</h3>
    """
    
    for sensitivity, count in sensitivity_counts.items():
        bar_class = {
            "Sensitive PII": "sensitive-pii-bar",
            "Non-sensitive PII": "non-sensitive-pii-bar",
            "Confidential Information": "confidential-bar",
            "Licensed Data": "licensed-bar"
        }.get(sensitivity, "")
        
        percentage = (count / max_count) * 100
        
        chart_html += f"""
        <div class="chart-bar">
            <div class="chart-label">{sensitivity}</div>
            <div class="chart-bar-fill {bar_class}" 
                 style="width: {percentage}%"></div>
            <div class="chart-value">{count}</div>
        </div>
        """
    
    chart_html += "</div>"
    return chart_html

@cl.on_settings_update
async def handle_settings_update(settings):
    """Handle settings updates"""
    pass

if __name__ == "__main__":
    cl.run()
