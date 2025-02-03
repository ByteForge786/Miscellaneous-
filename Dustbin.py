import chainlit as cl
# ... [previous imports remain the same]

@cl.on_message
async def handle_selections(message: cl.Message):
    """Handle all selections in a single message handler"""
    msg = message.content
    
    if msg.startswith("schema:"):
        schema = msg.replace("schema:", "")
        state.current_schema = schema
        state.analysis_complete = False
        state.analysis_df = None
        
        object_type_select = cl.Select(
            id="object_type",
            label="Select Object Type",
            values=["TABLE", "VIEW"]
        )
        await object_type_select.send()
        
    elif msg.startswith("type:"):
        obj_type = msg.replace("type:", "")
        state.current_object_type = obj_type
        state.analysis_complete = False
        state.analysis_df = None
        
        objects = await get_schema_objects(state.current_schema)
        object_list = objects["tables"] if obj_type == "TABLE" else objects["views"]
        
        object_select = cl.Select(
            id="object",
            label=f"Select {obj_type}",
            values=object_list
        )
        await object_select.send()
        
    elif msg.startswith("object:"):
        selected_object = msg.replace("object:", "")
        state.current_object = selected_object
        state.analysis_complete = False
        state.analysis_df = None
        
        ddl, samples = await get_ddl_and_samples(
            state.current_schema, 
            selected_object, 
            state.current_object_type
        )
        
        await cl.Message(content=f"```sql\n{ddl}\n```").send()
        
        analyze_action = cl.Action(
            name="analyze_structure",
            label="🔍 Analyze Structure"
        )
        await cl.Message(content="Click to analyze:", actions=[analyze_action]).send()

@cl.on_chat_start
async def start():
    schemas = await get_schema_list()
    schema_elements = [
        cl.Select(
            id="schema",
            label="Select Schema",
            values=[f"schema:{s}" for s in schemas]
        )
    ]
    await cl.Message(content="Welcome to DDL Analyzer 🔍", elements=schema_elements).send()

# ... [rest of the code remains the same]
