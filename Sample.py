import streamlit as st
import snowflake.connector
import pandas as pd

def get_snowflake_connection():
    try:
        env = "dev"
        property_values = fetch_properties(env)
        conn = snowflake.connector.connect(
            user=property_values['snowflake.dbUsername'],
            password=property_values['snowflake.dbPassword'],
            account=property_values['snowflake.account']
        )
        return conn
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_databases(conn):
    cur = conn.cursor()
    cur.execute("SHOW DATABASES")
    return [row[1] for row in cur.fetchall()]

def get_schemas(conn, database):
    cur = conn.cursor()
    cur.execute(f"SHOW SCHEMAS IN DATABASE {database}")
    return [row[1] for row in cur.fetchall()]

def get_objects(conn, database, schema, object_type):
    cur = conn.cursor()
    cur.execute(f"SHOW {object_type}S IN SCHEMA {database}.{schema}")
    return [row[1] for row in cur.fetchall()]

def get_ddl(conn, database, schema, object_name, object_type):
    cur = conn.cursor()
    cur.execute(f"SELECT GET_DDL('{object_type}', '{database}.{schema}.{object_name}')")
    return cur.fetchone()[0]

def main():
    st.title("Snowflake Object Explorer")
    
    conn = get_snowflake_connection()
    if not conn:
        return

    # Database selection
    databases = get_databases(conn)
    selected_db = st.selectbox("Select Database", databases)

    if selected_db:
        # Schema selection
        schemas = get_schemas(conn, selected_db)
        selected_schema = st.selectbox("Select Schema", schemas)

        if selected_schema:
            # Object type selection
            object_type = st.selectbox("Select Object Type", ["TABLE", "VIEW"])
            
            # Object selection
            objects = get_objects(conn, selected_db, selected_schema, object_type)
            selected_object = st.selectbox(f"Select {object_type}", objects)

            if selected_object:
                # Show DDL
                ddl = get_ddl(conn, selected_db, selected_schema, selected_object, object_type)
                st.code(ddl, language='sql')

if __name__ == "__main__":
    main()
