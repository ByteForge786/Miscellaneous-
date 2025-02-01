import snowflake.connector
from contextlib import contextmanager
import streamlit as st

class SnowflakeConnectionManager:
    def __init__(self):
        self._connection = None
        
    @property
    def connection(self):
        if self._connection is None or self._connection.is_closed():
            self._connection = snowflake.connector.connect(
                user=property_values["snowflake.user"],
                password=property_values["snowflake.password"],
                account=property_values["snowflake.account"],
                warehouse=property_values["snowflake.warehouse"],
                database=property_values["snowflake.database"],
                schema=property_values["snowflake.schema"]
            )
        return self._connection
    
    def close(self):
        if self._connection and not self._connection.is_closed():
            self._connection.close()
            self._connection = None

@st.cache_resource
def get_connection_manager():
    return SnowflakeConnectionManager()

def get_data_sf(query, timeout=50000):
    result_container = []
    conn_manager = get_connection_manager()
    
    try:
        with conn_manager.connection.cursor() as cur:
            logging.info("Executing query...")
            cur.execute(query)
            result_container = cur.fetch_pandas_all()
            logging.info("Query executed successfully")
            
    except snowflake.connector.ProgrammingError as e:
        logging.error("Snowflake error %s", str(e))
        result_container = f"snowflake error: {str(e)}"
        raise HTTPException(status_code=500, detail=f"Snowflake error: {str(e)}")
        
    except Exception as ex:
        logging.error("Snowflake error %s", str(ex))
        result_container = f"snowflake error: {str(ex)}"
        raise HTTPException(status_code=500, detail=f"Error: {str(ex)}")
        
    return result_container

# Add cleanup for Streamlit shutdown
def cleanup():
    conn_manager = get_connection_manager()
    conn_manager.close()

# Register cleanup handler
st.session_state.on_close = cleanup
