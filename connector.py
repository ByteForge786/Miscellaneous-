import snowflake.connector
import streamlit as st
import logging
import time
from datetime import datetime
import queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('snowflake_connection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PoolTimeoutError(Exception):
    pass

class DynamicSnowflakePool:
    def __init__(self, 
                 initial_pool_size=5,
                 min_pool_size=3,
                 max_pool_size=50,
                 connection_timeout=30):
        
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._current_pool_size = initial_pool_size
        self.connection_timeout = connection_timeout
        
        # Use queue for free connections
        self._free_connections = queue.Queue()
        self._active_connections = {}
        
        # Initialize pool
        self._initialize_pool()
        logger.info(f"Initialized pool with size: {self._current_pool_size}")

    def _create_connection(self):
        return snowflake.connector.connect(
            user=property_values["snowflake.user"],
            password=property_values["snowflake.password"],
            account=property_values["snowflake.account"],
            warehouse=property_values["snowflake.warehouse"],
            database=property_values["snowflake.database"],
            schema=property_values["snowflake.schema"]
        )

    def _initialize_pool(self):
        for _ in range(self._current_pool_size):
            try:
                conn = self._create_connection()
                self._free_connections.put(conn)
            except Exception as e:
                logger.error(f"Error creating initial connection: {str(e)}")

    def _is_connection_valid(self, connection):
        try:
            # Quick validation query
            connection.cursor().execute("SELECT 1")
            return True
        except:
            return False

    def get_connection(self):
        start_time = datetime.now()
        wait_count = 0  # Track how many times we've waited for a free connection
        
        while True:
            # PRIORITY 1: Try to get a free connection
            try:
                connection = self._free_connections.get_nowait()
                if self._is_connection_valid(connection):
                    conn_id = str(id(connection))
                    self._active_connections[conn_id] = connection
                    logger.info(f"Retrieved free connection. Active: {len(self._active_connections)}, Free: {self._free_connections.qsize()}")
                    return connection
            except queue.Empty:
                pass  # No free connections available

            # PRIORITY 2: Create new connection if within current pool size
            if len(self._active_connections) < self._current_pool_size:
                try:
                    connection = self._create_connection()
                    conn_id = str(id(connection))
                    self._active_connections[conn_id] = connection
                    logger.info(f"Created new connection within pool limit. Active: {len(self._active_connections)}")
                    return connection
                except Exception as e:
                    logger.error(f"Error creating new connection: {str(e)}")

            # PRIORITY 3: If waiting too long, consider scaling up
            wait_time = (datetime.now() - start_time).total_seconds()
            if wait_time >= 5:  # Wait 5 seconds before considering scale up
                if self._current_pool_size < self._max_pool_size:
                    new_size = min(self._current_pool_size + 2, self._max_pool_size)  # Increment by 2
                    logger.info(f"Scaling up pool from {self._current_pool_size} to {new_size}")
                    self._current_pool_size = new_size
                    try:
                        connection = self._create_connection()
                        conn_id = str(id(connection))
                        self._active_connections[conn_id] = connection
                        return connection
                    except Exception as e:
                        logger.error(f"Error creating connection during scale up: {str(e)}")

            # Check timeout
            if wait_time >= self.connection_timeout:
                raise PoolTimeoutError(f"Could not get connection after {self.connection_timeout} seconds")

            time.sleep(1)
            wait_count += 1

    def release_connection(self, connection):
        try:
            conn_id = str(id(connection))
            if conn_id in self._active_connections:
                del self._active_connections[conn_id]
                
                # Only reuse valid connections
                if self._is_connection_valid(connection):
                    self._free_connections.put(connection)
                    logger.info(f"Released valid connection to pool. Active: {len(self._active_connections)}, Free: {self._free_connections.qsize()}")
                else:
                    try:
                        connection.close()
                    except:
                        pass
                    logger.info("Closed invalid connection")

        except Exception as e:
            logger.error(f"Error releasing connection: {str(e)}")
            try:
                connection.close()
            except:
                pass

    def close_all(self):
        # Close active connections
        for conn in self._active_connections.values():
            try:
                conn.close()
            except:
                pass
        self._active_connections.clear()
        
        # Close free connections
        while True:
            try:
                conn = self._free_connections.get_nowait()
                try:
                    conn.close()
                except:
                    pass
            except queue.Empty:
                break
                
        logger.info("All connections closed")

@st.cache_resource
def get_connection_manager():
    return DynamicSnowflakePool(
        initial_pool_size=5,
        min_pool_size=3,
        max_pool_size=50,
        connection_timeout=30
    )

def get_data_sf(query, timeout=50000):
    result_container = []
    conn_manager = get_connection_manager()
    connection = None
    
    try:
        connection = conn_manager.get_connection()
        with connection.cursor() as cur:
            logger.info("Executing query...")
            cur.execute(query)
            result_container = cur.fetch_pandas_all()
            logger.info("Query executed successfully")
            
    except PoolTimeoutError as te:
        logger.error(f"Connection pool timeout: {str(te)}")
        raise HTTPException(status_code=503, 
            detail="Service is experiencing high load. Please try again in a few moments.")
            
    except snowflake.connector.ProgrammingError as e:
        logger.error(f"Snowflake error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Snowflake error: {str(e)}")
        
    except Exception as ex:
        logger.error(f"Error: {str(ex)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(ex)}")
        
    finally:
        if connection:
            conn_manager.release_connection(connection)
    
    return result_container

# Register cleanup handler
if 'on_close' not in st.session_state:
    st.session_state.on_close = lambda: get_connection_manager().close_all()
