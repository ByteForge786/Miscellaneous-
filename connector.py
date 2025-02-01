import snowflake.connector
from snowflake.connector.connection_pool import SimpleConnectionPool
import streamlit as st
import logging
import time
from datetime import datetime

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
                 scale_up_threshold=0.8,
                 scale_down_threshold=0.3,
                 connection_timeout=30):
        
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._current_pool_size = initial_pool_size
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._active_connections = 0
        self._free_connections = set()  # Track free connections
        self.connection_timeout = connection_timeout
        
        self._create_pool(self._current_pool_size)
        logger.info(f"Initialized dynamic pool with size: {self._current_pool_size}")

    def _create_pool(self, size):
        self._pool = SimpleConnectionPool(
            min_connections=size,
            max_connections=size,
            connection_kwargs={
                'user': property_values["snowflake.user"],
                'password': property_values["snowflake.password"],
                'account': property_values["snowflake.account"],
                'warehouse': property_values["snowflake.warehouse"],
                'database': property_values["snowflake.database"],
                'schema': property_values["snowflake.schema"]
            }
        )
        self._current_pool_size = size

    def _check_and_scale(self):
        # Only scale up if we have no free connections
        if len(self._free_connections) == 0:
            utilization = self._active_connections / self._current_pool_size
            
            # Scale up logic
            if utilization >= self._scale_up_threshold and self._current_pool_size < self._max_pool_size:
                new_size = min(
                    self._current_pool_size * 2,
                    self._max_pool_size
                )
                logger.info(f"No free connections available. Scaling up pool from {self._current_pool_size} to {new_size}")
                self._recreate_pool(new_size)
                
            # Scale down logic
            elif utilization <= self._scale_down_threshold and self._current_pool_size > self._min_pool_size:
                new_size = max(
                    self._current_pool_size // 2,
                    self._min_pool_size
                )
                logger.info(f"Pool underutilized. Scaling down from {self._current_pool_size} to {new_size}")
                self._recreate_pool(new_size)

    def _recreate_pool(self, new_size):
        old_pool = self._pool
        self._create_pool(new_size)
        try:
            old_pool.close_all_connections()
        except:
            pass
        self._free_connections.clear()

    def get_connection(self):
        start_time = datetime.now()
        
        while True:
            try:
                # First priority: Check for free connections
                if self._free_connections:
                    connection = self._free_connections.pop()
                    try:
                        # Test if connection is still valid
                        connection.cursor().execute("SELECT 1")
                        self._active_connections += 1
                        logger.info(f"Reused free connection. Active: {self._active_connections}/{self._current_pool_size}")
                        return connection
                    except:
                        # If connection is invalid, continue to get new connection
                        pass

                if self._active_connections >= self._current_pool_size:
                    wait_time = (datetime.now() - start_time).total_seconds()
                    if wait_time >= self.connection_timeout:
                        raise PoolTimeoutError(f"Timeout after {wait_time} seconds")
                    
                    # Check scaling only if no free connections
                    if not self._free_connections:
                        self._check_and_scale()
                    time.sleep(1)
                    continue
                
                connection = self._pool.get_connection()
                self._active_connections += 1
                logger.info(f"New connection acquired. Active: {self._active_connections}/{self._current_pool_size}")
                return connection
                
            except PoolTimeoutError:
                raise
            except Exception as e:
                wait_time = (datetime.now() - start_time).total_seconds()
                if wait_time >= self.connection_timeout:
                    raise PoolTimeoutError(f"Failed to get connection after {self.connection_timeout} seconds")
                time.sleep(1)

    def release_connection(self, connection):
        try:
            # Add to free connections pool instead of immediate release
            self._free_connections.add(connection)
            self._active_connections -= 1
            logger.info(f"Connection released to free pool. Active: {self._active_connections}/{self._current_pool_size}, Free: {len(self._free_connections)}")
            
            # Only check scaling if we have too many free connections
            if len(self._free_connections) > self._current_pool_size // 2:
                self._check_and_scale()
        except Exception as e:
            logger.error(f"Error releasing connection: {str(e)}")
            try:
                self._pool.release_connection(connection)
            except:
                pass

    def close_all(self):
        self._free_connections.clear()
        self._pool.close_all_connections()
        self._active_connections = 0
        logger.info("All connections closed")

@st.cache_resource
def get_connection_manager():
    return DynamicSnowflakePool(
        initial_pool_size=5,
        min_pool_size=3,
        max_pool_size=50,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
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

# Cleanup function for Streamlit shutdown
def cleanup():
    try:
        conn_manager = get_connection_manager()
        conn_manager.close_all()
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Register cleanup handler
if 'on_close' not in st.session_state:
    st.session_state.on_close = cleanup
