import logging
import os
from datetime import datetime

# Generate dynamic log file path
ct = datetime.today()
log_file_path = (property_values['data_load.log'] +
                 f"data_load_{table_name}_{ct.strftime('%Y%m%d%H%M%S')}.log")

# Ensure the directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Configure logging to write to a file
logging.basicConfig(
    filename=log_file_path,  # Specify the log file
    level=logging.DEBUG,  # Set the appropriate log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Test logging
logging.info("This is an info message.")
logging.debug("This is a debug message.")
logging.error("This is an error message.")
