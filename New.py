import logging
import os
from datetime import datetime

# Generate dynamic log file path
ct = datetime.today()
log_file_path = (property_values['data_load.log'] +
                 f"data_load_{table_name}_{ct.strftime('%Y%m%d%H%M%S')}.log")

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,  # Adjust level if necessary
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Test logging
logging.info("Log entry created successfully!")
logging.error("This is a test error message.")
