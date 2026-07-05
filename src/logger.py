import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # File path and naming convention.
os.makedirs(logs_path, exist_ok=True) # Even if file already exist, we will still append

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Config logging path, format, levels, etc
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", # Suggested format
    level = logging.INFO, # we will get our logs when we call logging.INFO
)

# For testing
if __name__ == "__main__":
    logging.info("Logging has started")