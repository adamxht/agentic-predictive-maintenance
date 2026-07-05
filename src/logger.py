import logging
import os
import sys
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_DIRECTORY = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)
LOG_FORMAT = "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)

if __name__ == "__main__":
    logging.info("Logging has started")
