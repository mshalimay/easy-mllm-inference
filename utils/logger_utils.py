import logging
import os
import random
import shutil
import time
from pathlib import Path

# Setup logging
from utils.constants import LOG_FOLDER

Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"


def setup_logger(name: str = "logger", log_file_path: str = LOG_FILE_PATH) -> logging.Logger:
    logger = logging.getLogger(name)
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler and logging format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")  # Simple format for console
        console_handler.setFormatter(console_formatter)

        # File handler and logging format
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


def save_log_file_path(save_dir: str) -> None:
    with open(os.path.join(save_dir, "log_files.txt"), "a+") as f:
        f.write(f"'{LOG_FILE_PATH}'\n")


def save_log_file(save_dir: str, log_filename: str = "log.txt") -> None:
    shutil.copy(LOG_FILE_PATH, os.path.join(save_dir, log_filename))


# Create a default logger when module is imported
logger = setup_logger()
