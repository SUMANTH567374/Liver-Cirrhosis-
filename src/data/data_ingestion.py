# src/data/data_load.py

import os
import pandas as pd
import logging

def setup_logger(log_file: str):
    """Set up file and terminal (console) logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV data with error handling."""
    logger = setup_logger("logs/data_load.log")
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("File is empty or has incorrect format.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading file: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("data", "raw", "cirrhosis.csv")
    df = load_data(csv_path)
    print(df.head())
