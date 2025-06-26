import os
import pandas as pd
import logging
from tests.test_logger import setup_logger  # Shared logger config

# Initialize logging to shared file
setup_logger()

def test_data_ingestion():
    try:
        raw_path = "data/raw/cirrhosis.csv"
        assert os.path.exists(raw_path), "❌ Raw data file missing"
        logging.info("✅ Raw data file found.")

        df = pd.read_csv(raw_path)
        assert isinstance(df, pd.DataFrame), "❌ Not a DataFrame"
        assert df.shape[0] > 0, "❌ DataFrame is empty"
        assert 'Status' in df.columns, "❌ 'Status' column missing"
        
        logging.info(f"✅ DataFrame loaded with shape: {df.shape}")
        logging.info("✅ 'Status' column is present.")
    except AssertionError as e:
        logging.error(f"❌ Assertion failed: {e}")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        raise

