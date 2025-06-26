import os
import pandas as pd
import logging
from src.data.data_preprocessing import preprocess_data
from tests.test_logger import setup_logger  # shared logger config

# Setup centralized logger
setup_logger()

def test_preprocessing_output():
    try:
        df = pd.read_csv("data/raw/cirrhosis.csv")
        logging.info("✅ Raw data loaded for preprocessing.")

        cleaned = preprocess_data(df)
        assert isinstance(cleaned, pd.DataFrame), "❌ Preprocessing output is not a DataFrame"
        assert cleaned.isnull().sum().sum() == 0, "❌ Missing values found after preprocessing"
        assert 'Status' in cleaned.columns, "❌ 'Status' column missing in cleaned data"

        logging.info(f"✅ Cleaned DataFrame shape: {cleaned.shape}")
        logging.info("✅ No missing values and 'Status' column is present.")
    except AssertionError as e:
        logging.error(f"❌ Assertion failed: {e}")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        raise

