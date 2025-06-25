# src/data/preprocessing.py

import os
import pandas as pd
import numpy as np
import logging
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Import load_data and setup_logger from data_load.py
from src.data.data_ingestion import load_data, setup_logger

def preprocess_data(data: pd.DataFrame, logger=None) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger()

    try:
        logger.info("Preprocessing started...")

        # Binary encode 'Status': D = 1, else 0
        data['Status'] = data['Status'].apply(lambda x: 1 if x == 'D' else 0)

        # Label encode
        label_cols = ['Sex', 'Drug', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        for col in label_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        # Drop ID column
        if 'ID' in data.columns:
            data = data.drop(columns=['ID'])

        # Impute missing values
        imputer = IterativeImputer(random_state=42)
        imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Convert columns to integers
        imputed_data['Stage'] = imputed_data['Stage'].round().astype('int64')
        imputed_data['Platelets'] = imputed_data['Platelets'].round().astype('int64')

        # Remove outliers using z-score
        z_scores = np.abs(zscore(imputed_data.select_dtypes(include=['float64', 'int64'])))
        data_no_outliers = imputed_data[(z_scores < 3).all(axis=1)].reset_index(drop=True)

        logger.info(f"Cleaned data shape: {data_no_outliers.shape}")
        return data_no_outliers

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Set base paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "cirrhosis.csv")
    CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    LOG_PATH = os.path.join(BASE_DIR, "logs", "preprocessing.log")

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)

    # Set up logger
    logger = setup_logger(LOG_PATH)

    # Load and preprocess
    data = load_data(RAW_DATA_PATH)
    cleaned_data = preprocess_data(data, logger)

    # Save processed data
    cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Saved cleaned data to {CLEANED_DATA_PATH}")
