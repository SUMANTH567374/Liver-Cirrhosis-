import os
import pandas as pd
import logging
from src.features.feature_selection import select_features
from tests.test_logger import setup_logger
import yaml

# Setup centralized logger
setup_logger()

# Load parameters from params.yaml
def load_params():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    params_path = os.path.join(base_dir, "params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def test_feature_selection():
    try:
        df = pd.read_csv("data/processed/cleaned_data.csv")
        logging.info("✅ Cleaned data loaded for feature selection.")

        params = load_params()
        top_n = params["features"]["top_n"]

        selected_features = select_features(
            data=df,
            target_column="Status",
            drop_columns=["N_Days"],
            top_n=top_n
        )

        # Validations
        assert isinstance(selected_features, list), "❌ selected_features is not a list"
        assert len(selected_features) == top_n, f"❌ Expected {top_n} features, got {len(selected_features)}"

        logging.info(f"✅ Feature selection complete: {len(selected_features)} features selected.")
    except AssertionError as e:
        logging.error(f"❌ AssertionError: {e}")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        raise
