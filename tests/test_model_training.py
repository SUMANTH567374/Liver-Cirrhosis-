import os
import logging
from joblib import load
from tests.test_logger import setup_logger  # Import central logger setup

# Initialize logger
setup_logger()

def test_model_artifacts():
    models = [
        "models/RandomForest_best.pkl",
        "models/GradientBoosting_best.pkl",
        "models/LogisticRegression_best.pkl",
        "models/KNN_best.pkl",
        "models/SVM_best.pkl"
    ]

    for model_path in models:
        try:
            assert os.path.exists(model_path), f"{model_path} not found"
            model = load(model_path)
            assert hasattr(model, "predict"), f"{model_path} is not a valid model"
            logging.info(f"✅ Loaded and validated model: {model_path}")
        except AssertionError as ae:
            logging.error(f"❌ AssertionError - {ae}")
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error loading {model_path} - {e}")
            raise
