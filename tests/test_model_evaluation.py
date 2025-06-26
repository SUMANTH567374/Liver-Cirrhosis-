import os
import yaml
import logging
from tests.test_logger import setup_logger  # centralized logger setup

# Setup logging
setup_logger()

def test_metrics_file():
    try:
        with open("reports/metrics.yaml", "r") as f:
            metrics = yaml.safe_load(f)

        assert isinstance(metrics, dict), "❌ Metrics file is not a valid dictionary."

        for model_name in ["GradientBoosting", "KNN", "LogisticRegression", "RandomForest", "SVM"]:
            assert model_name in metrics, f"❌ {model_name} not found in metrics"
            assert "accuracy" in metrics[model_name], f"❌ {model_name} missing accuracy"
            assert metrics[model_name]["accuracy"] > 0.5, f"❌ {model_name} accuracy is too low"

            logging.info(f"✅ {model_name} accuracy: {metrics[model_name]['accuracy']}")
    except AssertionError as e:
        logging.error(f"❌ AssertionError: {e}")
        raise
    except Exception as ex:
        logging.error(f"❌ Unexpected error occurred: {ex}")
        raise
