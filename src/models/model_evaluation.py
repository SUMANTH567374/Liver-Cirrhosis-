

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split

# Setup base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(BASE_DIR, "data", "final", "balanced_data.csv")
models_dir = os.path.join(BASE_DIR, "models")
reports_dir = os.path.join(BASE_DIR, "reports")
plots_dir = os.path.join(BASE_DIR, "graphs", "evaluation")
logs_dir = os.path.join(BASE_DIR, "logs")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Setup logger
def setup_logger(log_file):
    logger = logging.getLogger("evaluation_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = setup_logger(os.path.join(logs_dir, "model_evaluation.log"))

try:
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Status'])
    y = data['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_files = [f for f in os.listdir(models_dir) if f.endswith("_best.pkl")]
    if not model_files:
        logger.warning("No models found for evaluation.")
        raise FileNotFoundError("No model files present in models/ directory.")

    # Create metrics.txt
    metrics_txt_path = os.path.join(reports_dir, "metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("Model Evaluation Summary\n")
        f.write("========================\n\n")

    # Store metrics for all models
    metrics_yaml_data = {}

    # Evaluate models
    for model_file in model_files:
        model_name = model_file.replace("_best.pkl", "")
        model_path = os.path.join(models_dir, model_file)

        logger.info(f"Evaluating model: {model_name}")
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = float(round(report['accuracy'], 4))
        auc = float(round(roc_auc_score(y_test, y_prob), 4)) if y_prob is not None else None

        # Save report as JSON
        result = {
            "model": model_name,
            "accuracy": accuracy,
            "classification_report": report,
            "roc_auc_score": auc
        }
        report_path = os.path.join(reports_dir, f"{model_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved evaluation report for {model_name} to {report_path}")

        # Save metrics for all models
        metrics_yaml_data[model_name] = {
            "accuracy": accuracy,
            "roc_auc_score": auc
        }

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        cm_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Saved confusion matrix for {model_name} to {cm_path}")

        # Save ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            roc_path = os.path.join(plots_dir, f"{model_name}_roc_curve.png")
            plt.savefig(roc_path)
            plt.close()
            logger.info(f"Saved ROC curve for {model_name} to {roc_path}")

        # Append to metrics.txt
        with open(metrics_txt_path, "a", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            if auc is not None:
                f.write(f"ROC AUC: {auc:.4f}\n")
            f.write("\n")

    # Save all metrics in YAML for DVC
    metrics_yaml_path = os.path.join(reports_dir, "metrics.yaml")
    with open(metrics_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(metrics_yaml_data, f)
    logger.info(f"Saved structured metrics to {metrics_yaml_path}")

    logger.info("✅ Model evaluation completed successfully.")

except Exception as e:
    logger.error(f"❌ Model evaluation failed: {str(e)}")
