# import os
# import joblib
# import re

# def get_best_model(reports_path="reports/metrics.txt", models_dir="models"):
#     if not os.path.exists(reports_path):
#         raise FileNotFoundError(f"'{reports_path}' not found.")

#     best_model_name = None
#     best_accuracy = -1.0
#     current_model = None

#     with open(reports_path, "r") as f:
#         lines = f.readlines()

#     for line in lines:
#         model_match = re.match(r"Model:\s*(\w+)", line)
#         accuracy_match = re.match(r"Accuracy:\s*([\d.]+)", line)

#         if model_match:
#             current_model = model_match.group(1)

#         if accuracy_match and current_model:
#             acc = float(accuracy_match.group(1))
#             if acc > best_accuracy:
#                 best_accuracy = acc
#                 best_model_name = current_model

#     if best_model_name is None:
#         raise ValueError("No valid model accuracy found in metrics.txt.")

#     # Try loading compatible model first
#     compatible_file = f"{best_model_name}_compatible.pkl"
#     best_file = f"{best_model_name}_best.pkl"

#     compatible_path = os.path.join(models_dir, compatible_file)
#     best_path = os.path.join(models_dir, best_file)

#     model = None

#     # Prefer compatible file if available
#     if os.path.exists(compatible_path):
#         try:
#             model = joblib.load(compatible_path)
#             print(f"✅ Loaded model from: {compatible_path}")
#             return model, best_model_name, best_accuracy
#         except Exception as e:
#             print(f"⚠️ Failed to load compatible model: {e}")

#     # Fall back to original best file
#     if os.path.exists(best_path):
#         try:
#             model = joblib.load(best_path)
#             print(f"✅ Loaded model from: {best_path}")
#             return model, best_model_name, best_accuracy
#         except Exception as e:
#             raise RuntimeError(f"❌ Could not load model from {best_path}: {e}")

#     raise FileNotFoundError(f"❌ Neither {compatible_file} nor {best_file} found in {models_dir}.")

import os
import joblib
import re

def get_best_model(reports_path="reports/metrics.txt", models_dir="models"):
    if not os.path.exists(reports_path):
        raise FileNotFoundError(f"'{reports_path}' not found.")

    best_model_name = None
    best_accuracy = -1.0
    current_model = None

    with open(reports_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        model_match = re.match(r"Model:\s*(\w+)", line)
        accuracy_match = re.match(r"Accuracy:\s*([\d.]+)", line)

        if model_match:
            current_model = model_match.group(1)

        if accuracy_match and current_model:
            acc = float(accuracy_match.group(1))
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = current_model

    if best_model_name is None:
        raise ValueError("No valid model accuracy found in metrics.txt.")

    best_file = f"{best_model_name}_best.pkl"
    best_path = os.path.join(models_dir, best_file)

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"❌ Model file not found: {best_file}")

    try:
        model = joblib.load(best_path)
        print(f"✅ Loaded model from: {best_path}")
        return model, best_model_name, best_accuracy
    except Exception as e:
        raise RuntimeError(f"❌ Could not load model from {best_path}: {e}")
