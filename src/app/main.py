# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# import logging
# from src.app.load_best_model import get_best_model

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# app = FastAPI(
#     title="Liver Cirrhosis Prediction API",
#     description="Predict liver cirrhosis status using the best trained ML model",
#     version="1.0.0"
# )

# # Global variables
# model = None
# model_name = ""
# model_accuracy = 0.0

# # Input schema
# class PatientData(BaseModel):
#     Bilirubin: float
#     Copper: float
#     Prothrombin: float
#     Age: float
#     SGOT: float
#     Albumin: float
#     Cholesterol: float
#     Platelets: float
#     Alk_Phos: float
#     Tryglicerides: float
#     Ascites: int
#     Drug: int
#     Stage: int

# @app.on_event("startup")
# def load_model():
#     """Load the best model on startup."""
#     global model, model_name, model_accuracy
#     try:
#         model, model_name, model_accuracy = get_best_model()
#         logging.info(f"✅ Loaded model: {model_name} with accuracy: {model_accuracy:.4f}")
#     except Exception as e:
#         logging.exception("❌ Failed to load model.")
#         raise RuntimeError("Could not load model. Check metrics.txt and model files.") from e

# @app.get("/")
# def read_root():
#     return {
#         "message": "Liver Cirrhosis Prediction API is running.",
#         "model": model_name,
#         "accuracy": f"{model_accuracy:.4f}"
#     }

# @app.post("/predict")
# def predict(data: PatientData):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded. Try again later.")

#     try:
#         df = pd.DataFrame([data.dict()])

#         # Ensure correct feature order if available
#         if hasattr(model, "feature_names_in_"):
#             missing_features = set(model.feature_names_in_) - set(df.columns)
#             if missing_features:
#                 raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
#             df = df[model.feature_names_in_]

#         prediction = model.predict(df)[0]

#         label_map = {
#             0: "Death",
#             1: "Censored"
#         }

#         return {
#             "prediction": int(prediction),
#             "label": label_map.get(int(prediction), "Unknown"),
#             "model": model_name,
#             "confidence": "✔ Model selected based on best accuracy from metrics.txt"
#         }

#     except Exception as e:
#         logging.exception("❌ Prediction failed.")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
from src.app.load_best_model import get_best_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Liver Cirrhosis Prediction API",
    description="Predict liver cirrhosis status using the best trained ML model",
    version="1.0.0"
)

# Global variables
model = None
model_name = ""
model_accuracy = 0.0

# Input schema
class PatientData(BaseModel):
    Bilirubin: float
    Copper: float
    Prothrombin: float
    Age: float
    SGOT: float
    Albumin: float
    Cholesterol: float
    Platelets: float
    Alk_Phos: float
    Tryglicerides: float
    Ascites: int
    Drug: int
    Stage: int

@app.on_event("startup")
def load_model():
    global model, model_name, model_accuracy
    try:
        model, model_name, model_accuracy = get_best_model()
        logging.info(f"✅ Loaded model: {model_name} with accuracy: {model_accuracy:.4f}")
    except Exception as e:
        logging.exception("❌ Failed to load model.")
        raise RuntimeError("Could not load model. Check metrics.txt and model files.") from e

@app.get("/")
def read_root():
    return {
        "message": "Liver Cirrhosis Prediction API is running.",
        "model": model_name,
        "accuracy": f"{model_accuracy:.4f}"
    }

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Try again later.")

    try:
        df = pd.DataFrame([data.dict()])

        # Ensure correct feature order
        if hasattr(model, "feature_names_in_"):
            missing = set(model.feature_names_in_) - set(df.columns)
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
            df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]

        label_map = {
            0: "Death",
            1: "Censored"
        }

        return {
            "prediction": int(prediction),
            "label": label_map.get(int(prediction), "Unknown"),
            "model": model_name,
            "confidence": "✔ Model selected based on best accuracy from metrics.txt"
        }

    except Exception as e:
        logging.exception("❌ Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
