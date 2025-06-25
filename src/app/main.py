from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.app.load_best_model import get_best_model

app = FastAPI(
    title="Liver Cirrhosis Prediction API",
    description="Predict liver cirrhosis status using the best trained ML model",
    version="1.0.0"
)

# Global variables to hold the model
model = None
model_name = ""
model_accuracy = 0.0

# Define input schema
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
    """Load the best model on startup."""
    global model, model_name, model_accuracy
    try:
        model, model_name, model_accuracy = get_best_model()
        print(f"✅ Loaded model: {model_name} with accuracy: {model_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError("Could not load model. Check metrics.txt and model files.")

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
        prediction = model.predict(df)[0]
        return {
            "prediction": int(prediction),
            "model": model_name,
            "confidence": "✔ Model selected based on best accuracy from metrics.txt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
