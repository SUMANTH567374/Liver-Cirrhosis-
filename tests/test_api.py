import pytest
from fastapi.testclient import TestClient 
from src.app.main import app, load_model  # Import load_model directly
import logging

client = TestClient(app)

# Sample valid input
sample_input = {
    "Bilirubin": 1.2,
    "Copper": 50.0,
    "Prothrombin": 12.5,
    "Age": 60,
    "SGOT": 85.0,
    "Albumin": 3.5,
    "Cholesterol": 180.0,
    "Platelets": 300.0,
    "Alk_Phos": 120.0,
    "Tryglicerides": 100.0,
    "Ascites": 0,
    "Drug": 1,
    "Stage": 3
}


@pytest.fixture(scope="module", autouse=True)
def setup_model_once():
    """Ensure model is loaded before any tests run."""
    try:
        load_model()
    except Exception as e:
        logging.error(f"❌ Setup failed: {e}")
        pytest.fail(f"Model failed to load during setup: {e}")


def test_api_predict_success():
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200, f"❌ Prediction failed: {response.status_code} - {response.text}"
    data = response.json()
    assert "prediction" in data
    assert "label" in data
