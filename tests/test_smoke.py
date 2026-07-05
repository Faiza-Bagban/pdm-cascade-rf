"""Smoke tests -- verify trained artifacts and API respond correctly.
Run: pytest tests/ -v
"""
import os

import joblib
from fastapi.testclient import TestClient

from api import app
from src.features import FEATURE_COLUMNS, engineer_features, load_data


def test_model_files_exist():
    assert os.path.exists("stage1_model.joblib"), "Run: python -m src.train_cascade"
    assert os.path.exists("stage2_model.joblib"), "Run: python -m src.train_cascade"


def test_models_load():
    bundle = joblib.load("stage1_model.joblib")
    assert "model" in bundle and "threshold" in bundle
    stage2 = joblib.load("stage2_model.joblib")
    assert hasattr(stage2, "predict")


def test_feature_engineering_shape():
    df = load_data("data/ai4i2020.csv")
    df = engineer_features(df)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing expected feature column: {col}"


def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_api_predict():
    payload = {
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.2,
        "rotational_speed_rpm": 1400,
        "torque_nm": 65.0,
        "tool_wear_min": 200,
        "machine_type": "L",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert body["failure_type"] in ["No Failure", "HDF", "OSF", "PWF", "RNF", "TWF"]