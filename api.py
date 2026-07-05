"""REST inference API for the predictive maintenance cascade.

Usage:
    uvicorn api:app --reload
    Docs: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_COLUMNS

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load("stage1_model.joblib")
    models["stage1"] = bundle["model"]
    models["threshold"] = bundle["threshold"]
    models["stage2"] = joblib.load("stage2_model.joblib")
    yield
    models.clear()


app = FastAPI(
    title="Predictive Maintenance AI API",
    description="Two-stage cascade (Random Forest) for industrial equipment failure prediction",
    version="1.0.0",
    lifespan=lifespan,
)


class SensorReading(BaseModel):
    air_temperature_k: float = Field(..., ge=290, le=310, json_schema_extra={"example": 300.0})
    process_temperature_k: float = Field(..., ge=300, le=320, json_schema_extra={"example": 310.0})
    rotational_speed_rpm: float = Field(..., ge=900, le=3200, json_schema_extra={"example": 1500})
    torque_nm: float = Field(..., ge=0, le=90, json_schema_extra={"example": 40.0})
    tool_wear_min: float = Field(..., ge=0, le=280, json_schema_extra={"example": 100})
    machine_type: str = Field(..., pattern="^[LMH]$", json_schema_extra={"example": "L"})


class PredictionResponse(BaseModel):
    failure_probability: float
    is_failure: bool
    stage1_threshold: float
    failure_type: str
    type_probabilities: dict


def build_features(reading: SensorReading) -> pd.DataFrame:
    power = reading.torque_nm * (reading.rotational_speed_rpm * 2 * 3.14159 / 60)
    temp_diff = reading.process_temperature_k - reading.air_temperature_k
    wear_x_torque = reading.tool_wear_min * reading.torque_nm

    row = {
        "Air temperature [K]": reading.air_temperature_k,
        "Process temperature [K]": reading.process_temperature_k,
        "Rotational speed [rpm]": reading.rotational_speed_rpm,
        "Torque [Nm]": reading.torque_nm,
        "Tool wear [min]": reading.tool_wear_min,
        "Power_W": power,
        "Temp_diff_K": temp_diff,
        "Wear_x_Torque": wear_x_torque,
        "Type_H": 1 if reading.machine_type == "H" else 0,
        "Type_L": 1 if reading.machine_type == "L" else 0,
        "Type_M": 1 if reading.machine_type == "M" else 0,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.get("/")
def root():
    return {"status": "ok", "model": "two-stage RF cascade", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    try:
        X = build_features(reading)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    stage1 = models["stage1"]
    threshold = models["threshold"]
    stage2 = models["stage2"]

    fail_proba = float(stage1.predict_proba(X)[0, 1])
    is_failure = fail_proba >= threshold

    failure_type = "No Failure"
    type_probs = {}

    if is_failure:
        failure_type = stage2.predict(X)[0]
        proba_arr = stage2.predict_proba(X)[0]
        type_probs = {cls: round(float(p), 4) for cls, p in zip(stage2.classes_, proba_arr)}

    return PredictionResponse(
        failure_probability=round(fail_proba, 4),
        is_failure=is_failure,
        stage1_threshold=round(threshold, 4),
        failure_type=failure_type,
        type_probabilities=type_probs,
    )