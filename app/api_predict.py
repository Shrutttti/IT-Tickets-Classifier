"""
api_predict.py
--------------
FastAPI app that loads the trained LightGBM model and exposes
a /predict endpoint for IT ticket assignment group prediction.

Usage:
    # Set model path first
    export MODEL_PATH="/home/azureuser/localfiles/models/lgbm_classifier.joblib"
    uvicorn api_predict:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /          - health check
    POST /predict   - predict assignment group for a ticket
"""

import joblib
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "lgbm_classifier.joblib"))

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        f"Set the MODEL_PATH environment variable or run train.py first."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
print(f"[INFO] Model loaded from {MODEL_PATH}")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IT Ticket Classifier",
    description="Predicts the assignment group for an IT support ticket.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TicketInput(BaseModel):
    short_description: str = Field(..., example="Laptop not connecting to VPN")
    description: str       = Field(..., example="User unable to connect to VPN since this morning.")
    business_service: str  = Field(..., example="Business_Service_1")
    service_offering: str  = Field(..., example="Service_Offering_1")
    urgency: str           = Field(..., example="2 - Medium")
    priority: str          = Field(..., example="3 - Moderate")
    impact: str            = Field(..., example="2 - Medium")
    incident_type: str     = Field(..., example="Incident")


class PredictionOutput(BaseModel):
    predicted_assignment_group: str
    confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "IT Ticket Classifier API is running."}


@app.post("/predict", response_model=PredictionOutput)
def predict(ticket: TicketInput):
    try:
        now = datetime.now()

        input_df = pd.DataFrame([{
            "short_description": ticket.short_description,
            "description":       ticket.description,
            "business_service":  ticket.business_service,
            "service_offering":  ticket.service_offering,
            "urgency":           ticket.urgency,
            "priority":          ticket.priority,
            "impact":            ticket.impact,
            "incident_type":     ticket.incident_type,
            "opened_date":       now.day,
            "opened_month":      now.month,
            "opened_weekday":    now.weekday(),
        }])

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df).max()

        return PredictionOutput(
            predicted_assignment_group=str(prediction),
            confidence=round(float(proba), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))