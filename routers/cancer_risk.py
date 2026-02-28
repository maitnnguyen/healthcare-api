"""
routers/cancer_risk.py
──────────────────────
Cancer risk prediction router.
Endpoints:
  POST /cancer-risk/predict  → predict risk for a single patient
  GET  /cancer-risk/data     → dataset for Power BI
  GET  /cancer-risk/info     → model metadata
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/cancer-risk", tags=["Cancer Risk"])

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
MODEL_PATH    = BASE_DIR / "models" / "cancer_risk_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "cancer_risk_metadata.json"
DATA_PATH     = BASE_DIR / "models" / "cancer_risk_sample_data.csv"

# ── Lazy-load ─────────────────────────────────────────────────────────────────
_model    = None
_metadata = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Ensure cancer_risk_model.pkl is in models/ folder."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

def get_metadata():
    global _metadata
    if _metadata is None:
        _metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    return _metadata


# ── Schemas ───────────────────────────────────────────────────────────────────
class ClinicalFeatures(BaseModel):
    mean_radius:          float = Field(..., ge=0, example=14.0)
    mean_texture:         float = Field(..., ge=0, example=19.0)
    mean_perimeter:       float = Field(..., ge=0, example=92.0)
    mean_area:            float = Field(..., ge=0, example=600.0)
    mean_concavity:       float = Field(..., ge=0, example=0.09)
    mean_concave_points:  float = Field(..., ge=0, example=0.05)
    worst_radius:         float = Field(..., ge=0, example=16.0)
    worst_perimeter:      float = Field(..., ge=0, example=107.0)
    worst_area:           float = Field(..., ge=0, example=800.0)
    worst_concave_points: float = Field(..., ge=0, example=0.15)

class LifestyleFeatures(BaseModel):
    age:               int   = Field(..., ge=18, le=110, example=55)
    smoking:           int   = Field(..., ge=0,  le=1,   example=1,  description="1=Yes, 0=No")
    family_history:    int   = Field(..., ge=0,  le=1,   example=0,  description="1=Yes, 0=No")
    bmi:               float = Field(..., ge=10, le=70,  example=27.5)
    alcohol_use:       int   = Field(..., ge=0,  le=1,   example=0,  description="1=Yes, 0=No")
    physical_activity: int   = Field(..., ge=0,  le=2,   example=1,  description="0=Low, 1=Medium, 2=High")

class PredictRequest(BaseModel):
    patient_id: Optional[str]    = Field(None, example="PAT-001")
    clinical:   ClinicalFeatures
    lifestyle:  LifestyleFeatures

class PredictResponse(BaseModel):
    patient_id:         Optional[str]
    risk_label:         str
    risk_probability:   float
    risk_score_percent: int
    confidence:         str
    top_risk_factors:   list
    recommendation:     str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_factors(req: PredictRequest) -> list:
    factors = []
    c, l = req.clinical, req.lifestyle
    if l.smoking:                        factors.append({"factor": "Smoking",               "value": "Yes",          "impact": "high"})
    if l.family_history:                 factors.append({"factor": "Family History",         "value": "Yes",          "impact": "high"})
    if c.worst_concave_points > 0.17:    factors.append({"factor": "Worst Concave Points",   "value": f"{c.worst_concave_points:.3f}", "impact": "high"})
    if c.worst_radius > 16:              factors.append({"factor": "Worst Radius",            "value": f"{c.worst_radius:.1f}",        "impact": "medium"})
    if c.mean_concavity > 0.1:           factors.append({"factor": "Mean Concavity",          "value": f"{c.mean_concavity:.3f}",      "impact": "medium"})
    if l.age > 60:                       factors.append({"factor": "Age",                     "value": str(l.age),     "impact": "medium"})
    if l.bmi > 30:                       factors.append({"factor": "BMI",                     "value": str(l.bmi),     "impact": "low"})
    if l.alcohol_use:                    factors.append({"factor": "Alcohol Use",              "value": "Yes",          "impact": "low"})
    if l.physical_activity == 0:         factors.append({"factor": "Physical Activity",        "value": "Low",          "impact": "low"})
    return factors[:5]

def _recommendation(p: float) -> str:
    if p >= 0.75:   return "⚠️ High risk. Immediate specialist referral and diagnostic imaging recommended."
    elif p >= 0.45: return "🔶 Moderate risk. Schedule follow-up screening within 3 months."
    else:           return "✅ Low risk. Continue routine annual screening and healthy lifestyle habits."

def _confidence(p: float) -> str:
    d = abs(p - 0.5)
    return "High" if d > 0.35 else "Medium" if d > 0.2 else "Low"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse,
             summary="Predict cancer risk for a patient")
async def predict(request: PredictRequest):
    model = get_model()
    c, l  = request.clinical, request.lifestyle

    features = np.array([[
        c.mean_radius, c.mean_texture, c.mean_perimeter, c.mean_area,
        c.mean_concavity, c.mean_concave_points,
        c.worst_radius, c.worst_perimeter, c.worst_area, c.worst_concave_points,
        l.age, l.smoking, l.family_history, l.bmi, l.alcohol_use, l.physical_activity
    ]])

    probability = float(model.predict_proba(features)[0][1])
    prediction  = int(model.predict(features)[0])

    return PredictResponse(
        patient_id         = request.patient_id,
        risk_label         = "High Risk" if prediction == 1 else "Low Risk",
        risk_probability   = round(probability, 4),
        risk_score_percent = int(probability * 100),
        confidence         = _confidence(probability),
        top_risk_factors   = _risk_factors(request),
        recommendation     = _recommendation(probability),
    )


@router.get("/data", summary="Sample dataset for Power BI")
async def get_data(limit: int = 500):
    """
    Returns prediction dataset as JSON.
    Connect Power BI via: Get Data → Web → this URL.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample data not found. Run train_cancer_risk_model.py first.")
    df = pd.read_csv(DATA_PATH).head(limit)
    df[df.select_dtypes('float').columns] = df.select_dtypes('float').round(4)
    return JSONResponse({"count": len(df), "columns": list(df.columns), "data": df.to_dict(orient="records")})


@router.get("/info", summary="Model metadata")
async def info():
    meta = get_metadata()
    return {
        "model":       "Gradient Boosting Classifier",
        "version":     "1.0.0",
        "description": "Cancer risk prediction from clinical and lifestyle features",
        "performance": {
            "roc_auc_test":    meta.get("model_auc"),
            "roc_auc_cv_mean": meta.get("cv_auc_mean"),
        },
        "data_source":      "ehr-fhir-pipeline → Kaggle global-cancer-patients-2015-2024",
        "powerbi_endpoint": "/cancer-risk/data",
        "predict_endpoint": "/cancer-risk/predict",
    }
