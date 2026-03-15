"""
routers/heart_disease.py
────────────────────────
Heart disease prediction router.

Endpoints:
  POST /heart-disease/predict  → predict heart disease risk
  GET  /heart-disease/data     → dataset for Power BI
  GET  /heart-disease/info     → model metadata
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

router    = APIRouter(prefix="/heart-disease", tags=["Heart Disease"])
BASE_DIR  = Path(__file__).parent.parent
MODEL_PATH    = BASE_DIR / "models" / "heart_disease_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "heart_disease_metadata.json"
DATA_PATH     = BASE_DIR / "models" / "heart_disease_sample_data.csv"

_model = _metadata = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run heart_disease_prediction.ipynb first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def get_metadata():
    global _metadata
    if _metadata is None:
        _metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    return _metadata


# ── Schemas ───────────────────────────────────────────────────────────────────
class HeartDiseaseRequest(BaseModel):
    patient_id: Optional[str] = Field(None, example="PAT-001")
    age:        int   = Field(..., ge=18, le=110, example=55)
    sex:        int   = Field(..., ge=0,  le=1,   example=1,   description="1=Male, 0=Female")
    cp:         int   = Field(..., ge=0,  le=3,   example=2,   description="Chest pain: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
    trestbps:   int   = Field(..., ge=80, le=220,  example=130, description="Resting blood pressure (mmHg)")
    chol:       int   = Field(..., ge=100,le=600,  example=240, description="Serum cholesterol (mg/dl)")
    fbs:        int   = Field(..., ge=0,  le=1,   example=0,   description="Fasting blood sugar >120 mg/dl")
    restecg:    int   = Field(..., ge=0,  le=2,   example=1,   description="Resting ECG: 0=Normal, 1=ST-T abnorm, 2=LV hypertrophy")
    thalach:    int   = Field(..., ge=60, le=220,  example=150, description="Max heart rate achieved")
    exang:      int   = Field(..., ge=0,  le=1,   example=0,   description="Exercise induced angina")
    oldpeak:    float = Field(..., ge=0,  le=7,   example=1.5, description="ST depression induced by exercise")
    slope:      int   = Field(..., ge=0,  le=2,   example=1,   description="Slope of ST segment: 0=Up, 1=Flat, 2=Down")
    ca:         int   = Field(..., ge=0,  le=3,   example=0,   description="Number of major vessels (0-3)")
    thal:       int   = Field(..., ge=0,  le=3,   example=2,   description="Thalassemia: 0=Normal, 1=Fixed defect, 2=Reversible defect")

class HeartDiseaseResponse(BaseModel):
    patient_id:        Optional[str]
    prediction_label:  str
    probability:       float
    risk_score_percent:int
    confidence:        str
    risk_factors:      list
    recommendation:    str

FEATURE_ORDER = ['age','sex','cp','trestbps','chol','fbs',
                 'restecg','thalach','exang','oldpeak','slope','ca','thal']

def _risk_factors(req: HeartDiseaseRequest) -> list:
    factors = []
    if req.ca > 0:       factors.append({"factor": "Major Vessels Blocked", "value": str(req.ca),       "impact": "high"})
    if req.thal == 2:    factors.append({"factor": "Thalassemia",           "value": "Reversible",       "impact": "high"})
    if req.cp == 3:      factors.append({"factor": "Chest Pain",            "value": "Asymptomatic",     "impact": "high"})
    if req.oldpeak > 2:  factors.append({"factor": "ST Depression",         "value": str(req.oldpeak),   "impact": "high"})
    if req.thalach < 120:factors.append({"factor": "Max Heart Rate",        "value": str(req.thalach),   "impact": "medium"})
    if req.exang == 1:   factors.append({"factor": "Exercise Angina",       "value": "Yes",              "impact": "medium"})
    if req.chol > 240:   factors.append({"factor": "Cholesterol",           "value": str(req.chol),      "impact": "medium"})
    if req.age > 60:     factors.append({"factor": "Age",                   "value": str(req.age),       "impact": "low"})
    return factors[:5]

def _recommendation(p: float) -> str:
    if p >= 0.7:   return "⚠️ High risk. Urgent cardiology referral and further diagnostic testing recommended."
    elif p >= 0.4: return "🔶 Moderate risk. Schedule cardiology consultation and stress test within 30 days."
    else:          return "✅ Low risk. Maintain healthy lifestyle and continue routine annual check-ups."

def _confidence(p: float) -> str:
    return "High" if abs(p-0.5) > 0.3 else "Medium" if abs(p-0.5) > 0.15 else "Low"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=HeartDiseaseResponse,
             summary="Predict heart disease risk")
async def predict(request: HeartDiseaseRequest):
    model    = get_model()
    features = np.array([[getattr(request, f) for f in FEATURE_ORDER]])
    prob     = float(model.predict_proba(features)[0][1])
    pred     = int(model.predict(features)[0])
    return HeartDiseaseResponse(
        patient_id         = request.patient_id,
        prediction_label   = "Heart Disease" if pred == 1 else "No Disease",
        probability        = round(prob, 4),
        risk_score_percent = int(prob * 100),
        confidence         = _confidence(prob),
        risk_factors       = _risk_factors(request),
        recommendation     = _recommendation(prob),
    )

@router.get("/data", summary="Heart disease dataset for Power BI")
async def get_data(limit: int = 500):
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample data not found.")
    df = pd.read_csv(DATA_PATH).head(limit)
    return JSONResponse({"count": len(df), "columns": list(df.columns),
                         "data": df.to_dict(orient="records")})

@router.get("/info", summary="Heart disease model metadata")
async def info():
    meta = get_metadata()
    return {"model": "Gradient Boosting", "version": "1.0.0",
            "target": "Heart disease presence",
            "performance": {"roc_auc_test": meta.get("model_auc"),
                            "roc_auc_cv":   meta.get("cv_auc_mean")},
            "data_source": "Cleveland Heart Disease Dataset — Kaggle"}
