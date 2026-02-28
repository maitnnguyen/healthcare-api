"""
main.py
───────
Healthcare API — FastAPI application
Serves ML model predictions for healthcare risk assessment.

Deployed on Render: https://healthcare-api.onrender.com
Swagger UI:         https://healthcare-api.onrender.com/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import cancer_risk

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Healthcare API",
    description="""
## 🏥 Healthcare Prediction API

A modular FastAPI service serving ML-powered healthcare risk predictions.
Built on harmonized clinical data from the [ehr-fhir-pipeline](https://github.com/maitnnguyen/ehr-fhir-pipeline).

### Available Models
| Endpoint prefix     | Model                  | Status  |
|---------------------|------------------------|---------|
| `/cancer-risk`      | Cancer Risk Classifier | ✅ Live |
| `/readmission`      | Hospital Readmission   | 🔜 Soon |
| `/diabetes`         | Diabetes Risk          | 🔜 Soon |

### Data Source
Clinical data harmonized via FHIR R4 pipeline from Kaggle.

### Power BI Integration
Connect Power BI to `/cancer-risk/data` for live dashboard updates.
""",
    version="1.0.0",
    contact={
        "name":  "Mai Nguyen",
        "url":   "https://github.com/maitnnguyen",
    },
    license_info={"name": "MIT"},
)

# ── CORS (allow Power BI + browser access) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(cancer_risk.router)
# Future:
# app.include_router(readmission.router)
# app.include_router(diabetes.router)

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service":     "Healthcare API",
        "version":     "1.0.0",
        "status":      "healthy",
        "docs":        "/docs",
        "endpoints": {
            "cancer_risk_predict": "/cancer-risk/predict",
            "cancer_risk_data":    "/cancer-risk/data",
            "cancer_risk_info":    "/cancer-risk/info",
        },
        "coming_soon": ["/readmission/predict", "/diabetes/predict"],
        "github":      "https://github.com/maitnnguyen/healthcare-api",
        "portfolio":   "https://github.com/maitnnguyen",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
