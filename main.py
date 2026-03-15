"""
main.py
───────
Healthcare API — FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import heart_disease
#from routers import cancer_risk, admissions

app = FastAPI(
    title="Healthcare API",
    description="""
## 🏥 Healthcare Prediction API

A modular FastAPI service serving ML-powered healthcare risk predictions.
Built on harmonized clinical data from the [ehr-fhir-pipeline](https://github.com/maitnnguyen/ehr-fhir-pipeline).

### Available Models
| Endpoint prefix     | Model                        | Status  |
|---------------------|------------------------------|---------|
| `/heart-disease`      | Heart Disease Prediction     | ✅ Live |
| `/cancer-risk`      | Cancer Risk Classification   | 🔜 Soon |
| `/admissions`       | 30-day Readmission Prediction| 🔜 Soon |
| `/diabetes`         | Diabetes Risk                | 🔜 Soon |

### Power BI Integration
| Dashboard Page | Connect to | Status |
|---|---|---|
| Heart Disease  | `/heart-disease/data` | ✅ Live |
| Cancer Risk    | `/cancer-risk/data` | 🔜 Soon |
| Readmissions   | `/admissions/data`  | 🔜 Soon |
""",
    version="1.1.0",
    contact={"name": "Mai Nguyen", "url": "https://github.com/maitnnguyen"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(heart_disease.router)
#app.include_router(cancer_risk.router) # comming soon
#app.include_router(admissions.router) # coming soon
# app.include_router(diabetes.router)  # coming soon

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Healthcare API",
        "version": "1.1.0",
        "status":  "healthy",
        "docs":    "/docs",
        "models": {
            "heart_disease": {
                "predict": "/heart-disease/predict",
                "data":    "/heart-disease/data",
                "info":    "/heart-disease/info",
            },
        },
        "coming_soon": [
                "/diabetes/predict",
		"/icu-mortality/predict",
		"/admissions/predict",
        ],
        "github": "https://github.com/maitnnguyen/healthcare-api",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
