# 🏥 Healthcare API

> A modular FastAPI service serving ML-powered healthcare risk predictions — built for extensibility across multiple clinical use cases.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)
![License](https://img.shields.io/badge/License-MIT-green)

**[🚀 Live API →](https://healthcare-api.onrender.com)**  
**[📖 Swagger Docs →](https://healthcare-api.onrender.com/docs)**

---

## 🏗️ Portfolio Architecture

```
ehr-fhir-pipeline          cancer-ml-models
(Kaggle → FHIR → CSV)  →   (EDA + training → .pkl)
                                    ↓
                          healthcare-api  (this repo)
                          FastAPI on Render
                                    ↓
                              Power BI Dashboard
                          /cancer-risk/data endpoint
```

---

## 📁 Repository Structure

```
healthcare-api/
├── main.py                        # FastAPI app entry point
├── routers/
│   ├── __init__.py
│   ├── cancer_risk.py             # ✅ Cancer risk endpoints
│   └── (readmission.py)           # 🔜 Coming soon
├── models/
│   ├── README.md                  # How to populate model files
│   ├── cancer_risk_model.pkl      # gitignored — generate locally
│   ├── cancer_risk_metadata.json  # gitignored — generate locally
│   └── cancer_risk_sample_data.csv# gitignored — generate locally
├── render.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔌 Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info & endpoint map |
| `GET` | `/health` | Health check (used by Render) |
| `POST` | `/cancer-risk/predict` | Predict cancer risk for a patient |
| `GET` | `/cancer-risk/data` | Full dataset — **connect Power BI here** |
| `GET` | `/cancer-risk/info` | Model metadata & performance |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🤖 Models

| Model | Endpoint | Algorithm | Status |
|---|---|---|---|
| Cancer Risk | `/cancer-risk` | Gradient Boosting | ✅ Live |
| Hospital Readmission | `/readmission` | — | 🔜 Soon |
| Diabetes Risk | `/diabetes` | — | 🔜 Soon |

---

## ⚙️ Local Setup

```bash
# 1. Clone
git clone https://github.com/maitnnguyen/healthcare-api.git
cd healthcare-api

# 2. Install
pip install -r requirements.txt

# 3. Generate model files (from cancer-ml-models)
cd ../cancer-ml-models
python train_cancer_risk_model.py
cp cancer_risk_model.pkl       ../healthcare-api/models/
cp cancer_risk_metadata.json   ../healthcare-api/models/
cp cancer_risk_sample_data.csv ../healthcare-api/models/

# 4. Run
cd ../healthcare-api
uvicorn main:app --reload

# 5. Open
# http://localhost:8000/docs
```

---

## 🚀 Deploy to Render

1. Create a new repo: `healthcare-api`
2. Push this code
3. Go to [render.com](https://render.com) → **New Web Service**
4. Connect the repo — Render auto-detects `render.yaml`
5. Click **Deploy** — live in ~2 minutes

> ⚠️ Before deploying, copy the `.pkl` and `.json` model files into `models/` and commit them **once** (they are gitignored by default to keep the repo clean during development).

---

## 📊 Power BI Integration

1. Open Power BI Desktop
2. **Get Data → Web**
3. Enter: `https://healthcare-api.onrender.com/cancer-risk/data?limit=500`
4. See `POWERBI_SETUP.md` for full dashboard build guide

---

## 🔗 Related Repositories

| Repo | Role |
|---|---|
| [ehr-fhir-pipeline](https://github.com/maitnnguyen/ehr-fhir-pipeline) | Data layer — FHIR harmonization |
| [cancer-ml-models](https://github.com/maitnnguyen/cancer-ml-models) | ML layer — training notebooks |
| [dataready](https://github.com/maitnnguyen/dataready) | AI consulting tool |

---

*Built by [Mai Nguyen](https://github.com/maitnnguyen) — data & AI consultant.*
