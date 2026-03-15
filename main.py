"""
main.py — Healthcare API v1.2
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from routers import heart_disease

app = FastAPI(
    title="Healthcare API",
    description="""
## 🏥 Healthcare Prediction API

| Endpoint prefix   | Model         | Status     |
|-------------------|---------------|------------|
| `/heart-disease`  | Heart Disease | ✅ Live    |
| `/cancer-risk`    | Cancer Risk   | 🔜 Soon   |
| `/icu-mortality`  | ICU Mortality | 🔜 Soon   |
""",
    version="1.2.0",
    contact={"name": "Mai Nguyen", "url": "https://github.com/maitnnguyen"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(heart_disease.router)

# ── Serve static files ────────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Dashboard UI ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def dashboard():
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>Healthcare API</h1><p><a href='/docs'>Swagger Docs</a></p>")

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}

@app.get("/api", tags=["Health"])
async def api_info():
    return {
        "service": "Healthcare API",
        "version": "1.2.0",
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
            "/cancer-risk/predict",
            "/icu-mortality/predict",
        ],
        "github": "https://github.com/maitnnguyen/healthcare-api",
    }
