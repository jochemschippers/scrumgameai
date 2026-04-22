from __future__ import annotations
import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Load API_KEY (and any other vars) from .env file if present
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from api.routes_autopilot import router as autopilot_router
from api.routes_campaigns import router as campaigns_router
from api.routes_checkpoints import router as checkpoints_router
from api.routes_configs import router as configs_router
from api.routes_jobs import router as jobs_router
from api.routes_play import router as play_router
from api.routes_runs import router as runs_router
from api.routes_testing import router as testing_router
from services.app_paths import ENGINE_ROOT
from storage.jobs_db import init_db

app = FastAPI(
    title="Scrum Game Control Center API",
    version="0.1.0",
    description="Custom backend for configs, runs, checkpoints, training jobs, testing, and play.",
)

# 1. FIX THE 405 ERRORS (CORS)
# This tells the browser: "It is okay to talk to me from the frontend."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your specific IP/Domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Block requests that don't carry the correct X-API-Key header.

    The check is skipped when:
    - No API_KEY env var is set (useful for local dev without a key).
    - The request is a CORS preflight (OPTIONS).
    - The path looks like a static asset (has a file extension, e.g. .js/.css/.html).
    - The path is the root "/" (which serves index.html).
    """
    api_key = os.environ.get("API_KEY")
    if api_key:
        is_preflight = request.method == "OPTIONS"
        is_static = request.url.path == "/" or bool(re.search(r"\.[a-zA-Z0-9]+$", request.url.path))
        if not is_preflight and not is_static:
            provided = request.headers.get("X-API-Key", "")
            if provided != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )
    return await call_next(request)


init_db()

app.include_router(autopilot_router)
app.include_router(campaigns_router)
app.include_router(configs_router)
app.include_router(runs_router)
app.include_router(checkpoints_router)
app.include_router(jobs_router)
app.include_router(play_router)
app.include_router(testing_router)

@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "engine_root": str(ENGINE_ROOT),
        "api_version": "0.1.0",
    }

# 2. FIX THE 404 ERRORS (Static Files)
# This mounts your frontend folder so you can visit http://127.0.0.1:8000/ instead of opening the file
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
