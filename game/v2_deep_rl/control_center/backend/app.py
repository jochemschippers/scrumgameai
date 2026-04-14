from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes_autopilot import router as autopilot_router
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

init_db()

app.include_router(autopilot_router)
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
