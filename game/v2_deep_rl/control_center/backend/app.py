"""
FastAPI application assembly for the Scrum Game Control Center.

This module acts as the core entrypoint/assembly for the FastAPI backend service.
It sets up global HTTP middleware, registers API routing endpoints, handles database initialization,
and mounts the static web frontend files.

Key Responsibilities:
  1. CORS Management: Configures cross-origin resource sharing for development environments.
  2. Middleware Authentication: Evaluates incoming requests to enforce Bearer JWT authentication for protected API prefixes.
  3. Router Registration: Integrates modules from `api/` (configs, jobs, runs, checkpoints, campaigns, autopilot, etc.).
  4. Frontend Hosting: Mounts the frontend SPA directory statically to serve HTML, JS, and CSS components.

Connections:
  - Spawned By: `control_center/backend/run_api.py`
  - Imports: Routes from `api/`, path configurations from `services.app_paths`, database from `storage.jobs_db`.
  - Serves: HTML and static JS files from `control_center/frontend/`.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes_auth import router as auth_router, decode_token
from api.routes_autopilot import router as autopilot_router
from api.routes_campaigns import router as campaigns_router
from api.routes_checkpoints import router as checkpoints_router
from api.routes_configs import router as configs_router
from api.routes_db_admin import router as db_admin_router
from api.routes_jobs import router as jobs_router
from api.routes_play import router as play_router
from api.routes_runs import router as runs_router
from api.routes_testing import router as testing_router
from services.app_paths import ENGINE_ROOT
from storage.jobs_db import init_db

# Authentication is scoped to API prefixes. Static assets and the login page
# must remain public so a browser without a token can start the login flow.
_PROTECTED_PREFIXES = (
    "/autopilot",
    "/campaigns",
    "/configs",
    "/checkpoints",
    "/jobs",
    "/play",
    "/runs",
    "/testing",
    "/db-admin",
)

app = FastAPI(
    title="Scrum Game Control Center API",
    version="0.1.0",
    description="Custom backend for configs, runs, checkpoints, training jobs, testing, and play.",
)

app.add_middleware(
    CORSMiddleware,
    # LAN development uses different browser origins. Internet-facing deployments
    # should replace this wildcard with an explicit allowlist.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Block access to API routes unless a valid JWT is present."""
    path = request.url.path

    if (
        request.method == "OPTIONS"
        or path.startswith("/auth/")
        or path == "/health"
        or not path.startswith(_PROTECTED_PREFIXES)
    ):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    token = auth_header[len("Bearer "):]
    if decode_token(token) is None:
        return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)

    return await call_next(request)


init_db()

app.include_router(auth_router)
app.include_router(autopilot_router)
app.include_router(campaigns_router)
app.include_router(configs_router)
app.include_router(db_admin_router)
app.include_router(runs_router)
app.include_router(checkpoints_router)
app.include_router(jobs_router)
app.include_router(play_router)
app.include_router(testing_router)


@app.get("/health", tags=["system"])
def health():
    """Return liveness information without importing the ML runtime."""
    return {
        "status": "ok",
        "engine_root": str(ENGINE_ROOT),
        "api_version": "0.1.0",
    }

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    # Mount last so explicit API routes take precedence over frontend files.
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
