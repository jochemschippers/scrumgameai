# Backend

This backend will expose a small Python API over the existing RL engine.

Planned responsibilities:

- config asset listing and persistence
- run and checkpoint browsing
- compatibility checks
- queued training and evaluation jobs
- playable session lifecycle

Suggested internal layout:

- `api/` for HTTP route handlers
- `schemas/` for request and response models
- `services/` for orchestration around existing RL modules
- `jobs/` for queue and worker logic
- `storage/` for SQLite and JSON-backed app state

## Current scaffold

The first backend scaffold now includes:

- `app.py` FastAPI entrypoint
- `run_api.py` local server launcher
- `api/routes_configs.py`
- `api/routes_runs.py`
- `api/routes_checkpoints.py`
- `api/routes_jobs.py`
- `services/catalog_service.py`
- `services/app_paths.py`
- `services/checkpoint_service.py`
- `requirements.txt`
- `storage/jobs_db.py`
- `jobs/queue_manager.py`
- `jobs/job_runner.py`

## Install and run

From the repo root:

```powershell
pip install -r game\v2_deep_rl\control_center\backend\requirements.txt
python game\v2_deep_rl\control_center\backend\run_api.py
```

First available routes:

- `GET /health`
- `GET /configs/game`
- `GET /configs/training`
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /checkpoints`
- `GET /checkpoints/{checkpoint_id}/compatibility?game_config_id=...`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/train`
- `POST /jobs/evaluate`
- `POST /jobs/{job_id}/stop`

## Job storage

Queued and completed jobs are persisted in:

- `storage/control_center.db`

The queue model is:

- one active worker at a time
- new jobs are persisted as `queued`
- the wrapper worker promotes jobs to `running`, then `completed` or `failed`
- queued and running jobs can be stopped through the job API
