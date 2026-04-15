from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from services.app_paths import BACKEND_DIR, ENGINE_ROOT, RUNS_DIR, ensure_engine_import_path
from storage.jobs_db import create_job, delete_job, get_job, init_db, list_jobs as list_jobs_db, update_job, utc_now_iso

ensure_engine_import_path()


def _slugify_run_name(value: str | None) -> str:
    text = str(value or "").strip().lower()
    slug = "".join(character if character.isalnum() else "_" for character in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:48]


def _create_timestamped_run_directory(run_name=None):
    # Inlined from train_dqn to avoid importing torch/matplotlib in the web venv.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("run_%Y-%m-%d_%H%M")
    run_suffix = _slugify_run_name(run_name)
    if run_suffix:
        base_name = f"{base_name}_{run_suffix}"
    candidate = RUNS_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = RUNS_DIR / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


VALID_JOB_TYPES = {"train", "fine_tune", "evaluate", "robustness"}
RUNNER_PATH = BACKEND_DIR / "jobs" / "job_runner.py"

_CACHED_PYTHON_COMMAND: str | None = None


def _choose_python_command() -> str:
    """Return a Python executable that can import torch.

    The web backend may run from a venv that lacks torch/matplotlib.  Training
    subprocesses *must* use the system Python (or whichever interpreter has
    torch installed).  We probe candidates once and cache the result.
    """
    global _CACHED_PYTHON_COMMAND
    if _CACHED_PYTHON_COMMAND is not None:
        return _CACHED_PYTHON_COMMAND

    # Fast path: current interpreter already has torch.
    try:
        import torch  # noqa: F401
        _CACHED_PYTHON_COMMAND = sys.executable
        return _CACHED_PYTHON_COMMAND
    except ImportError:
        pass

    # Probe common executables for one that has torch installed.
    import shutil
    for candidate in ("python", "python3", "py"):
        exe = shutil.which(candidate)
        if not exe:
            continue
        try:
            probe = subprocess.run(
                [exe, "-c", "import torch"],
                capture_output=True,
                timeout=15,
            )
            if probe.returncode == 0:
                _CACHED_PYTHON_COMMAND = exe
                return _CACHED_PYTHON_COMMAND
        except Exception:
            continue

    # No torch-capable interpreter found; fall back so jobs fail with a clear error.
    _CACHED_PYTHON_COMMAND = sys.executable
    return _CACHED_PYTHON_COMMAND


def _is_pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but we lack kill permission — treat as alive.
        return True
    except OSError as exc:
        if os.name == "nt":
            # WinError 87 (ERROR_INVALID_PARAMETER) definitively means the PID does not exist.
            # WinError 6 (ERROR_INVALID_HANDLE) occurs when os.kill is called from inside a
            # DETACHED_PROCESS even when the target process is alive — treat as alive.
            # Any other winerror is ambiguous; assume alive to avoid false "failed" jobs.
            return getattr(exc, "winerror", None) != 87
        return False


def _create_job_run_dir(job_type: str, run_name: str | None = None) -> Path:
    if job_type in {"train", "fine_tune"}:
        return _create_timestamped_run_directory(run_name=run_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / f"{job_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_stdout_log(run_dir: Path, job_type: str) -> Path:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{job_type}_stdout.log"


def refresh_job_states() -> list[dict]:
    init_db()
    jobs = list_jobs_db()
    for job in jobs:
        if job["status"] == "running" and not _is_pid_running(job.get("worker_pid")):
            update_job(
                job["id"],
                status="failed",
                ended_at=utc_now_iso(),
                error_message=job.get("error_message") or "Worker process exited unexpectedly.",
            )
    return list_jobs_db()


def dispatch_next_job() -> dict | None:
    init_db()
    jobs = refresh_job_states()
    if any(job["status"] == "running" for job in jobs):
        return None

    queued_job = next((job for job in reversed(jobs) if job["status"] == "queued"), None)
    if queued_job is None:
        return None

    stdout_path = queued_job.get("stdout_log_path") or ""
    stdout_log_path = Path(stdout_path) if stdout_path else _default_stdout_log(Path(queued_job["run_dir"]), queued_job["job_type"])
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_log_path.open("ab") as stdout_handle:
        popen_kwargs = {
            "cwd": BACKEND_DIR,
            "stdout": stdout_handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            [
                _choose_python_command(),
                str(RUNNER_PATH),
                "--job-id",
                str(queued_job["id"]),
            ],
            **popen_kwargs,
        )

    time.sleep(0.2)
    return update_job(
        queued_job["id"],
        status="running",
        started_at=utc_now_iso(),
        worker_pid=process.pid,
        stdout_log_path=str(stdout_log_path),
        error_message=None,
    )


def list_jobs() -> list[dict]:
    init_db()
    return refresh_job_states()


def get_job_details(job_id: int) -> dict | None:
    init_db()
    refresh_job_states()
    return get_job(job_id)


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _tail_csv_rows(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit <= 0:
        return rows
    return rows[-limit:]


def _read_run_metadata(run_dir: Path | None) -> dict:
    if not run_dir:
        return {}
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def get_job_progress(job_id: int) -> dict | None:
    init_db()
    refresh_job_states()
    job = get_job(job_id)
    if job is None:
        return None

    run_dir_raw = job.get("run_dir")
    run_dir = Path(run_dir_raw) if run_dir_raw else None
    payload = job.get("payload", {})
    run_metadata = _read_run_metadata(run_dir)
    total_episodes = _safe_int(str(run_metadata.get("episodes_this_run", "")))
    if total_episodes is None:
        total_episodes = _safe_int(str(payload.get("episodes", "")))
    start_episode = _safe_int(str(run_metadata.get("start_episode", ""))) or 1

    progress = {
        "job_id": job["id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "run_dir": str(run_dir) if run_dir else "",
        "stdout_log_path": job.get("stdout_log_path") or "",
        "error_message": job.get("error_message"),
        "total_episodes": total_episodes,
        "start_episode": start_episode,
        "end_episode": _safe_int(str(run_metadata.get("end_episode", ""))),
        "latest_episode": 0,
        "completed_episodes": 0,
        "progress_ratio": 0.0,
        "latest_training_row": None,
        "latest_evaluation_row": None,
        "training_series": [],
        "evaluation_series": [],
    }

    if not run_dir:
        return progress

    reports_dir = run_dir / "reports"
    training_rows = _tail_csv_rows(reports_dir / "logs.csv", limit=240)
    evaluation_rows = _tail_csv_rows(reports_dir / "evaluation_history.csv", limit=120)

    training_series = []
    for row in training_rows:
        episode = _safe_int(row.get("episode"))
        if episode is None:
            continue
        training_series.append(
            {
                "episode": episode,
                "episode_reward": _safe_float(row.get("episode_reward")),
                "rolling_average_reward": _safe_float(row.get("rolling_average_reward")),
                "mean_recent_loss": _safe_float(row.get("mean_recent_loss")),
                "average_ending_money": _safe_float(row.get("average_ending_money")),
                "epsilon": _safe_float(row.get("epsilon")),
            }
        )

    evaluation_series = []
    for row in evaluation_rows:
        episode = _safe_int(row.get("episode"))
        if episode is None:
            continue
        evaluation_series.append(
            {
                "episode": episode,
                "average_reward": _safe_float(row.get("average_reward")),
                "bankruptcy_rate": _safe_float(row.get("bankruptcy_rate")),
                "average_ending_money": _safe_float(row.get("average_ending_money")),
                "invalid_action_rate": _safe_float(row.get("invalid_action_rate")),
            }
        )

    latest_training = training_series[-1] if training_series else None
    latest_evaluation = evaluation_series[-1] if evaluation_series else None
    latest_episode = latest_training["episode"] if latest_training else 0
    completed_episodes = max(0, latest_episode - start_episode + 1)
    ratio = 0.0
    if total_episodes and total_episodes > 0:
        ratio = max(0.0, min(1.0, completed_episodes / total_episodes))

    progress.update(
        {
            "latest_episode": latest_episode,
            "completed_episodes": completed_episodes,
            "progress_ratio": ratio,
            "latest_training_row": latest_training,
            "latest_evaluation_row": latest_evaluation,
            "training_series": training_series,
            "evaluation_series": evaluation_series,
        }
    )
    return progress


def get_job_log_tail(job_id: int, max_lines: int = 80) -> dict | None:
    init_db()
    refresh_job_states()
    job = get_job(job_id)
    if job is None:
      return None

    log_path_raw = job.get("stdout_log_path") or ""
    log_path = Path(log_path_raw) if log_path_raw else None
    if not log_path or not log_path.exists():
        return {
            "job_id": job_id,
            "stdout_log_path": log_path_raw,
            "lines": [],
        }

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    return {
        "job_id": job_id,
        "stdout_log_path": str(log_path),
        "lines": [line.rstrip("\n") for line in lines[-max_lines:]],
    }


def enqueue_train_job(payload: dict) -> dict:
    init_db()
    resume_from = payload.get("resume_from")
    resume_mode = payload.get("resume_mode", "strict")
    job_type = "fine_tune" if resume_from and resume_mode in {"fine_tune", "fine-tune"} else "train"
    run_dir = _create_job_run_dir(job_type, run_name=payload.get("run_name"))
    stdout_log_path = _default_stdout_log(run_dir, job_type)

    job = create_job(
        job_type=job_type,
        payload=payload,
        stdout_log_path=str(stdout_log_path),
        run_dir=str(run_dir),
    )
    dispatch_next_job()
    return get_job(job["id"])


def enqueue_evaluation_job(payload: dict) -> dict:
    init_db()
    job_type = payload.get("job_type", "robustness")
    if job_type not in {"evaluate", "robustness"}:
        raise ValueError("Only evaluate and robustness job types are supported.")

    run_dir = Path(payload["run_dir"]).resolve()
    stdout_log_path = _default_stdout_log(run_dir, job_type)
    result_path = str(run_dir / "robustness_results.csv")

    job = create_job(
        job_type=job_type,
        payload=payload,
        stdout_log_path=str(stdout_log_path),
        run_dir=str(run_dir),
        result_path=result_path,
    )
    dispatch_next_job()
    return get_job(job["id"])


def stop_job(job_id: int) -> dict:
    init_db()
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"Job `{job_id}` was not found.")

    if job["status"] == "queued":
        return update_job(job_id, status="stopped", ended_at=utc_now_iso(), error_message="Stopped before execution.")

    if job["status"] != "running":
        return job

    pid = job.get("worker_pid")
    if _is_pid_running(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                os.kill(pid, signal.SIGTERM)

    updated = update_job(job_id, status="stopped", ended_at=utc_now_iso(), error_message="Stopped by user request.")
    dispatch_next_job()
    return updated


def dismiss_job(job_id: int) -> dict:
    init_db()
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"Job `{job_id}` was not found.")

    if job["status"] in {"queued", "running"}:
        raise ValueError(f"Job `{job_id}` cannot be dismissed while it is {job['status']}.")

    deleted = delete_job(job_id)
    if not deleted:
        raise ValueError(f"Job `{job_id}` could not be dismissed.")

    return {"ok": True, "job_id": job_id}
