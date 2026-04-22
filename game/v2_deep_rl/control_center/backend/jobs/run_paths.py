from __future__ import annotations

from datetime import datetime
from pathlib import Path

from services.app_paths import RUNS_DIR


def slugify_run_name(value: str | None) -> str:
    text = str(value or "").strip().lower()
    slug = "".join(character if character.isalnum() else "_" for character in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:48]


def create_timestamped_run_directory(run_name=None):
    # Inlined from training.train_dqn to avoid importing torch/matplotlib in the web venv.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("run_%Y-%m-%d_%H%M")
    run_suffix = slugify_run_name(run_name)
    if run_suffix:
        base_name = f"{base_name}_{run_suffix}"
    candidate = RUNS_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = RUNS_DIR / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def create_job_run_dir(job_type: str, run_name: str | None = None) -> Path:
    if job_type in {"train", "fine_tune"}:
        return create_timestamped_run_directory(run_name=run_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / f"{job_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def default_stdout_log(run_dir: Path, job_type: str) -> Path:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{job_type}_stdout.log"
