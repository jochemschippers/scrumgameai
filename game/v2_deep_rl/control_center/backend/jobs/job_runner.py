from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.app_paths import BACKEND_DIR, ENGINE_ROOT
from storage.jobs_db import get_job, init_db, update_job, utc_now_iso


def build_command(job: dict) -> list[str]:
    payload = job["payload"]
    python_command = sys.executable

    if job["job_type"] in {"train", "fine_tune"}:
        command = [
            python_command,
            str(ENGINE_ROOT / "train_dqn.py"),
            "--run-dir",
            str(job["run_dir"]),
        ]
        if payload.get("game_config_path"):
            command.extend(["--game-config", str(payload["game_config_path"])])
        if payload.get("training_config_path"):
            command.extend(["--training-config", str(payload["training_config_path"])])
        if payload.get("episodes") is not None:
            command.extend(["--episodes", str(int(payload["episodes"]))])
        if payload.get("evaluation_episodes") is not None:
            command.extend(["--evaluation-episodes", str(int(payload["evaluation_episodes"]))])
        if payload.get("learning_rate") is not None:
            command.extend(["--learning-rate", str(payload["learning_rate"])])
        if payload.get("gamma") is not None:
            command.extend(["--gamma", str(payload["gamma"])])
        if payload.get("seed") is not None:
            command.extend(["--seed", str(int(payload["seed"]))])
        if payload.get("run_notes"):
            command.extend(["--notes", str(payload["run_notes"])])
        if payload.get("resume_from"):
            command.extend(["--resume-from", str(payload["resume_from"])])
            command.extend(["--resume-mode", str(payload.get("resume_mode", "strict"))])
        return command

    if job["job_type"] in {"evaluate", "robustness"}:
        return [
            python_command,
            str(ENGINE_ROOT / "evaluate_ddqn_robustness.py"),
            "--run-dir",
            str(payload["run_dir"]),
        ]

    raise ValueError(f"Unsupported job type: {job['job_type']}")


def run_job(job_id: int) -> int:
    init_db()
    job = get_job(job_id)
    if job is None:
        raise RuntimeError(f"Job `{job_id}` was not found.")

    stdout_log_path = Path(job["stdout_log_path"])
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(job)

    with stdout_log_path.open("ab") as stdout_handle:
        result = subprocess.run(
            command,
            cwd=ENGINE_ROOT,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )

    status = "completed" if result.returncode == 0 else "failed"
    error_message = None if result.returncode == 0 else f"Command exited with code {result.returncode}."
    update_job(
        job_id,
        status=status,
        ended_at=utc_now_iso(),
        error_message=error_message,
    )
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run one queued Control Center job.")
    parser.add_argument("--job-id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_job(args.job_id))
