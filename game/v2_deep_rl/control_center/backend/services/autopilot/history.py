"""Implement history behavior for the autopilot package."""

from __future__ import annotations

import json
from pathlib import Path

from services.app_paths import RUNS_DIR


def write_decision_record(run_dir: Path, decision: dict) -> None:
    """Append a serializable autopilot decision dictionary to a local JSONL report log file."""
    records_path = run_dir / "reports" / "autopilot_decisions.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision) + "\n")


def get_autopilot_history(run_id: str) -> list[dict]:
    """Return all autopilot decisions recorded for a run, oldest first."""
    run_dir = RUNS_DIR / run_id
    records_path = run_dir / "reports" / "autopilot_decisions.jsonl"
    if not records_path.exists():
        return []
    records = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
