from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    return read_json(path)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value


def read_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    return sanitize_json_value(read_json(path))


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tail_csv_rows(path: Path, limit: int = 200) -> list[dict]:
    rows = read_csv_rows(path)
    if limit <= 0:
        return rows
    return rows[-limit:]


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
