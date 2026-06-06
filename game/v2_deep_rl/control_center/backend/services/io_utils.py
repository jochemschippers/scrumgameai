"""
JSON and CSV File IO Utilities.

This module provides clean, robust helper routines to read JSON and CSV files,
sanitize infinite float values (e.g. converting `NaN`/`Infinity` to `None`), and safely parse
numeric inputs. This prevents JSON serialization exceptions when dealing with raw ML logs.

Connections:
  - Imported and called by services like `catalog_service` and `play_service` to parse log streams or configuration assets.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict:
    """Reads a JSON file from the filesystem.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed dictionary contents.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_or_empty(path: Path) -> dict:
    """Reads a JSON file, returning an empty dictionary if the file does not exist.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed dictionary, or {} if missing.
    """
    if not path.exists():
        return {}
    return read_json(path)


def sanitize_json_value(value: Any) -> Any:
    """Recursively walks a data structure and replaces non-finite floats (NaN, Inf) with None.

    This ensures the object can be serialized via python's standard json library.

    Args:
        value: Any nested data structure.

    Returns:
        The sanitized data structure.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value


def read_json_safe(path: Path) -> dict:
    """Reads a JSON file and sanitizes any infinite floats to make it safe for serialization.

    Args:
        path: Path to the JSON file.

    Returns:
        The sanitized dictionary, or {} if missing.
    """
    if not path.exists():
        return {}
    return sanitize_json_value(read_json(path))


def read_csv_rows(path: Path) -> list[dict]:
    """Reads all rows of a CSV file as a list of dictionaries.

    Args:
        path: Path to the CSV file.

    Returns:
        A list of dictionary rows representing CSV headers to values.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tail_csv_rows(path: Path, limit: int = 200) -> list[dict]:
    """Reads the last N lines of a CSV file.

    Args:
        path: Path to the CSV file.
        limit: Max number of trailing lines to read (<= 0 reads everything).

    Returns:
        A list of trailing dictionary rows.
    """
    rows = read_csv_rows(path)
    if limit <= 0:
        return rows
    return rows[-limit:]


def safe_float(value: Any) -> float | None:
    """Converts a value to float, returning None if empty or non-finite.

    Args:
        value: The raw input to parse.

    Returns:
        A finite float, or None.
    """
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_int(value: Any) -> int | None:
    """Converts a value to int, returning None if empty, invalid, or non-finite.

    Args:
        value: The raw input to parse.

    Returns:
        An integer value, or None.
    """
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
