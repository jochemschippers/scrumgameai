"""Implement state behavior for the autopilot package."""

from __future__ import annotations

import json

from services.app_paths import ARTIFACTS_DIR
from services.io_utils import read_json_or_empty

STOP_FLAG_PATH = ARTIFACTS_DIR / "autopilot_stop_requested.flag"
SETTINGS_PATH = ARTIFACTS_DIR / "autopilot_settings.json"


def get_settings() -> dict:
    """Return autopilot feature toggles (logic_enabled, ai_enabled)."""
    defaults = {"logic_enabled": True, "ai_enabled": True}
    if not SETTINGS_PATH.exists():
        return defaults
    try:
        stored = read_json_or_empty(SETTINGS_PATH)
        return {**defaults, **stored}
    except Exception:
        return defaults


def save_settings(payload: dict) -> dict:
    """Update one or both toggles and persist."""
    current = get_settings()
    if "logic_enabled" in payload:
        current["logic_enabled"] = bool(payload["logic_enabled"])
    if "ai_enabled" in payload:
        current["ai_enabled"] = bool(payload["ai_enabled"])
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current


def request_stop_after_cycle() -> None:
    """Signal the autopilot to stop after the current training block finishes."""
    STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    STOP_FLAG_PATH.touch()


def clear_stop_request() -> None:
    """Clear a pending stop-after-cycle request so the autopilot resumes."""
    STOP_FLAG_PATH.unlink(missing_ok=True)


def is_stop_requested() -> bool:
    """Return True if the user has requested stop-after-cycle."""
    return STOP_FLAG_PATH.exists()
