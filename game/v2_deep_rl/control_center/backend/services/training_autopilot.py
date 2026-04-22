from __future__ import annotations

from services.app_paths import ARTIFACTS_DIR, RUNS_DIR

from .autopilot import analysis as _analysis
from .autopilot import runner as _runner
from .autopilot.advisor import probe_ai_advisor
from .autopilot.constants import (
    CONTINUE_EPISODES,
    EPSILON_EXTENSION_FACTOR,
    INVALID_ACTION_HIGH,
    LR_MIN as _LR_MIN,
    MAX_LR_REDUCTIONS,
)
from .autopilot.history import get_autopilot_history
from .autopilot.naming import derive_base_run_name
from .autopilot.rating import compute_run_rating
from .autopilot.state import (
    clear_stop_request,
    get_settings,
    is_stop_requested,
    request_stop_after_cycle,
    save_settings,
)


def analyze_run(run_id: str, context: dict | None = None) -> dict:
    """Compatibility facade for tests and callers that patch this module."""
    _analysis.RUNS_DIR = RUNS_DIR
    return _analysis.analyze_run(run_id, context=context)


def _call_ai_advisor(*args, **kwargs) -> dict:
    return _runner.call_ai_advisor(*args, **kwargs)


def _derive_base_run_name(run_id: str) -> str:
    return derive_base_run_name(run_id)


def run_autopilot(run_id: str, dry_run: bool = False, context: dict | None = None) -> dict:
    """Compatibility facade that keeps historical monkeypatch points working."""
    original_values = {
        "runner_runs_dir": _runner.RUNS_DIR,
        "runner_analyze_run": _runner.analyze_run,
        "runner_call_ai_advisor": _runner.call_ai_advisor,
        "runner_get_settings": _runner.get_settings,
        "runner_is_stop_requested": _runner.is_stop_requested,
        "runner_clear_stop_request": _runner.clear_stop_request,
        "runner_derive_base_run_name": _runner.derive_base_run_name,
        "analysis_runs_dir": _analysis.RUNS_DIR,
    }
    try:
        _runner.RUNS_DIR = RUNS_DIR
        _runner.analyze_run = analyze_run
        _runner.call_ai_advisor = _call_ai_advisor
        _runner.get_settings = get_settings
        _runner.is_stop_requested = is_stop_requested
        _runner.clear_stop_request = clear_stop_request
        _runner.derive_base_run_name = _derive_base_run_name
        _analysis.RUNS_DIR = RUNS_DIR
        return _runner.run_autopilot(run_id, dry_run=dry_run, context=context)
    finally:
        _runner.RUNS_DIR = original_values["runner_runs_dir"]
        _runner.analyze_run = original_values["runner_analyze_run"]
        _runner.call_ai_advisor = original_values["runner_call_ai_advisor"]
        _runner.get_settings = original_values["runner_get_settings"]
        _runner.is_stop_requested = original_values["runner_is_stop_requested"]
        _runner.clear_stop_request = original_values["runner_clear_stop_request"]
        _runner.derive_base_run_name = original_values["runner_derive_base_run_name"]
        _analysis.RUNS_DIR = original_values["analysis_runs_dir"]


__all__ = [
    "ARTIFACTS_DIR",
    "CONTINUE_EPISODES",
    "EPSILON_EXTENSION_FACTOR",
    "INVALID_ACTION_HIGH",
    "MAX_LR_REDUCTIONS",
    "RUNS_DIR",
    "analyze_run",
    "clear_stop_request",
    "compute_run_rating",
    "get_autopilot_history",
    "get_settings",
    "is_stop_requested",
    "probe_ai_advisor",
    "request_stop_after_cycle",
    "run_autopilot",
    "save_settings",
    "_call_ai_advisor",
    "_derive_base_run_name",
    "_LR_MIN",
]
