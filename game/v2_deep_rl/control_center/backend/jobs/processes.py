from __future__ import annotations

import os
import subprocess
import sys

_CACHED_PYTHON_COMMAND: str | None = None


def choose_python_command() -> str:
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


def is_pid_running(pid: int | None) -> bool:
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
