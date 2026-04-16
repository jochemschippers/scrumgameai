from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

from app import app

# ---------------------------------------------------------------------------
# Startup test validation
# ---------------------------------------------------------------------------
# Run the test suite once when the API server starts.  This catches regressions
# early and gives a clear signal if the deployed code is broken.  Tests run in
# a subprocess so torch stubs/real-torch import isolation doesn't bleed into
# the live server process.

_TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"


def _run_startup_tests() -> None:
    if not _TESTS_DIR.exists():
        print(f"[startup] tests directory not found at {_TESTS_DIR}, skipping.", flush=True)
        return

    print(f"[startup] Running test suite at {_TESTS_DIR} ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(_TESTS_DIR), "-q", "--tb=short", "--no-header"],
        capture_output=False,   # stream output directly to the terminal
        text=True,
    )
    if result.returncode == 0:
        print("[startup] All tests passed.", flush=True)
    elif result.returncode == 5:
        # exit code 5 = no tests collected (e.g. empty dir) — not a failure
        print("[startup] No tests collected — continuing.", flush=True)
    else:
        print(
            f"[startup] WARNING: test suite exited with code {result.returncode}. "
            "The server will still start, but please investigate the failures above.",
            flush=True,
        )


def _find_available_port(host: str, preferred_port: int, fallback_count: int = 15) -> int:
    """Pick the first available port starting from preferred_port."""
    for port in range(preferred_port, preferred_port + fallback_count + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find a free port in range {preferred_port}-{preferred_port + fallback_count}."
    )


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit(
            "uvicorn is not installed. Install backend requirements first: "
            "pip install -r game/v2_deep_rl/control_center/backend/requirements.txt"
        ) from error

    _run_startup_tests()

    host = "0.0.0.0"
    preferred_port = 8000
    port = _find_available_port(host, preferred_port)
    if port != preferred_port:
        print(
            f"Port {preferred_port} is in use. Starting Control Center API on http://{host}:{port}"
        )

    uvicorn.run(app, host=host, port=port)
