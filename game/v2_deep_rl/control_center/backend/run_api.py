from __future__ import annotations

import socket

from app import app


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

    host = "0.0.0.0"
    preferred_port = 8000
    port = _find_available_port(host, preferred_port)
    if port != preferred_port:
        print(
            f"Port {preferred_port} is in use. Starting Control Center API on http://{host}:{port}"
        )

    uvicorn.run(app, host=host, port=port)
