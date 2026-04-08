from __future__ import annotations

from app import app


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit(
            "uvicorn is not installed. Install backend requirements first: "
            "pip install -r game/v2_deep_rl/control_center/backend/requirements.txt"
        ) from error

    uvicorn.run(app, host="127.0.0.1", port=8000)
