from __future__ import annotations

import sys

import pytest

fastapi = pytest.importorskip("fastapi", exc_type=ImportError)
testclient = pytest.importorskip("fastapi.testclient", exc_type=ImportError)

FastAPI = fastapi.FastAPI
TestClient = testclient.TestClient

for module_name in ("shared_match_runner", "match_runner", "scrum_game_env"):
    sys.modules.pop(module_name, None)

from api.routes_play import router as play_router  # noqa: E402
from services import play_service  # noqa: E402


def setup_function():
    play_service.PLAY_SESSIONS.clear()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(play_router)
    return TestClient(app)


def test_play_api_creates_fetches_and_advances_shared_session():
    client = _client()
    create_response = client.post(
        "/play/session",
        json={
            "mode": "shared",
            "game_config_id": "default_game_config",
            "base_seed": 42,
            "seats": [{"type": "random", "display_name": "AI 1"}, {"type": "heuristic", "display_name": "AI 2"}],
        },
    )
    assert create_response.status_code == 200
    session = create_response.json()
    assert session["mode"] == "shared"
    assert session["board"]["products"]

    fetch_response = client.get(f"/play/session/{session['id']}")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["id"] == session["id"]

    advance_response = client.post(f"/play/session/{session['id']}/action", json={})
    assert advance_response.status_code == 200
    assert advance_response.json()["round_number"] == 2
    assert len(advance_response.json()["turn_log"]) == 2


def test_play_api_rejects_invalid_shared_sessions():
    client = _client()

    too_many = client.post(
        "/play/session",
        json={
            "mode": "shared",
            "game_config_id": "default_game_config",
            "seats": [{"type": "random"} for _ in range(5)],
        },
    )
    assert too_many.status_code == 400

    missing_checkpoint = client.post(
        "/play/session",
        json={
            "mode": "shared",
            "game_config_id": "default_game_config",
            "seats": [{"type": "model", "profile_name": "expert"}],
        },
    )
    assert missing_checkpoint.status_code == 400
