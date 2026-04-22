from __future__ import annotations

import pytest


def test_db_admin_requires_configured_password(monkeypatch):
    from api.routes_db_admin import require_db_admin

    class Credentials:
        username = "admin"
        password = "anything"

    monkeypatch.delenv("CONTROL_CENTER_DB_PASSWORD", raising=False)
    monkeypatch.delenv("DB_ADMIN_PASSWORD", raising=False)

    with pytest.raises(Exception) as excinfo:
        require_db_admin(Credentials())

    assert getattr(excinfo.value, "status_code", None) == 503


def test_db_admin_accepts_configured_password(monkeypatch):
    from api.routes_db_admin import require_db_admin

    class Credentials:
        username = "admin"
        password = "secret"

    monkeypatch.setenv("CONTROL_CENTER_DB_PASSWORD", "secret")
    monkeypatch.delenv("DB_ADMIN_PASSWORD", raising=False)

    assert require_db_admin(Credentials()) == "admin"


def test_db_admin_rejects_wrong_password(monkeypatch):
    from api.routes_db_admin import require_db_admin

    class Credentials:
        username = "admin"
        password = "wrong"

    monkeypatch.setenv("CONTROL_CENTER_DB_PASSWORD", "secret")

    with pytest.raises(Exception) as excinfo:
        require_db_admin(Credentials())

    assert getattr(excinfo.value, "status_code", None) == 401
