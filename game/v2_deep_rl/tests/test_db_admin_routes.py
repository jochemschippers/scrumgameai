"""Test db admin routes behavior."""

from __future__ import annotations

import pytest


# Handle credentials.
def _credentials(username: str, password: str):
    # Group the state and behavior for credentials.
    class Credentials:
        pass

    credentials = Credentials()
    credentials.username = username
    credentials.password = password
    return credentials


# Handle temp user db.
@pytest.fixture
def temp_user_db(tmp_path, monkeypatch):
    from storage import jobs_db

    monkeypatch.setattr(jobs_db, "DB_PATH", tmp_path / "control_center.db")
    monkeypatch.setenv("CONTROL_CENTER_ADMIN_USER", "admin")
    monkeypatch.setenv("CONTROL_CENTER_ADMIN_PASSWORD", "admin-secret")
    monkeypatch.setenv("CONTROL_CENTER_GUEST_USER", "guest")
    monkeypatch.setenv("CONTROL_CENTER_GUEST_PASSWORD", "guest-secret")
    jobs_db.init_db()
    return jobs_db


# Verify init db seeds admin and guest users.
def test_init_db_seeds_admin_and_guest_users(temp_user_db):
    users = temp_user_db.list_users()

    assert [user["username"] for user in users] == ["admin", "guest"]
    assert [user["role"] for user in users] == ["admin", "guest"]
    assert all("password_hash" not in user for user in users)


# Verify db admin accepts seeded admin user.
def test_db_admin_accepts_seeded_admin_user(temp_user_db):
    from api.routes_db_admin import require_db_admin

    assert require_db_admin(_credentials("admin", "admin-secret")) == "admin"


# Verify db admin accepts seeded guest user.
def test_db_admin_accepts_seeded_guest_user(temp_user_db):
    from api.routes_db_admin import require_db_admin

    assert require_db_admin(_credentials("guest", "guest-secret")) == "guest"


# Verify db admin rejects wrong password.
def test_db_admin_rejects_wrong_password(temp_user_db):
    from api.routes_db_admin import require_db_admin

    with pytest.raises(Exception) as excinfo:
        require_db_admin(_credentials("admin", "wrong"))

    assert getattr(excinfo.value, "status_code", None) == 401
