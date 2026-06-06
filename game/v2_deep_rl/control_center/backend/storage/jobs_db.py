"""
Control Center Relational Database Storage.

This module manages the SQLite schema initialization and CRUD helper queries for the Control Center database (`control_center.db`).
It handles table creation, user password hashing/salting using HMAC PBKDF2-SHA256, session authentication checks,
and background job queue logging (storing parameters, execution logs, run dirs, result paths, and worker PIDs).

Database Schema:
  - `users` Table: Stores secure credentials (username, salt/hash, role: admin/guest, activity flag).
  - `jobs` Table: Tracks long-running training/evaluation background task states (queued, running, completed, failed, stopped).

Connections:
  - Imports: System path settings from `services.app_paths`.
  - Used by: Route guards (`dependencies.py`), auth routers (`routes_auth.py`), and job scheduling managers (`queue_manager.py`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import secrets
import sqlite3

from services.app_paths import BACKEND_DIR


DB_PATH = BACKEND_DIR / "storage" / "control_center.db"
PASSWORD_HASH_ITERATIONS = 260_000


def utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    """Open and return a new SQLite database connection with row factory configured."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Initialize database tables and seed default users if not already present."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                worker_pid INTEGER,
                stdout_log_path TEXT,
                run_dir TEXT,
                result_path TEXT,
                error_message TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'guest')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _seed_user(
            connection,
            username=os.environ.get("CONTROL_CENTER_ADMIN_USER", "admin"),
            password=(
                os.environ.get("CONTROL_CENTER_ADMIN_PASSWORD")
                or os.environ.get("CONTROL_CENTER_DB_PASSWORD")
                or "admin"
            ),
            role="admin",
        )
        _seed_user(
            connection,
            username=os.environ.get("CONTROL_CENTER_GUEST_USER", "guest"),
            password=os.environ.get("CONTROL_CENTER_GUEST_PASSWORD") or "guest",
            role="guest",
        )
        connection.commit()


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique salt."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against a PBKDF2-HMAC-SHA256 hash in a constant-time comparison."""
    try:
        algorithm, iterations_raw, salt, expected = str(password_hash).split("$", 3)
        iterations = int(iterations_raw)
    except (ValueError, TypeError):
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(actual, expected)


def _seed_user(connection, username, password, role):
    """Insert a seed user if the username does not exist in the database."""
    existing = connection.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing is not None:
        return
    now = utc_now_iso()
    connection.execute(
        "INSERT INTO users (username, password_hash, role, is_active, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
        (username, hash_password(password), role, now, now),
    )


def get_user_by_username(username: str):
    """Retrieve user details dictionary by username."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def authenticate_user(username: str, password: str):
    """Authenticate credentials and return user dict if valid and active."""
    user = get_user_by_username(username)
    if not user or not int(user.get("is_active", 0)):
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def list_users():
    """Retrieve all users sorted by ID."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, role, is_active, created_at, updated_at FROM users ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _row_to_job(row):
    """Parse sqlite3.Row and deserialize job payload JSON."""
    if row is None:
        return None
    payload = dict(row)
    payload["payload"] = json.loads(payload.pop("payload_json"))
    return payload


def create_job(job_type, payload, stdout_log_path="", run_dir="", result_path=""):
    """Create a new job in the database with status 'queued'."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (job_type, status, payload_json, created_at, stdout_log_path, run_dir, result_path, error_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_type, "queued", json.dumps(payload), utc_now_iso(), stdout_log_path, run_dir, result_path, None),
        )
        conn.commit()
        return get_job(cursor.lastrowid)


def list_jobs():
    """List all database jobs sorted from newest to oldest."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    return [_row_to_job(r) for r in rows]


def get_job(job_id):
    """Retrieve a single job record by its ID."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row)


def delete_job(job_id):
    """Delete a job record by its ID and return True if successful."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
    return cursor.rowcount > 0


def update_job(job_id, **fields):
    """Update job fields in database and return the updated job record."""
    if not fields:
        return get_job(job_id)
    assignments = []
    values = []
    for key, value in fields.items():
        if key == "payload":
            assignments.append("payload_json = ?")
            values.append(json.dumps(value))
        else:
            assignments.append(f"{key} = ?")
            values.append(value)
    values.append(job_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
            tuple(values),
        )
        conn.commit()
    return get_job(job_id)
