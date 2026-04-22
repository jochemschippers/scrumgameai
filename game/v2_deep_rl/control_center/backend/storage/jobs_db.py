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
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
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
    """Hash a password using stdlib PBKDF2 so no extra auth dependency is needed."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
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


def _seed_user(connection: sqlite3.Connection, username: str, password: str, role: str) -> None:
    existing = connection.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if existing is not None:
        return

    now = utc_now_iso()
    connection.execute(
        """
        INSERT INTO users (username, password_hash, role, is_active, created_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?)
        """,
        (username, hash_password(password), role, now, now),
    )


def get_user_by_username(username: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def authenticate_user(username: str, password: str) -> dict | None:
    user = get_user_by_username(username)
    if not user or not int(user.get("is_active", 0)):
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def list_users() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, username, role, is_active, created_at, updated_at
            FROM users
            ORDER BY id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _row_to_job(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    payload = dict(row)
    payload["payload"] = json.loads(payload.pop("payload_json"))
    return payload


def create_job(job_type: str, payload: dict, stdout_log_path: str = "", run_dir: str = "", result_path: str = "") -> dict:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO jobs (
                job_type, status, payload_json, created_at, stdout_log_path, run_dir, result_path, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_type,
                "queued",
                json.dumps(payload),
                utc_now_iso(),
                stdout_log_path,
                run_dir,
                result_path,
                None,
            ),
        )
        connection.commit()
        return get_job(cursor.lastrowid)


def list_jobs() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    return [_row_to_job(row) for row in rows]


def get_job(job_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row)


def delete_job(job_id: int) -> bool:
    with get_connection() as connection:
        cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        connection.commit()
    return cursor.rowcount > 0


def update_job(job_id: int, **fields) -> dict | None:
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

    with get_connection() as connection:
        connection.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
            tuple(values),
        )
        connection.commit()
    return get_job(job_id)
