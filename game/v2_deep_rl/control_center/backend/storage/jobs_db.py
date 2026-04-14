from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3

from services.app_paths import BACKEND_DIR


DB_PATH = BACKEND_DIR / "storage" / "control_center.db"


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
        connection.commit()


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
