"""
Database Admin Route Controller.

This module provides a read-only, HTML-based database browser for inspecting SQLite tables
contained in `control_center.db`. It uses Basic HTTP Auth to verify administrative credentials
and renders schema layouts and tabular row content for debug purposes.

Key Endpoints:
  - `GET /admin/db`: Renders the index panel listing all SQLite tables.
  - `GET /admin/db/tables/{table}`: Paginated view displaying the selected table schema and rows.

Connections:
  - Imports: DB credentials checking via `authenticate_user` from `storage.jobs_db`.
  - Guards: `require_db_admin` enforces Basic authentication.
"""

from __future__ import annotations

import html
import os
import secrets
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from services.app_paths import BACKEND_DIR
from storage.jobs_db import authenticate_user


router = APIRouter(prefix="/admin/db", tags=["admin"])
security = HTTPBasic()

DB_PATH = BACKEND_DIR / "storage" / "control_center.db"
PAGE_SIZE_MAX = 500


def _admin_password() -> str:
    """Retrieve the db admin password from the environment."""
    return os.environ.get("CONTROL_CENTER_DB_PASSWORD") or os.environ.get("DB_ADMIN_PASSWORD") or ""


def _admin_username() -> str:
    """Retrieve the db admin username from the environment."""
    return os.environ.get("CONTROL_CENTER_DB_USER") or "admin"


def require_db_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Require database administrator credentials via Basic Auth."""
    user = authenticate_user(credentials.username, credentials.password)
    if user and user.get("role") in {"admin", "guest"}:
        return str(user["username"])

    password = _admin_password()
    env_fallback_enabled = bool(password)
    username_ok = secrets.compare_digest(credentials.username, _admin_username())
    password_ok = secrets.compare_digest(credentials.password, password) if env_fallback_enabled else False
    if not (env_fallback_enabled and username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid database admin credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _connect() -> sqlite3.Connection:
    """Create a read-only sqlite3 connection to the control center database."""
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _database_exists() -> None:
    """Ensure the control center database file exists on disk."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Database file not found: {DB_PATH}")


def _tables(connection: sqlite3.Connection) -> list[str]:
    """List all user-defined table names in the database."""
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _require_table(connection: sqlite3.Connection, table: str) -> str:
    """Verify that a table exists, raising 404 if missing."""
    if table not in _tables(connection):
        raise HTTPException(status_code=404, detail=f"Table not found: {table}")
    return table


def _render_page(title: str, body: str) -> HTMLResponse:
    """Render a basic HTML dashboard page with responsive styling."""
    safe_title = html.escape(title)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{ margin: 0; background: #f7f8fb; color: #172033; }}
    header {{ background: #172033; color: white; padding: 18px 24px; }}
    header h1 {{ font-size: 20px; margin: 0; font-weight: 700; }}
    main {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
    a {{ color: #1358d8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .panel {{ background: white; border: 1px solid #dce1eb; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .muted {{ color: #667085; }}
    .table-list {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ border: 1px solid #c9d3e5; border-radius: 999px; padding: 6px 10px; background: #f3f6fb; }}
    table {{ width: 100%; border-collapse: collapse; background: white; font-size: 13px; }}
    th, td {{ border: 1px solid #dce1eb; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f8; position: sticky; top: 0; z-index: 1; }}
    td {{ max-width: 420px; overflow-wrap: anywhere; }}
    code {{ background: #eef2f8; border-radius: 4px; padding: 2px 5px; }}
    .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #0f1724; color: #e8eef9; }}
      .panel, table {{ background: #151f2e; border-color: #2b3a50; }}
      th, td {{ border-color: #2b3a50; }}
      th, code, .pill {{ background: #202d40; }}
      .muted {{ color: #9aa7ba; }}
    }}
  </style>
</head>
<body>
  <header><h1>{safe_title}</h1></header>
  <main>{body}</main>
</body>
</html>"""
    )


@router.get("", response_class=HTMLResponse)
def db_home(_: str = Depends(require_db_admin)):
    """Render the main database schema summary page."""
    _database_exists()
    with _connect() as connection:
        tables = _tables(connection)
        table_links = "\n".join(
            f'<a class="pill" href="/admin/db/tables/{html.escape(table)}">{html.escape(table)}</a>'
            for table in tables
        )
    body = f"""
    <section class="panel">
      <p class="muted">Read-only SQLite browser for <code>{html.escape(str(DB_PATH))}</code>.</p>
      <div class="table-list">{table_links or '<span class="muted">No tables found.</span>'}</div>
    </section>
    """
    return _render_page("Database Admin", body)


@router.get("/tables/{table}", response_class=HTMLResponse)
def db_table(
    table: str,
    _: str = Depends(require_db_admin),
    limit: int = Query(100, ge=1, le=PAGE_SIZE_MAX),
    offset: int = Query(0, ge=0),
):
    """Render a paginated HTML data browser table view for a specific SQLite database table."""

    _database_exists()
    with _connect() as connection:
        table = _require_table(connection, table)
        columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        schema_rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchall()
        total = connection.execute(f'SELECT COUNT(*) AS count FROM "{table}"').fetchone()["count"]
        rows = connection.execute(
            f'SELECT * FROM "{table}" LIMIT ? OFFSET ?',
            (limit, offset),
        ).fetchall()

    column_names = [str(column["name"]) for column in columns]
    header = "".join(f"<th>{html.escape(name)}</th>" for name in column_names)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape('' if row[name] is None else str(row[name]))}</td>"
            for name in column_names
        )
        body_rows.append(f"<tr>{cells}</tr>")

    previous_offset = max(0, offset - limit)
    next_offset = offset + limit
    previous_link = (
        f'<a href="/admin/db/tables/{html.escape(table)}?limit={limit}&offset={previous_offset}">Previous</a>'
        if offset > 0
        else '<span class="muted">Previous</span>'
    )
    next_link = (
        f'<a href="/admin/db/tables/{html.escape(table)}?limit={limit}&offset={next_offset}">Next</a>'
        if next_offset < total
        else '<span class="muted">Next</span>'
    )
    schema_sql = html.escape(schema_rows[0]["sql"] if schema_rows else "")

    page_body = f"""
    <section class="panel">
      <div class="controls">
        <a href="/admin/db">All tables</a>
        <span class="muted">Table <code>{html.escape(table)}</code></span>
        <span class="muted">{int(total)} rows</span>
        {previous_link}
        {next_link}
      </div>
    </section>
    <section class="panel">
      <h2>Schema</h2>
      <pre><code>{schema_sql}</code></pre>
    </section>
    <section class="panel">
      <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows) or f'<tr><td colspan="{max(len(column_names), 1)}" class="muted">No rows.</td></tr>'}</tbody>
      </table>
    </section>
    """
    return _render_page(f"Database Admin: {table}", page_body)
