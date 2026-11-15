"""Read-only SQL access for the LLM assistant: defence in depth.

1. The database role `aq_reader` can only SELECT from the `bi` views, and its sessions
   are read-only with a statement timeout. This is the real guarantee.
2. Before sending, a query must be a single SELECT/WITH statement (cheap pre-check that
   gives the model a clear error to correct).
3. Results are capped at MAX_ROWS rows.
"""

from __future__ import annotations

import os
import re

from sqlalchemy import Engine, text

from airquality.config import READER_ROLE

MAX_ROWS = 200
STATEMENT_TIMEOUT_MS = 10_000
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|copy|call|do|"
    r"vacuum|analyze|refresh|set|reset|lock|listen|notify|pg_sleep|pg_read_file|dblink)\b",
    re.IGNORECASE,
)


class UnsafeQuery(ValueError):
    pass


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def validate_select(sql: str) -> str:
    """Return the cleaned query, or raise UnsafeQuery with a message for the model."""
    q = _strip_comments(sql).strip().rstrip(";").strip()
    if not q:
        raise UnsafeQuery("empty query")
    if ";" in q:
        raise UnsafeQuery("only a single statement is allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", q):
        raise UnsafeQuery("only SELECT queries (optionally starting with WITH) are allowed")
    # Ignore string literals so e.g. WHERE city = 'Set' doesn't trip the keyword check.
    if m := FORBIDDEN.search(re.sub(r"'(?:[^']|'')*'", "''", q)):
        raise UnsafeQuery(f"keyword not allowed: {m.group(0).upper()}")
    return q


def run_select(engine: Engine, sql: str) -> dict:
    """Execute a validated query; return columns, rows (<= MAX_ROWS) and a truncation flag."""
    q = validate_select(sql)
    with engine.connect() as conn:
        result = conn.execute(text(q))
        cols = list(result.keys())
        rows = result.fetchmany(MAX_ROWS + 1)
    truncated = len(rows) > MAX_ROWS
    return {"columns": cols, "rows": [list(r) for r in rows[:MAX_ROWS]], "truncated": truncated}


def ensure_reader_role(owner: Engine) -> None:
    """Create/refresh the read-only role (idempotent). Needs AQ_READER_PASSWORD in .env."""
    password = os.environ.get("AQ_READER_PASSWORD")
    if not password:
        raise RuntimeError("set AQ_READER_PASSWORD in .env first")
    db = owner.url.database
    with owner.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"),
                              {"r": READER_ROLE}).first()
        verb = "ALTER" if exists else "CREATE"
        # DDL can't take bound parameters, so let PostgreSQL quote the identifier and the
        # password itself (format %I / %L) and then run the resulting statement.
        ddl = conn.execute(
            text("SELECT format(CAST(:tpl AS text), CAST(:role AS text), CAST(:pw AS text))"),
            {"tpl": f"{verb} ROLE %I LOGIN PASSWORD %L", "role": READER_ROLE, "pw": password},
        ).scalar_one()
        conn.exec_driver_sql(ddl.replace("%", "%%"))
        for stmt in (
            f"REVOKE ALL ON SCHEMA public FROM {READER_ROLE}",
            f"GRANT CONNECT ON DATABASE {db} TO {READER_ROLE}",
            f"GRANT USAGE ON SCHEMA bi TO {READER_ROLE}",
            f"GRANT SELECT ON ALL TABLES IN SCHEMA bi TO {READER_ROLE}",
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA bi GRANT SELECT ON TABLES TO {READER_ROLE}",
            f"ALTER ROLE {READER_ROLE} SET default_transaction_read_only = on",
            f"ALTER ROLE {READER_ROLE} SET statement_timeout = {STATEMENT_TIMEOUT_MS}",
            f"ALTER ROLE {READER_ROLE} SET search_path = bi",
        ):
            conn.exec_driver_sql(stmt)
