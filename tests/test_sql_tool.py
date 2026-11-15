"""SQL guard of the assistant: validation + the read-only database role."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DBAPIError

from airquality.assistant.sql_tool import (
    UnsafeQuery,
    ensure_reader_role,
    run_select,
    validate_select,
)
from airquality.db import get_engine

TEST_DB = "airquality_test"


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "  with x as (select 1 as a) select a from x;",
    "SELECT city FROM bi.dim_station WHERE city = 'Set' -- trailing comment",
])
def test_valid_selects_pass(sql):
    assert validate_select(sql).lower().lstrip().startswith(("select", "with"))


@pytest.mark.parametrize("sql, reason", [
    ("DROP TABLE measurements", "only SELECT"),
    ("SELECT 1; DELETE FROM stations", "single statement"),
    ("WITH d AS (DELETE FROM stations RETURNING *) SELECT * FROM d", "DELETE"),
    ("SELECT pg_sleep(100)", "PG_SLEEP"),
    ("/* hi */ UPDATE stations SET name = 'x'", "only SELECT"),
    ("", "empty"),
])
def test_unsafe_queries_rejected(sql, reason):
    with pytest.raises(UnsafeQuery, match=reason):
        validate_select(sql)


@pytest.mark.integration
def test_reader_role_is_read_only_and_limited_to_bi(engine):
    ensure_reader_role(engine)
    ensure_reader_role(engine)  # idempotent
    reader = get_engine(TEST_DB, reader=True)
    assert run_select(reader, "SELECT count(*) FROM bi.dim_date")["rows"][0][0] > 0
    with reader.connect() as conn:  # bypass the validator: the database itself must refuse
        with pytest.raises(DBAPIError):
            conn.exec_driver_sql("SELECT * FROM public.measurements LIMIT 1")
    with reader.connect() as conn:
        with pytest.raises(DBAPIError):
            conn.exec_driver_sql("CREATE TABLE bi.hack (x int)")
