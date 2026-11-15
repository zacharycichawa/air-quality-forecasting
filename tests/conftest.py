"""Shared fixtures. DB tests use the `airquality_test` database, never `airquality`."""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, text

from airquality.db import get_engine, init_schema

TEST_DB = "airquality_test"
TABLES = "forecasts, weather_hourly, measurements, sensors, stations, ingestion_runs"


@pytest.fixture
def engine() -> Engine:
    eng = get_engine(TEST_DB)
    init_schema(eng)
    with eng.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    return eng
