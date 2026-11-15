"""`update` / `backfill-api` start-point and watermark logic (mocked API, test database)."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from airquality import cli
from airquality.db import upsert_measurements
from airquality.ingest.gios_api import BASE_URL, GiosClient

pytestmark = pytest.mark.integration

END = pd.Timestamp("2026-11-15 12:00", tz="UTC")
XLSX_END = pd.Timestamp("2025-12-31 23:00", tz="UTC")


@pytest.fixture
def db(engine):
    """Three sensors: 1 has a watermark, 2 has only xlsx data, 3 has nothing."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO stations (station_code, name) VALUES ('S1','1'), ('S2','2'), ('S3','3')"
        )
        conn.exec_driver_sql(
            "INSERT INTO sensors (gios_sensor_id, station_code, pollutant, loaded_until) VALUES "
            "(1,'S1','PM10','2026-06-01Z'), (2,'S2','PM10',NULL), (3,'S3','PM10',NULL)"
        )
    xlsx = pd.DataFrame(
        {
            "station_code": ["S2"],
            "pollutant": ["PM10"],
            "ts_utc": [XLSX_END],
            "value": [1.0],
            "source": ["archive_xlsx"],
        }
    )
    upsert_measurements(engine, xlsx)
    return engine


def starts(jobs: list[cli.SensorJob]) -> dict[int, pd.Timestamp]:
    return {j.sensor_id: j.start for j in jobs}


def mock_empty(*ids: int) -> None:
    body = {"Lista archiwalnych danych pomiarowych": [], "totalPages": 0}
    for i in ids:
        respx.get(f"{BASE_URL}/archivalData/getDataBySensor/{i}").mock(
            return_value=httpx.Response(200, json=body)
        )


def watermarks(engine) -> dict[int, pd.Timestamp | None]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("SELECT gios_sensor_id, loaded_until FROM sensors").all()
    return {r[0]: (pd.Timestamp(r[1]).tz_convert("UTC") if r[1] else None) for r in rows}


def test_update_start_points(db):
    got = starts(cli.plan_update(db, END))
    assert got[1] == pd.Timestamp("2026-05-29", tz="UTC")  # watermark - 3 days
    assert got[2] == XLSX_END - pd.Timedelta(days=3)  # no watermark: end of xlsx data
    assert got[3] == END - pd.Timedelta(days=33)  # neither: now - 30 days - 3 days


def test_update_does_not_skip_a_gap_after_xlsx_data(db):
    """Regression: a later API island must not hide Jan-Sep from `update`."""
    island = pd.DataFrame(
        {
            "station_code": ["S2"],
            "pollutant": ["PM10"],
            "ts_utc": [pd.Timestamp("2026-10-05", tz="UTC")],
            "value": [1.0],
            "source": ["api_archival"],
        }
    )
    upsert_measurements(db, island)
    assert starts(cli.plan_update(db, END))[2] < pd.Timestamp("2026-01-01", tz="UTC")


@respx.mock
def test_update_advances_watermark_only_for_successful_sensors(db):
    mock_empty(1, 2)
    respx.get(f"{BASE_URL}/archivalData/getDataBySensor/3").mock(return_value=httpx.Response(400))
    client = GiosClient(delay=0, backoff=0)
    _, failed = cli.load_sensors(db, client, cli.plan_update(db, END), END)
    assert failed == 1
    assert watermarks(db) == {1: END, 2: END, 3: None}


def test_backfill_advances_only_if_contiguous_with_existing_coverage(db):
    # Sensor 1 watermark 2026-06-01, sensor 2 xlsx end 2025-12-31 23:00, sensor 3 now-30d.
    # Hourly data: starting at the very next hour (Jan 1 00:00) is still contiguous.
    jobs = cli.plan_backfill(db, pd.Timestamp("2026-01-02", tz="UTC"), END)
    assert {j.sensor_id: j.advance for j in jobs} == {1: True, 2: False, 3: True}
    jobs = cli.plan_backfill(db, pd.Timestamp("2026-01-01", tz="UTC"), END)
    assert {j.sensor_id: j.advance for j in jobs} == {1: True, 2: True, 3: True}


@respx.mock
def test_backfill_island_leaves_watermark_alone(db):
    mock_empty(1, 2, 3)
    start = pd.Timestamp("2026-10-01", tz="UTC")
    jobs = cli.plan_backfill(db, start, END)
    cli.load_sensors(db, GiosClient(delay=0, backoff=0), jobs, END)
    # Start (Oct 1) is after the watermarks of sensors 1 (June) and 2 (xlsx end): not advanced.
    # Sensor 3 has no data, so its effective watermark is now-30d (Oct 16) >= start: advanced.
    assert watermarks(db) == {1: pd.Timestamp("2026-06-01", tz="UTC"), 2: None, 3: END}
