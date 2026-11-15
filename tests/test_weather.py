"""Tests for the Open-Meteo weather ingestion (no network: HTTP is mocked with respx)."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import httpx
import pandas as pd
import pytest
import respx
from sqlalchemy import text

from airquality.ingest import weather as w

# Two stations in the same 0.25 degree cell (Warsaw) and one in Krakow.
STATIONS = [
    ("MzWarszA", 52.23, 21.01),
    ("MzWarszB", 52.30, 20.95),
    ("MpKrakowC", 50.06, 19.94),
]


# ---------------------------------------------------------------- pure functions

def test_grid_cell_groups_nearby_points():
    assert w.grid_cell(52.23, 21.01) == w.grid_cell(52.30, 20.95) == (52.25, 21.0)
    assert w.grid_cell(50.06, 19.94) != w.grid_cell(52.23, 21.01)


def test_request_cost_matches_open_meteo_weighting():
    assert w.request_cost(1, 31) == pytest.approx(1.7714, abs=1e-3)   # measured live
    assert w.request_cost(3, 366) == pytest.approx(62.74, abs=0.01)   # measured live
    assert w.request_cost(1, 1) == 1.0


def test_date_runs_groups_and_splits():
    days = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 5)]
    assert w.date_runs(days) == [
        (date(2024, 1, 1), date(2024, 1, 2)), (date(2024, 1, 5), date(2024, 1, 5))]
    assert len(w.date_runs([date(2024, 1, 1) + pd.Timedelta(days=i) for i in range(10)], 4)) == 3


def test_plan_archive_skips_complete_days_and_batches_cells():
    cells = {(52.25, 21.0): ["A", "B"], (50.0, 20.0): ["C"]}
    start, end = date(2024, 1, 1), date(2024, 1, 10)
    every_day = {date(2024, 1, d) for d in range(1, 11)}
    # Cell 1: A complete, B missing Jan 10 -> only Jan 10. Cell 2: fully complete.
    complete = {"A": every_day, "B": every_day - {date(2024, 1, 10)}, "C": every_day}
    plan = w.plan_archive(cells, complete, start, end)
    assert plan == [(date(2024, 1, 10), date(2024, 1, 10), [(52.25, 21.0)])]

    fresh = w.plan_archive(cells, {}, start, end)   # nothing loaded: one batched request
    assert fresh == [(start, end, [(52.25, 21.0), (50.0, 20.0)])]


# ---------------------------------------------------------------- HTTP behaviour

def _payload(n_hours: int = 2) -> dict:
    times = [f"2024-01-01T{h:02d}:00" for h in range(n_hours)]
    return {"hourly": {"time": times, **{v: [1.0] * n_hours for v in w.VARIABLES}}}


def test_parse_location_returns_utc_timestamps():
    df = w.parse_location(_payload())
    assert str(df["ts_utc"].dt.tz) == "UTC"
    assert df["ts_utc"].iloc[1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")


@respx.mock
def test_get_json_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    route = respx.get(w.ARCHIVE_URL).mock(side_effect=[
        httpx.Response(429, json={"error": True, "reason": "Minutely API request limit exceeded"}),
        httpx.Response(200, json=_payload()),
    ])
    with httpx.Client() as client:
        objs, _ = w.get_json(client, w.ARCHIVE_URL, {})
    assert route.call_count == 2 and len(objs) == 1


@respx.mock
@pytest.mark.parametrize("reason", ["Hourly API request limit exceeded. Try again next hour.",
                                    "Daily API request limit exceeded. Try again tomorrow."])
def test_get_json_fails_fast_on_hourly_and_daily_limit(monkeypatch, reason):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    route = respx.get(w.ARCHIVE_URL).mock(
        return_value=httpx.Response(429, json={"reason": reason}))
    with httpx.Client() as client, pytest.raises(w.DailyLimitExceeded, match="rerun later"):
        w.get_json(client, w.ARCHIVE_URL, {})
    assert route.call_count == 1


# ---------------------------------------------------------------- integration

@pytest.fixture
def seeded_engine(engine):
    """conftest's clean `engine` plus stations, PM2.5 sensors and PM2.5 measurements."""
    with engine.begin() as conn:
        for i, (code, lat, lon) in enumerate(STATIONS, start=1):
            conn.execute(
                text("INSERT INTO stations (station_code, name, latitude, longitude) "
                     "VALUES (:c, :c, :la, :lo)"),
                {"c": code, "la": lat, "lo": lon})
            conn.execute(
                text("INSERT INTO sensors (gios_sensor_id, station_code, pollutant) "
                     "VALUES (:i, :c, 'PM2.5')"), {"i": i, "c": code})
            conn.execute(
                text("INSERT INTO measurements (station_code, pollutant, ts_utc, value, source,"
                     " source_rank) VALUES (:c, 'PM2.5', '2024-01-01T00:00Z', 10, "
                     "'archive_xlsx', 3)"), {"c": code})
        # A station without PM2.5 data must be ignored.
        conn.execute(text("INSERT INTO stations (station_code, name, latitude, longitude) "
                          "VALUES ('NoPm', 'NoPm', 51.0, 17.0)"))
    yield engine
    with engine.begin() as conn:  # conftest truncates before each test; also clean up after
        conn.execute(text("TRUNCATE weather_hourly, measurements, sensors, stations CASCADE"))


def _fake_open_meteo(request: httpx.Request) -> httpx.Response:
    """Return one hourly series per requested coordinate, covering the requested dates."""
    q = request.url.params
    n = len(q["latitude"].split(","))
    if "start_date" in q:
        hours = pd.date_range(q["start_date"], pd.Timestamp(q["end_date"]) + pd.Timedelta("23h"),
                              freq="h")
    else:
        hours = pd.date_range("2024-06-01", freq="h",
                              periods=24 * (int(q["past_days"]) + int(q["forecast_days"])))
    series = {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
                         **{v: [10.0] * len(hours) for v in w.VARIABLES}}}
    body = [series] * n
    return httpx.Response(200, content=json.dumps(body if n > 1 else body[0]))


@pytest.mark.integration
@respx.mock
def test_archive_dedupes_cells_and_is_resumable(seeded_engine):
    route = respx.get(w.ARCHIVE_URL).mock(side_effect=_fake_open_meteo)
    start, end = date(2024, 1, 1), date(2024, 1, 3)

    # 3 stations x 3 days x 24 h
    assert w.load_weather_archive(seeded_engine, start, end) == 3 * 72
    assert route.call_count == 1                                  # one request, 2 cells batched
    assert route.calls[0].request.url.params["latitude"] == "50,52.25"
    assert route.calls[0].request.url.params["models"] == "era5"

    with seeded_engine.connect() as conn:
        n, lo, hi, src = conn.execute(text(
            "SELECT count(*), min(ts_utc), max(ts_utc), min(source) FROM weather_hourly")).one()
    assert n == 216 and src == "archive"
    assert lo == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert hi == pd.Timestamp("2024-01-03 23:00", tz="UTC")

    assert w.load_weather_archive(seeded_engine, start, end) == 0        # nothing left to fetch
    assert route.call_count == 1

    # Extending the range only requests the new day.
    assert w.load_weather_archive(seeded_engine, start, date(2024, 1, 4)) == 3 * 24
    assert route.calls[1].request.url.params["start_date"] == "2024-01-04"


@pytest.mark.integration
@respx.mock
def test_forecast_does_not_overwrite_archive(seeded_engine):
    respx.get(w.ARCHIVE_URL).mock(side_effect=_fake_open_meteo)
    route = respx.get(w.FORECAST_URL).mock(side_effect=_fake_open_meteo)
    w.load_weather_archive(seeded_engine, date(2024, 6, 1), date(2024, 6, 1))

    rows = w.load_weather_forecast(seeded_engine, ["MzWarszA"])
    assert rows == 216              # the 24 archive hours are skipped by the upsert
    assert route.calls[0].request.url.params["forecast_days"] == "3"

    with seeded_engine.connect() as conn:
        counts = dict(conn.execute(text(
            "SELECT source, count(*) FROM weather_hourly WHERE station_code = 'MzWarszA' "
            "GROUP BY source")).all())
    assert counts == {"archive": 24, "forecast": 216}
    with seeded_engine.connect() as conn:
        status = conn.execute(text(
            "SELECT status FROM ingestion_runs WHERE job = 'weather_forecast' "
            "ORDER BY run_id DESC LIMIT 1")).scalar_one()
    assert status == "success"


@pytest.mark.integration
@respx.mock
def test_recent_end_is_clamped_and_all_null_hours_do_not_refetch(seeded_engine):
    def handler(request: httpx.Request) -> httpx.Response:
        resp = _fake_open_meteo(request)
        body = json.loads(resp.content)
        for obj in body if isinstance(body, list) else [body]:
            for v in w.VARIABLES:
                obj["hourly"][v][5] = None   # hour 05:00 has no data at all
        return httpx.Response(200, content=json.dumps(body))

    route = respx.get(w.ARCHIVE_URL).mock(side_effect=handler)
    latest = date.today() - timedelta(days=w.ERA5_LAG_DAYS)

    w.load_weather_archive(seeded_engine, latest - timedelta(days=1), date.today())
    assert route.call_count == 1
    assert route.calls[0].request.url.params["end_date"] == latest.isoformat()   # clamped
    with seeded_engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM weather_hourly")).scalar_one()
    assert n == 3 * 48                                                          # nulls kept

    assert w.load_weather_archive(seeded_engine, latest - timedelta(days=1), date.today()) == 0
    assert route.call_count == 1                                                # no re-request
