"""Upsert precedence rules, against the test database."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.db import WEATHER_COLS, upsert_measurements, upsert_weather

TS = pd.Timestamp("2024-01-01 00:00", tz="UTC")

pytestmark = pytest.mark.integration


@pytest.fixture
def station(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO stations (station_code, name) VALUES ('StA', 'A')")


def meas(value: float, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_code": ["StA"],
            "pollutant": ["PM10"],
            "ts_utc": [TS],
            "value": [value],
            "source": [source],
        }
    )


def stored(engine) -> tuple[float, str]:
    with engine.connect() as conn:
        return conn.exec_driver_sql("SELECT value, source FROM measurements").one()


RANKS = ["api_live", "api_archival", "archive_xlsx"]


@pytest.mark.parametrize("old", RANKS)
@pytest.mark.parametrize("new", RANKS)
def test_measurement_precedence_matrix(engine, station, old, new):
    upsert_measurements(engine, meas(1.0, old))
    n = upsert_measurements(engine, meas(2.0, new))
    if RANKS.index(new) >= RANKS.index(old):  # equal or higher rank wins
        assert (n, stored(engine)) == (1, (2.0, new))
    else:  # lower rank never overwrites
        assert (n, stored(engine)) == (0, (1.0, old))


def test_identical_rows_update_nothing(engine, station):
    assert upsert_measurements(engine, meas(1.0, "api_archival")) == 1
    assert upsert_measurements(engine, meas(1.0, "api_archival")) == 0


def weather(source: str, temp: float) -> pd.DataFrame:
    row = dict.fromkeys(WEATHER_COLS, 0.0)
    row.update(station_code="StA", ts_utc=TS, temperature_2m=temp, source=source)
    return pd.DataFrame([row])


def test_forecast_never_overwrites_archive_weather(engine, station):
    upsert_weather(engine, weather("archive", 5.0))
    assert upsert_weather(engine, weather("forecast", 9.0)) == 0
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT temperature_2m, source FROM weather_hourly").one() == (
            5.0,
            "archive",
        )


def test_archive_overwrites_forecast_weather(engine, station):
    upsert_weather(engine, weather("forecast", 9.0))
    assert upsert_weather(engine, weather("archive", 5.0)) == 1
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT temperature_2m, source FROM weather_hourly").one() == (
            5.0,
            "archive",
        )
