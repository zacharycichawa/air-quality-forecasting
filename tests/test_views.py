"""The daily_station materialized view: local-day assignment and validity rule."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.db import refresh_views, upsert_measurements

pytestmark = pytest.mark.integration


def _hours(start_utc: str, n: int, value: float = 10.0) -> pd.DataFrame:
    ts = pd.date_range(start_utc, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"station_code": "StA", "pollutant": "PM2.5", "ts_utc": ts, "value": value,
         "source": "archive_xlsx"}
    )


def _daily(engine) -> pd.DataFrame:
    refresh_views(engine)
    return pd.read_sql("SELECT day, n_hours, is_valid, mean_value FROM daily_station "
                       "ORDER BY day", engine)


@pytest.fixture
def station(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO stations (station_code, name) VALUES ('StA', 'A')")
    return engine


def test_winter_day_is_cet_hours_ending_01_to_24(station):
    # Local day 2024-01-15 (CET = UTC+1): hours ending 01:00..24:00 local
    # = ts_utc 2024-01-15 00:00 .. 2024-01-15 23:00.
    upsert_measurements(station, _hours("2024-01-15 00:00", 24))
    daily = _daily(station)
    assert daily["day"].astype(str).tolist() == ["2024-01-15"]
    assert daily["n_hours"].tolist() == [24]


def test_summer_day_uses_cest(station):
    # Local day 2024-07-15 (CEST = UTC+2) = hours starting 2024-07-14 22:00 UTC
    # = ts_utc (end of hour) 2024-07-14 23:00 .. 2024-07-15 22:00.
    upsert_measurements(station, _hours("2024-07-14 23:00", 24))
    daily = _daily(station)
    assert daily["day"].astype(str).tolist() == ["2024-07-15"]


def test_dst_spring_forward_day_has_23_hours(station):
    # 2024-03-31: clocks jump 02:00 -> 03:00 local, so that local day has 23 hours.
    # Local midnight starts at 2024-03-30 23:00 UTC; next local midnight is 22:00 UTC.
    upsert_measurements(station, _hours("2024-03-31 00:00", 23))  # ends of hours 23:00..21:00
    daily = _daily(station)
    assert daily["day"].astype(str).tolist() == ["2024-03-31"]
    assert daily["n_hours"].tolist() == [23]


def test_validity_requires_18_hours(station):
    upsert_measurements(station, _hours("2024-01-15 00:00", 17))
    upsert_measurements(station, _hours("2024-01-16 00:00", 18))
    daily = _daily(station)
    assert daily["is_valid"].tolist() == [False, True]


def test_null_hours_do_not_count(station):
    df = _hours("2024-01-15 00:00", 24)
    df.loc[:9, "value"] = None  # 10 missing hours -> 14 valid
    upsert_measurements(station, df)
    daily = _daily(station)
    assert daily["n_hours"].tolist() == [14]
    assert not daily["is_valid"].iloc[0]
