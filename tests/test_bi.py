"""BI star-schema views: they compile and the forecast fact joins actuals correctly."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.db import refresh_views, upsert_forecasts, upsert_measurements

pytestmark = pytest.mark.integration


def test_bi_views_join_forecasts_to_actuals(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO stations (station_code, name, city) "
                             "VALUES ('StA', 'A', 'Lodz')")
    target = pd.Timestamp("2025-01-15 12:00", tz="UTC")
    upsert_measurements(engine, pd.DataFrame({
        "station_code": ["StA"], "pollutant": ["PM2.5"], "ts_utc": [target],
        "value": [40.0], "source": ["archive_xlsx"]}))
    upsert_forecasts(engine, pd.DataFrame({
        "model": ["lightgbm"], "run_type": ["backtest"], "station_code": ["StA"],
        "issue_ts": [target - pd.Timedelta(hours=24)], "horizon_h": [24],
        "target_ts": [target], "predicted": [30.0], "p_episode": [0.2],
        "uses_weather": [False]}))
    refresh_views(engine)

    fc = pd.read_sql("SELECT * FROM bi.fact_forecast", engine)
    assert fc.loc[0, "actual_ug_m3"] == 40.0 and fc.loc[0, "abs_error"] == 10.0
    assert str(fc.loc[0, "target_local"]) == "2025-01-15 13:00:00"  # CET = UTC+1
    for view in ("dim_station", "dim_date", "dim_threshold", "fact_daily_air_quality"):
        pd.read_sql(f"SELECT * FROM bi.{view} LIMIT 5", engine)  # compiles and runs
    st = pd.read_sql("SELECT * FROM bi.dim_station", engine)
    assert st.loc[0, "city"] == "Lodz" and not st.loc[0, "is_mobile"]
