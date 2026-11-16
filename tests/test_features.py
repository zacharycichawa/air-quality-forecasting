"""Feature engineering: target alignment and no look-ahead leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.forecast.data import WEATHER_VARS, complete_hourly_grid
from airquality.forecast.features import build_features, feature_columns

T0 = pd.Timestamp("2024-01-10 00:00", tz="UTC")


def hourly_two_stations(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range(T0, periods=n, freq="h")
    a = pd.DataFrame({"station_code": "A", "ts_utc": ts, "pm25": np.arange(n, dtype=float)})
    b = pd.DataFrame({"station_code": "B", "ts_utc": ts, "pm25": 1000 + np.arange(n, dtype=float)})
    df = pd.concat([a, b], ignore_index=True).assign(pm10=1.0, no2=2.0)
    return complete_hourly_grid(df, ["pm25", "pm10", "no2"])


STATIONS = pd.DataFrame({"station_code": ["A", "B"], "latitude": [50.0, 52.0],
                         "longitude": [19.0, 21.0]})


@pytest.mark.parametrize("horizon", [24, 48])
def test_target_is_value_h_hours_after_issue(horizon):
    f = build_features(hourly_two_stations(), STATIONS, horizon)
    # In station A, pm25 equals the hour index, so target - lag0 == horizon exactly.
    a = f[f.station_code == "A"]
    assert (a["target"] - a["pm25_lag0"] == horizon).all()
    assert ((a["target_ts"] - a["issue_ts"]) == pd.Timedelta(hours=horizon)).all()


def test_lags_never_cross_stations():
    f = build_features(hourly_two_stations(), STATIONS, 24)
    b = f[f.station_code == "B"]
    # Station B values are all >= 1000; a lag leaking from A would be < 1000.
    assert (b["pm25_lag168"].dropna() >= 1000).all()
    assert b["pm25_lag168"].isna().sum() == 168  # first week of B has no lag-168 history


def test_rolling_features_use_only_the_past():
    hourly = hourly_two_stations()
    base = build_features(hourly, STATIONS, 24)
    # Corrupt everything after issue time t* in station A; features at t* must not change.
    t_star = T0 + pd.Timedelta(hours=200)
    corrupted = hourly.copy()
    mask = (corrupted.station_code == "A") & (corrupted.ts_utc > t_star)
    corrupted.loc[mask, ["pm25", "pm10", "no2"]] = 9999.0
    after = build_features(corrupted, STATIONS, 24)
    cols = [c for c in feature_columns(base, use_weather=False)]
    row = lambda f: f[(f.station_code == "A") & (f.issue_ts == t_star)][cols]  # noqa: E731
    pd.testing.assert_frame_equal(row(base).reset_index(drop=True),
                                  row(after).reset_index(drop=True))


def test_rows_require_target_and_current_value():
    hourly = hourly_two_stations()
    hourly.loc[(hourly.station_code == "A") & (hourly.ts_utc == T0 + pd.Timedelta(hours=50)),
               "pm25"] = np.nan
    f = build_features(hourly, STATIONS, 24)
    a = f[f.station_code == "A"]
    assert (T0 + pd.Timedelta(hours=50)) not in set(a["issue_ts"])   # no current value
    assert (T0 + pd.Timedelta(hours=26)) not in set(a["issue_ts"])   # target missing
    assert f[["target", "pm25_lag0"]].notna().all().all()


def test_calendar_is_local_time_of_target_hour():
    f = build_features(hourly_two_stations(), STATIONS, 24)
    # target_ts 2024-01-11 12:00Z ends the hour 12-13 CET -> local start hour 12.
    row = f[(f.station_code == "A") & (f.target_ts == pd.Timestamp("2024-01-11 12:00", tz="UTC"))]
    assert row["hour"].item() == 12


def test_weather_is_taken_at_target_hour():
    hourly = hourly_two_stations(100)
    ts = pd.date_range(T0, periods=200, freq="h")
    weather = pd.DataFrame({"station_code": "A", "ts_utc": ts})
    for v in WEATHER_VARS:
        weather[v] = 0.0
    weather["temperature_2m"] = np.arange(200, dtype=float)  # temperature = hour index
    f = build_features(hourly, STATIONS, 24, weather=weather)
    a = f[f.station_code == "A"].dropna(subset=["wx_temperature_2m"])
    hours_after_t0 = (a["issue_ts"] - T0) / pd.Timedelta(hours=1)
    assert np.allclose(a["wx_temperature_2m"], hours_after_t0 + 24)
    assert np.allclose(a["wx_temp_change"], 24)
    assert f[f.station_code == "B"]["wx_temperature_2m"].isna().all()  # no weather for B
    assert "wx_temperature_2m" not in feature_columns(f, use_weather=False)


def test_inference_mode_keeps_rows_without_target():
    hourly = hourly_two_stations(200)
    train_rows = build_features(hourly, STATIONS, 24)
    live_rows = build_features(hourly, STATIONS, 24, require_target=False)
    last_issue = hourly.ts_utc.max()
    assert last_issue not in set(train_rows.issue_ts)   # target lies beyond the data
    assert last_issue in set(live_rows.issue_ts)
    assert live_rows[live_rows.issue_ts == last_issue]["target"].isna().all()


def test_latest_issue_rows_skips_stale_stations():
    from airquality.forecast.predict import latest_issue_rows

    hourly = hourly_two_stations(200)
    hourly.loc[(hourly.station_code == "B") & (hourly.ts_utc > T0 + pd.Timedelta(hours=150)),
               "pm25"] = np.nan  # station B stopped reporting 49 h before the end
    f = build_features(hourly, STATIONS, 24, require_target=False)
    rows = latest_issue_rows(f, now=hourly.ts_utc.max())
    assert rows["station_code"].astype(str).tolist() == ["A"]
    assert rows["issue_ts"].item() == hourly.ts_utc.max()


def test_target_weather_source_overrides_only_the_target_hour():
    hourly = hourly_two_stations(100)
    ts = pd.date_range(T0, periods=200, freq="h")
    era5 = pd.DataFrame({"station_code": "A", "ts_utc": ts})
    for v in WEATHER_VARS:
        era5[v] = 1.0
    forecast = era5[["station_code", "ts_utc", "temperature_2m"]].assign(temperature_2m=5.0)
    f = build_features(hourly, STATIONS, 24, weather=era5, target_weather=forecast)
    a = f[f.station_code == "A"].dropna(subset=["wx_temperature_2m"])
    assert (a["wx_temperature_2m"] == 5.0).all()          # target hour from the forecast
    assert (a["wx_temp_change"] == 4.0).all()             # forecast(target) - ERA5(issue)
    assert a["wx_boundary_layer_height"].isna().all()     # not in the forecast source
    assert (a["wx_blh_issue"] == 1.0).all()               # issue time still from ERA5


def test_feature_columns_can_drop_boundary_layer_height():
    f = build_features(hourly_two_stations(100), STATIONS, 24, weather=pd.DataFrame(
        {"station_code": ["A"], "ts_utc": [T0], **{v: [1.0] for v in WEATHER_VARS}}))
    with_blh = feature_columns(f, use_weather=True)
    without = feature_columns(f, use_weather=True, use_blh=False)
    assert set(with_blh) - set(without) == {"wx_boundary_layer_height", "wx_blh_issue"}
