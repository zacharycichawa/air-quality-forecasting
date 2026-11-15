"""Feature engineering for direct h-hour-ahead PM2.5 forecasting.

One row = (station, issue time t). Target = PM2.5 at t + h.
Leakage rule: every feature uses data at or before t, except
  * calendar features of the target hour (known in advance), and
  * weather at the target hour (in production this comes from a weather forecast;
    in training we use ERA5 reanalysis as a proxy - an optimistic "perfect forecast").
Input must be on a complete hourly grid per station (see data.complete_hourly_grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airquality.forecast.data import WEATHER_VARS

LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 24, 72, 168]
TARGET_WEATHER = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure",
    "precipitation", "cloud_cover", "boundary_layer_height",
]
LOCAL_TZ = "Europe/Warsaw"


def _calendar(target_ts: pd.Series) -> pd.DataFrame:
    """Calendar of the target hour in local time (ts marks the END of the hour)."""
    local = (target_ts - pd.Timedelta(hours=1)).dt.tz_convert(LOCAL_TZ)
    # No day-of-year features: with station location they let the model memorise the
    # weather of specific training-year days (val MAE worse, early stop at ~60 trees).
    return pd.DataFrame({
        "hour": local.dt.hour,
        "dow": local.dt.day_of_week,
        "month": local.dt.month,
        "is_weekend": (local.dt.day_of_week >= 5).astype("int8"),
    }, index=target_ts.index)


def _history(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged and rolling pollution features, all ending at the issue time t."""
    g = df.groupby("station_code", observed=True, sort=False)
    out = {}
    pm = g["pm25"]
    for k in LAGS:
        out[f"pm25_lag{k}"] = pm.shift(k)
    for w in ROLL_WINDOWS:
        roll = pm.rolling(w, min_periods=max(1, w // 2))
        out[f"pm25_mean{w}"] = roll.mean().reset_index(level=0, drop=True)
    roll24 = pm.rolling(24, min_periods=12)
    out["pm25_std24"] = roll24.std().reset_index(level=0, drop=True)
    out["pm25_max24"] = roll24.max().reset_index(level=0, drop=True)
    out["pm25_n24"] = pm.rolling(24, min_periods=1).count().reset_index(level=0, drop=True)
    out["pm25_diff3"] = df["pm25"] - out["pm25_lag3"]
    out["pm25_diff24"] = df["pm25"] - out["pm25_lag24"]
    for col in ("pm10", "no2"):
        out[f"{col}_lag0"] = df[col]
        out[f"{col}_mean24"] = (
            g[col].rolling(24, min_periods=12).mean().reset_index(level=0, drop=True)
        )
    return pd.DataFrame(out, index=df.index)


def _weather(df: pd.DataFrame, weather: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Weather at the target hour t+h (forecast proxy) and at the issue time t."""
    w = weather.set_index(["station_code", "ts_utc"])
    key_issue = pd.MultiIndex.from_arrays([df["station_code"].astype(str), df["ts_utc"]])
    key_target = pd.MultiIndex.from_arrays(
        [df["station_code"].astype(str), df["ts_utc"] + pd.Timedelta(hours=horizon)]
    )
    at_target = w.reindex(key_target)
    at_issue = w.reindex(key_issue)
    out = {f"wx_{c}": at_target[c].to_numpy() for c in TARGET_WEATHER}
    rad = np.radians(at_target["wind_direction_10m"].to_numpy())
    out["wx_wind_u"] = (np.sin(rad) * at_target["wind_speed_10m"].to_numpy())
    out["wx_wind_v"] = (np.cos(rad) * at_target["wind_speed_10m"].to_numpy())
    # .to_numpy() on both sides: the two frames have different index keys, and Series
    # arithmetic would align on them (union) instead of subtracting row by row.
    out["wx_temp_change"] = (
        at_target["temperature_2m"].to_numpy() - at_issue["temperature_2m"].to_numpy()
    )
    out["wx_blh_issue"] = at_issue["boundary_layer_height"].to_numpy()
    return pd.DataFrame(out, index=df.index)


def build_features(
    hourly: pd.DataFrame,
    stations: pd.DataFrame,
    horizon: int,
    weather: pd.DataFrame | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Return one row per (station, issue time) with features, `target` and metadata.

    Rows are kept only where the current PM2.5 reading exists and, for training and
    evaluation (`require_target=True`), where the target exists too, so every model and
    baseline is scored on exactly the same rows. Use `require_target=False` for live
    forecasts, whose targets lie in the future.
    """
    df = hourly.sort_values(["station_code", "ts_utc"], kind="stable").reset_index(drop=True)
    target = df.groupby("station_code", observed=True, sort=False)["pm25"].shift(-horizon)
    target_ts = df["ts_utc"] + pd.Timedelta(hours=horizon)

    parts = [df[["station_code", "ts_utc"]].rename(columns={"ts_utc": "issue_ts"}),
             pd.DataFrame({"target_ts": target_ts, "target": target}),
             _history(df), _calendar(target_ts)]
    if weather is not None:
        parts.append(_weather(df, weather, horizon))
    feats = pd.concat(parts, axis=1)

    keep = feats["pm25_lag0"].notna()
    if require_target:
        keep &= feats["target"].notna()
    feats = feats[keep]
    loc = stations.set_index("station_code")[["latitude", "longitude"]]
    feats = feats.join(loc, on="station_code")
    num = feats.select_dtypes("number").columns
    feats[num] = feats[num].astype("float32")
    return feats.reset_index(drop=True)


META_COLS = ["station_code", "issue_ts", "target_ts", "target"]


def feature_columns(feats: pd.DataFrame, use_weather: bool) -> list[str]:
    cols = [c for c in feats.columns if c not in META_COLS]
    if not use_weather:
        cols = [c for c in cols if not c.startswith("wx_")]
    return cols


__all__ = ["build_features", "feature_columns", "META_COLS", "WEATHER_VARS"]
