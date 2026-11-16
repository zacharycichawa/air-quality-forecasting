"""Publish forecasts to the `forecasts` table: live (from the latest data) and backtests."""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import Engine

from airquality.db import upsert_forecasts
from airquality.forecast.data import (
    WEATHER_VARS,
    complete_hourly_grid,
    load_stations,
    read_sql_fast,
)
from airquality.forecast.features import build_features, feature_columns
from airquality.forecast.models import TargetTransform
from airquality.forecast.train import METRICS_DIR, MODEL_DIR, experiment_name

log = logging.getLogger(__name__)

HISTORY_DAYS = 9          # enough for the longest feature (168 h lag / rolling window)
MAX_STALENESS_H = 6       # don't issue a forecast from data older than this


def _recent_hourly(engine: Engine, since: pd.Timestamp) -> pd.DataFrame:
    sql = f"""
        SELECT station_code, ts_utc,
               max(value) FILTER (WHERE pollutant = 'PM2.5') AS pm25,
               max(value) FILTER (WHERE pollutant = 'PM10')  AS pm10,
               max(value) FILTER (WHERE pollutant = 'NO2')   AS no2
          FROM measurements WHERE ts_utc >= '{since.isoformat()}'
         GROUP BY station_code, ts_utc
    """
    df = read_sql_fast(engine, sql)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df[df.groupby("station_code")["pm25"].transform(lambda s: s.notna().any())]
    cols = ["pm25", "pm10", "no2"]
    return complete_hourly_grid(df.astype({c: "float32" for c in cols}), cols)


def _recent_weather(engine: Engine, since: pd.Timestamp) -> pd.DataFrame:
    sql = (f"SELECT station_code, ts_utc, {', '.join(WEATHER_VARS)} FROM weather_hourly "
           f"WHERE ts_utc >= '{since.isoformat()}'")
    df = read_sql_fast(engine, sql)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _load_models(name: str):
    meta = json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    reg = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{name}.txt"))
    clf_path = MODEL_DIR / f"lgbm_{name}_episode.txt"
    clf = lgb.Booster(model_file=str(clf_path)) if clf_path.exists() else None
    return meta, reg, clf


def latest_issue_rows(feats: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Per station, the most recent hour with a reading, if it is fresh enough."""
    last = feats.loc[feats.groupby("station_code", observed=True)["issue_ts"].idxmax()]
    return last[last["issue_ts"] >= now - pd.Timedelta(hours=MAX_STALENESS_H)]


def predict_live(engine: Engine, horizons: tuple[int, ...] = (24, 48),
                 use_weather: bool = False, now: pd.Timestamp | None = None) -> int:
    """Issue live forecasts. The weather variant uses the model without boundary layer
    height: its accuracy with real (archived) weather forecasts was measured, see
    forecast/realistic.py, so the published numbers match what users get."""
    now = now or pd.Timestamp.now(tz="UTC").floor("h")
    since = now - pd.Timedelta(days=HISTORY_DAYS)
    hourly = _recent_hourly(engine, since)
    weather = _recent_weather(engine, since) if use_weather else None
    stations = load_stations(engine)
    total = 0
    for h in horizons:
        name = experiment_name(h, use_weather, use_blh=False)
        meta, reg, clf = _load_models(name)
        feats = build_features(hourly, stations, h, weather, require_target=False)
        rows = latest_issue_rows(feats, now)
        cols = meta["features"]
        expected = feature_columns(feats, use_weather, use_blh=False)
        assert cols == expected, "feature set changed since training"
        tf = TargetTransform(meta["target_transform"])
        out = pd.DataFrame({
            "model": "lightgbm", "run_type": "live",
            "station_code": rows["station_code"].astype(str).to_numpy(),
            "issue_ts": rows["issue_ts"].to_numpy(), "horizon_h": h,
            "target_ts": rows["target_ts"].to_numpy(),
            "predicted": tf.inverse(reg.predict(rows[cols])),
            "p_episode": clf.predict(rows[cols]) if clf is not None else np.nan,
            "uses_weather": use_weather,
        })
        for c in ("issue_ts", "target_ts"):
            out[c] = pd.to_datetime(out[c], utc=True)
        total += upsert_forecasts(engine, out)
        log.info("%s: %d live forecasts issued (latest issue %s)", name, len(out),
                 rows["issue_ts"].max())
    return total


def load_backtests(engine: Engine, use_weather: bool = False,
                   horizons: tuple[int, ...] = (24, 48)) -> int:
    """Copy saved test-year predictions into `forecasts`.

    No weather: LightGBM + persistence. Weather: the realistic evaluation's predictions
    (model 'lightgbm_weather', fed with archived weather forecasts, not reanalysis).
    """
    if use_weather:
        return _load_realistic_backtests(engine, horizons)
    total = 0
    for h in horizons:
        name = experiment_name(h, use_weather)
        pred = pd.read_parquet(MODEL_DIR / f"test_predictions_{name}.parquet")
        for model, col in (("lightgbm", "lightgbm"), ("persistence", "persistence")):
            out = pd.DataFrame({
                "model": model, "run_type": "backtest",
                "station_code": pred["station_code"].astype(str),
                "issue_ts": pred["issue_ts"], "horizon_h": h, "target_ts": pred["target_ts"],
                "predicted": pred[col].astype(float),
                "p_episode": pred["p_episode"] if model == "lightgbm" and "p_episode" in pred
                else np.nan,
                "uses_weather": use_weather and model == "lightgbm",
            })
            total += upsert_forecasts(engine, out)
        log.info("%s: backtest predictions loaded", name)
    return total


def _load_realistic_backtests(engine: Engine, horizons: tuple[int, ...]) -> int:
    total = 0
    for h in horizons:
        pred = pd.read_parquet(MODEL_DIR / f"test_predictions_h{h}_realistic.parquet")
        out = pd.DataFrame({
            "model": "lightgbm_weather", "run_type": "backtest",
            "station_code": pred["station_code"].astype(str),
            "issue_ts": pred["issue_ts"], "horizon_h": h, "target_ts": pred["target_ts"],
            "predicted": pred["lightgbm_forecast_weather"].astype(float),
            "p_episode": np.nan, "uses_weather": True,
        })
        total += upsert_forecasts(engine, out)
        log.info("h%d realistic backtest predictions loaded", h)
    return total
