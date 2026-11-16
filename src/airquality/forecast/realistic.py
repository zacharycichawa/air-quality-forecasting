"""Realistic evaluation: score the weather model with *real archived forecasts*.

Training and the main test use ERA5 reanalysis for the target hour's weather, i.e. a
perfect weather forecast. Here the target-hour weather comes instead from Open-Meteo's
Previous Runs API: the forecast that was issued 1 day (24 h horizon) or 2 days (48 h
horizon) before the target hour, exactly what would have been available in production.

That archive has no boundary layer height, so this evaluation uses the weather model
trained without it (`--weather --no-blh`). Comparing that model on ERA5 vs on archived
forecasts, on identical rows, isolates the cost of weather-forecast error.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import httpx
import lightgbm as lgb
import pandas as pd
from sqlalchemy import Engine

from airquality.forecast.data import CACHE_DIR, load_hourly, load_stations, load_weather
from airquality.forecast.features import build_features
from airquality.forecast.models import (
    Split,
    TargetTransform,
    evaluate,
    fit_climatology,
    predict_baselines,
)
from airquality.forecast.train import METRICS_DIR, MODEL_DIR, experiment_name
from airquality.ingest.weather import (
    HTTP_TIMEOUT,
    _batches,
    get_json,
    grid_cell,
    read_locations,
)

log = logging.getLogger(__name__)
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEAD_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
             "surface_pressure", "precipitation", "cloud_cover"]
CELLS_PER_REQUEST = 5  # 5 cells x 365 days x 7 variables / 140 = ~91 weighted calls


def fetch_lead_forecasts(engine: Engine, lead_day: int, start: date, end: date) -> pd.DataFrame:
    """Archived forecasts issued `lead_day` days before each hour, per station (cached)."""
    path = CACHE_DIR / f"lead_forecast_day{lead_day}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    locs = read_locations(engine)
    cells: dict[tuple, list[str]] = {}
    for r in locs.itertuples():
        cells.setdefault(grid_cell(r.latitude, r.longitude), []).append(r.station_code)
    names = {f"{v}_previous_day{lead_day}": v for v in LEAD_VARS}
    params = {"hourly": ",".join(names), "start_date": str(start), "end_date": str(end),
              "timezone": "UTC", "wind_speed_unit": "ms"}
    parts = []
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        for batch in _batches(list(cells), CELLS_PER_REQUEST):
            objs, _ = get_json(client, PREVIOUS_RUNS_URL, {
                **params,
                "latitude": ",".join(f"{lat:g}" for lat, _ in batch),
                "longitude": ",".join(f"{lon:g}" for _, lon in batch)})
            for cell, obj in zip(batch, objs, strict=True):
                h = obj["hourly"]
                df = pd.DataFrame({base: h[api] for api, base in names.items()})
                df["ts_utc"] = pd.to_datetime(h["time"], utc=True)
                parts += [df.assign(station_code=code) for code in cells[cell]]
            log.info("lead-%d forecasts: %d / %d cells", lead_day, len(parts), len(locs))
    out = pd.concat(parts, ignore_index=True).astype({v: "float32" for v in LEAD_VARS})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def evaluate_realistic(engine: Engine, horizon: int, split: Split | None = None) -> dict:
    split = split or Split()
    name = experiment_name(horizon, use_weather=True, use_blh=False)
    meta = json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{name}.txt"))
    tf, cols = TargetTransform(meta["target_transform"]), meta["features"]

    start, end = date.fromisoformat(split.test_start), date.fromisoformat(split.test_end)
    lead = fetch_lead_forecasts(engine, horizon // 24, start, end)
    hourly, weather, stations = load_hourly(engine), load_weather(engine), load_stations(engine)

    oracle = build_features(hourly, stations, horizon, weather)
    real = build_features(hourly, stations, horizon, weather, target_weather=lead)
    test_o, test_r = (f[split.masks(f)["test"]] for f in (oracle, real))
    # Weather never changes which rows exist, so the two frames align row by row.
    assert (test_o[["station_code", "issue_ts"]].to_numpy()
            == test_r[["station_code", "issue_ts"]].to_numpy()).all()
    # Identical rows: those where both the reanalysis and the archived forecast exist.
    keep = (test_o["wx_temperature_2m"].notna() & test_r["wx_temperature_2m"].notna()).to_numpy()
    test_o, test_r = test_o[keep], test_r[keep]

    clim = fit_climatology(oracle[split.masks(oracle)["train"]])
    preds = predict_baselines(test_r, clim)
    preds["lightgbm_era5_weather"] = tf.inverse(model.predict(test_o[cols]))
    preds["lightgbm_forecast_weather"] = tf.inverse(model.predict(test_r[cols]))
    table = evaluate(test_r, preds)
    out = test_r[["station_code", "issue_ts", "target_ts", "target", "month"]].copy()
    out["lightgbm_forecast_weather"] = preds["lightgbm_forecast_weather"].to_numpy()
    out.to_parquet(MODEL_DIR / f"test_predictions_h{horizon}_realistic.parquet", index=False)
    summary = {
        "experiment": f"h{horizon}_realistic",
        "horizon_hours": horizon,
        "model": name,
        "note": (f"Same model (weather without boundary layer height) scored with ERA5 "
                 f"target-hour weather vs archived forecasts issued {horizon // 24} day(s) "
                 f"earlier (Open-Meteo Previous Runs API), on identical test rows."),
        "rows": int(keep.sum()),
        "metrics": table.round(4).reset_index().to_dict("records"),
    }
    (METRICS_DIR / f"h{horizon}_realistic.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("h%d realistic:\n%s", horizon,
             table[["mae", "rmse", "bias", "mae_heating", "skill_vs_persistence"]].round(3))
    return summary
