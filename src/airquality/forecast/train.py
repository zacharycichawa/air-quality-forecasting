"""End-to-end forecasting experiment for one horizon.

    features -> time split -> baselines -> LightGBM (choose target transform on val)
    -> evaluate on val and, once, on the held-out test year -> save artefacts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from airquality.config import PROJECT_ROOT
from airquality.forecast.data import load_hourly, load_stations, load_weather
from airquality.forecast.features import build_features, feature_columns
from airquality.forecast.models import (
    HIGH_POLLUTION,
    Split,
    TargetTransform,
    best_f1_threshold,
    classifier_metrics,
    evaluate,
    fit_climatology,
    predict_baselines,
    predict_lgbm,
    train_episode_classifier,
    train_lgbm,
)

log = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def experiment_name(horizon: int, use_weather: bool) -> str:
    return f"h{horizon}_{'weather' if use_weather else 'noweather'}"


def run_experiment(
    engine: Engine, horizon: int, use_weather: bool, split: Split | None = None,
    transforms: tuple[str, ...] = ("log", "raw"), refresh_cache: bool = False,
) -> dict:
    split = split or Split()
    name = experiment_name(horizon, use_weather)
    t0 = time.time()

    hourly = load_hourly(engine, refresh=refresh_cache)
    weather = load_weather(engine, refresh=refresh_cache) if use_weather else None
    feats = build_features(hourly, load_stations(engine), horizon, weather)
    del hourly, weather
    if use_weather:
        # Keep only rows where target-hour weather exists, so the comparison with the
        # no-weather model (scored on the same rows) is fair.
        feats = feats[feats["wx_temperature_2m"].notna()].reset_index(drop=True)
    cols = feature_columns(feats, use_weather)
    masks = split.masks(feats)
    log.info("%s: %d feature rows, %d features (%.0fs)", name, len(feats), len(cols),
             time.time() - t0)

    train, val, test = (feats[masks[k]] for k in ("train", "val", "test"))
    clim = fit_climatology(train)

    # Choose the target transform by validation MAE (test set untouched).
    candidates = {}
    for kind in transforms:
        tf = TargetTransform(kind)
        model = train_lgbm(train[cols], train["target"].to_numpy(),
                           val[cols], val["target"].to_numpy(), tf)
        val_mae = float((predict_lgbm(model, val[cols], tf) - val["target"]).abs().mean())
        log.info("%s transform=%s: best_iter=%d val MAE=%.3f", name, kind,
                 model.best_iteration, val_mae)
        candidates[kind] = (val_mae, model, tf)
    best_kind = min(candidates, key=lambda k: candidates[k][0])
    _, model, tf = candidates[best_kind]

    # High-pollution episodes: separate classifier, alert cut-off chosen on validation.
    clf = train_episode_classifier(train[cols], train["target"].to_numpy(),
                                   val[cols], val["target"].to_numpy())
    p_val = clf.predict(val[cols], num_iteration=clf.best_iteration)
    cutoff = best_f1_threshold(val["target"].to_numpy() > HIGH_POLLUTION, p_val)

    results, episodes = {}, {}
    for split_name, part in (("val", val), ("test", test)):
        preds = predict_baselines(part, clim)
        preds["lightgbm"] = predict_lgbm(model, part[cols], tf)
        results[split_name] = evaluate(part, preds)
        proba = clf.predict(part[cols], num_iteration=clf.best_iteration)
        episodes[split_name] = classifier_metrics(part["target"].to_numpy(), proba, cutoff)
        if split_name == "test":
            out = part[["station_code", "issue_ts", "target_ts", "target", "month"]].copy()
            out[preds.columns] = preds.to_numpy()
            out["p_episode"] = proba
            _save_predictions(out, name)
    log.info("%s episode classifier (cut-off %.2f): test %s", name, cutoff,
             {k: round(v, 3) for k, v in episodes["test"].items()})

    importance = pd.Series(model.feature_importance("gain"), index=cols).sort_values(
        ascending=False)
    summary = {
        "experiment": name,
        "horizon_hours": horizon,
        "use_weather": use_weather,
        "weather_note": ("target-hour weather = ERA5 reanalysis (perfect-forecast proxy)"
                         if use_weather else None),
        "split": split.__dict__,
        "rows": {k: int(m.sum()) for k, m in masks.items()},
        "features": cols,
        "target_transform": best_kind,
        "val_mae_by_transform": {k: v[0] for k, v in candidates.items()},
        "best_iteration": model.best_iteration,
        "metrics": {k: v.round(4).reset_index().to_dict("records") for k, v in results.items()},
        "episode_classifier": {"threshold_ug_m3": HIGH_POLLUTION, "cutoff": cutoff,
                               **{k: {m: round(x, 4) for m, x in v.items()}
                                  for k, v in episodes.items()}},
        "top_features_gain": importance.head(15).round(1).to_dict(),
        "runtime_s": round(time.time() - t0),
    }
    _save(model, summary, name)
    clf.save_model(str(MODEL_DIR / f"lgbm_{name}_episode.txt"), num_iteration=clf.best_iteration)
    log.info("%s done in %.0fs\n%s", name, time.time() - t0,
             results["test"][["mae", "rmse", "bias", "mae_heating", "skill_vs_persistence"]]
             .round(3).to_string())
    return summary


def _save_predictions(df: pd.DataFrame, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"test_predictions_{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def _save(model, summary: dict, name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / f"lgbm_{name}.txt"), num_iteration=model.best_iteration)
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str),
                                              encoding="utf-8")
