"""Splits, baselines and metrics of the forecasting module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.forecast.models import (
    Split,
    TargetTransform,
    episode_metrics,
    evaluate,
    fit_climatology,
    point_metrics,
    predict_baselines,
)


def frame(horizon: int = 24) -> pd.DataFrame:
    issue = pd.date_range("2023-12-25", "2025-01-05", freq="h", tz="UTC")
    return pd.DataFrame({"issue_ts": issue, "target_ts": issue + pd.Timedelta(hours=horizon)})


@pytest.mark.parametrize("horizon", [24, 48])
def test_splits_are_disjoint_and_respect_time(horizon):
    f = frame(horizon)
    m = Split(val_start="2024-01-01", test_start="2024-07-01", test_end="2025-01-01").masks(f)
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    # Training targets end before validation begins; nothing in val/test is issued earlier.
    assert f.loc[m["train"], "target_ts"].max() < pd.Timestamp("2024-01-01", tz="UTC")
    assert f.loc[m["val"], "issue_ts"].min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert f.loc[m["test"], "target_ts"].max() < pd.Timestamp("2025-01-01", tz="UTC")
    # Rows straddling a boundary belong to no set.
    straddle = (f["issue_ts"] < pd.Timestamp("2024-01-01", tz="UTC")) & (
        f["target_ts"] >= pd.Timestamp("2024-01-01", tz="UTC"))
    assert not (m["train"] | m["val"] | m["test"])[straddle].any()


@pytest.mark.parametrize("kind", ["log", "raw"])
def test_target_transform_round_trips(kind):
    tf = TargetTransform(kind)
    y = np.array([0.0, 1.5, 20.0, 400.0])
    assert np.allclose(tf.inverse(tf.forward(y)), y)
    assert (tf.inverse(np.array([-5.0])) >= 0).all()  # predictions clipped at 0


def test_point_and_episode_metrics():
    y = np.array([10.0, 60.0, 80.0, 20.0])
    yhat = np.array([12.0, 55.0, 40.0, 70.0])
    m = point_metrics(y, yhat)
    assert m["mae"] == pytest.approx((2 + 5 + 40 + 50) / 4)
    assert m["bias"] == pytest.approx((2 - 5 - 40 + 50) / 4)
    e = episode_metrics(y, yhat, thr=50)  # actual: idx 1,2; predicted: idx 1,3
    assert e["precision"] == pytest.approx(0.5) and e["recall"] == pytest.approx(0.5)
    assert e["positives"] == 2


def test_climatology_baseline_uses_training_means_with_fallback():
    train = pd.DataFrame({"station_code": ["A", "A"], "month": [1, 1], "hour": [8, 8],
                          "target": [10.0, 30.0]})
    clim = fit_climatology(train)
    f = pd.DataFrame({"station_code": ["A", "B"], "month": [1, 1], "hour": [8, 8],
                      "pm25_lag0": [5.0, 6.0], "pm25_mean24": [7.0, 9.0]})
    b = predict_baselines(f, clim)
    assert b["climatology"].tolist() == [20.0, 9.0]  # B unseen -> falls back to 24 h mean
    assert b["persistence"].tolist() == [5.0, 6.0]


def test_evaluate_reports_skill_relative_to_persistence():
    f = pd.DataFrame({"target": [10.0, 20.0], "month": [1, 7]})
    preds = pd.DataFrame({"persistence": [14.0, 24.0], "model": [12.0, 22.0]})
    table = evaluate(f, preds)
    assert table.loc["model", "skill_vs_persistence"] == pytest.approx(0.5)
    assert table.loc["model", "mae_heating"] == pytest.approx(2.0)


def test_baselines_never_return_nan():
    clim = pd.DataFrame(columns=["station_code", "month", "hour", "climatology"])
    f = pd.DataFrame({"station_code": ["A"], "month": [1], "hour": [8],
                      "pm25_lag0": [5.0], "pm25_mean24": [np.nan]})
    b = predict_baselines(f, clim.astype({"month": int, "hour": int, "climatology": float}))
    assert b.notna().all().all() and b.iloc[0].tolist() == [5.0, 5.0, 5.0]


def test_best_f1_threshold_and_classifier_metrics():
    from airquality.forecast.models import best_f1_threshold, classifier_metrics

    y = np.array([10.0, 20.0, 60.0, 70.0, 30.0, 80.0])
    proba = np.array([0.1, 0.2, 0.7, 0.9, 0.6, 0.8])
    cut = best_f1_threshold(y > 50, proba)
    assert cut == pytest.approx(0.7)  # separates the three episodes perfectly
    m = classifier_metrics(y, proba, cut)
    assert m["f1"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
