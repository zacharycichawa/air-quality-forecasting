"""Time-based splits, baselines, LightGBM training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

HIGH_POLLUTION = 50.0  # µg/m³ hourly PM2.5; project-chosen "high pollution hour" threshold


# ---------------------------------------------------------------- splits
@dataclass(frozen=True)
class Split:
    """Train on targets before `val_start`; validate on [val_start, test_start);
    test on [test_start, test_end). Rows whose issue and target straddle a boundary
    are dropped, so no target is shared between sets."""

    val_start: str = "2024-01-01"
    test_start: str = "2025-01-01"
    test_end: str = "2026-01-01"

    def masks(self, f: pd.DataFrame) -> dict[str, pd.Series]:
        ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
        v, t, e = ts(self.val_start), ts(self.test_start), ts(self.test_end)
        return {
            "train": f["target_ts"] < v,
            "val": (f["issue_ts"] >= v) & (f["target_ts"] < t),
            "test": (f["issue_ts"] >= t) & (f["target_ts"] < e),
        }


# ---------------------------------------------------------------- baselines
def fit_climatology(train: pd.DataFrame) -> pd.DataFrame:
    """Mean target by station x month x hour (training data only)."""
    return (train.groupby(["station_code", "month", "hour"], observed=True)["target"]
                 .mean().rename("climatology").reset_index())


def predict_baselines(f: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    clim = f[["station_code", "month", "hour"]].merge(
        climatology, on=["station_code", "month", "hour"], how="left")["climatology"]
    # Every baseline must predict every row (rows always have a current value):
    # the 24 h mean needs >= 12 valid hours, else fall back to the current value;
    # climatology cells unseen in training fall back to that 24 h mean.
    current = f["pm25_lag0"].to_numpy()
    mean_24h = np.where(np.isnan(f["pm25_mean24"].to_numpy()), current,
                        f["pm25_mean24"].to_numpy())
    clim = np.where(clim.isna().to_numpy(), mean_24h, clim.to_numpy())
    return pd.DataFrame(
        {"persistence": current, "mean_24h": mean_24h, "climatology": clim}, index=f.index
    )


# ---------------------------------------------------------------- LightGBM
# Chosen on the 2024 validation set (see README): small leaves + large leaf size
# generalise better across years than deep trees, which overfit specific episodes.
LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
    "seed": 42,
}


@dataclass
class TargetTransform:
    """`log` trains L2 on log1p(target) (EDA: skewed target); `raw` trains L1 on µg/m³."""

    kind: str = "log"

    @property
    def objective(self) -> str:
        return "l2" if self.kind == "log" else "l1"

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y) if self.kind == "log" else y

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(y), 0, None) if self.kind == "log" else np.clip(y, 0, None)


def train_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
    transform: TargetTransform, max_rounds: int = 3000, early_stopping: int = 100,
) -> lgb.Booster:
    params = {**LGB_PARAMS, "objective": transform.objective, "metric": "l1"}
    dtrain = lgb.Dataset(X_train, transform.forward(y_train), free_raw_data=True)
    dval = lgb.Dataset(X_val, transform.forward(y_val), reference=dtrain)
    return lgb.train(
        params, dtrain, num_boost_round=max_rounds, valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(200)],
    )


def predict_lgbm(model: lgb.Booster, X: pd.DataFrame, transform: TargetTransform) -> np.ndarray:
    return transform.inverse(model.predict(X, num_iteration=model.best_iteration))


# ---------------------------------------------------------------- episode classifier
def train_episode_classifier(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
    thr: float = HIGH_POLLUTION,
) -> lgb.Booster:
    """P(target > thr). A point forecast trained for low average error hedges toward the
    middle and rarely crosses the threshold, so episodes get their own model."""
    params = {**LGB_PARAMS, "objective": "binary", "metric": "average_precision"}
    dtrain = lgb.Dataset(X_train, (y_train > thr).astype(int))
    dval = lgb.Dataset(X_val, (y_val > thr).astype(int), reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Probability cut-off that maximises F1 (choose on validation data only)."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def classifier_metrics(y: np.ndarray, proba: np.ndarray, cutoff: float,
                       thr: float = HIGH_POLLUTION) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    actual = y > thr
    out = episode_metrics(actual.astype(float), (proba >= cutoff).astype(float), thr=0.5)
    out.update({"pr_auc": float(average_precision_score(actual, proba)),
                "roc_auc": float(roc_auc_score(actual, proba)), "cutoff": cutoff})
    return out


# ---------------------------------------------------------------- metrics
def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    err = yhat - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "n": int(len(y)),
    }


def episode_metrics(y: np.ndarray, yhat: np.ndarray, thr: float = HIGH_POLLUTION) -> dict:
    """Detection of high-pollution hours (target > thr) from the point forecast."""
    actual, pred = y > thr, yhat > thr
    tp = int(np.sum(actual & pred))
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(actual.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "positives": int(actual.sum())}


def evaluate(f: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Table of metrics per model: overall, heating season (Oct-Mar), and episodes."""
    y = f["target"].to_numpy()
    heating = f["month"].isin([10, 11, 12, 1, 2, 3]).to_numpy()
    rows = []
    for name in preds.columns:
        yhat = preds[name].to_numpy()
        overall = point_metrics(y, yhat)
        rows.append({
            "model": name, **overall,
            "mae_heating": point_metrics(y[heating], yhat[heating])["mae"],
            "mae_summer": point_metrics(y[~heating], yhat[~heating])["mae"],
            **{f"episode_{k}": v for k, v in episode_metrics(y, yhat).items()},
        })
    table = pd.DataFrame(rows).set_index("model")
    if "persistence" in table.index:
        table["skill_vs_persistence"] = 1 - table["mae"] / table.loc["persistence", "mae"]
    return table
