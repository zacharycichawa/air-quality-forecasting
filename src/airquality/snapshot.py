"""Export a compact, self-contained data snapshot for the Streamlit demo app.

The app is deployed publicly and can't reach the local PostgreSQL, so it reads a few
small Parquet files (a few MB in total) committed under app/data/.
"""

from __future__ import annotations

import json
import logging
import shutil

import pandas as pd
from sqlalchemy import Engine, text

from airquality.config import PROJECT_ROOT

log = logging.getLogger(__name__)
SNAPSHOT_DIR = PROJECT_ROOT / "app" / "data"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

QUERIES = {
    # Stations that have valid PM2.5 days (the app's focus).
    "stations": """
        SELECT station_code, station_name, city, voivodeship, latitude, longitude, is_mobile
          FROM bi.dim_station s
         WHERE EXISTS (SELECT 1 FROM daily_station d WHERE d.station_code = s.station_code
                         AND d.pollutant = 'PM2.5' AND d.is_valid)
    """,
    "daily_pm25": """
        SELECT station_code, date, round(mean_ug_m3::numeric, 1)::float AS pm25
          FROM bi.fact_daily_air_quality WHERE pollutant = 'PM2.5'
    """,
    "monthly": """
        SELECT station_code, pollutant, date_trunc('month', date)::date AS month,
               round(avg(mean_ug_m3)::numeric, 2)::float AS mean_ug_m3,
               count(*) AS valid_days,
               sum(exceeds_who_daily::int) AS days_over_who
          FROM bi.fact_daily_air_quality GROUP BY 1, 2, 3
    """,
    "forecasts_live": """
        SELECT station_code, horizon_h, issue_local, target_local,
               round(predicted_ug_m3::numeric, 1)::float AS predicted, p_episode, uses_weather
          FROM bi.fact_forecast
         WHERE run_type = 'live' AND model = 'lightgbm'
           AND issue_local = (SELECT max(f2.issue_local) FROM bi.fact_forecast f2
                               WHERE f2.run_type = 'live'
                                 AND f2.station_code = fact_forecast.station_code)
    """,
    # Hourly backtest for the winter months only (for the episode explorer).
    "backtest_winter": """
        SELECT model, station_code, horizon_h, target_local,
               round(predicted_ug_m3::numeric, 1)::float AS predicted,
               round(actual_ug_m3::numeric, 1)::float AS actual
          FROM bi.fact_forecast
         WHERE run_type = 'backtest' AND target_local >= '2025-01-01'
           AND target_local < '2025-03-01'
    """,
    "backtest_monthly_mae": """
        SELECT model, horizon_h, extract(month FROM target_local)::int AS month,
               avg(abs_error) AS mae, count(*) AS hours
          FROM bi.fact_forecast WHERE run_type = 'backtest' AND actual_ug_m3 IS NOT NULL
         GROUP BY 1, 2, 3
    """,
}


def export_snapshot(engine: Engine) -> dict[str, int]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql(text(sql), engine)
        df.to_parquet(SNAPSHOT_DIR / f"{name}.parquet", index=False, compression="zstd")
        counts[name] = len(df)
        log.info("snapshot %s: %d rows", name, len(df))
    for path in METRICS_DIR.glob("h*.json"):
        shutil.copy(path, SNAPSHOT_DIR / path.name)
    meta = {"exported_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": counts}
    (SNAPSHOT_DIR / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))
    return counts
