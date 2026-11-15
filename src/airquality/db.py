"""Database access: engine, schema setup, idempotent bulk upserts, run logging."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from airquality.config import SQL_DIR, db_settings

SOURCE_RANK = {"api_live": 1, "api_archival": 2, "archive_xlsx": 3}

MEASUREMENT_COLS = ["station_code", "pollutant", "ts_utc", "value", "source"]
WEATHER_COLS = [
    "station_code",
    "ts_utc",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "boundary_layer_height",
    "source",
]


@lru_cache(maxsize=8)
def get_engine(database: str | None = None, reader: bool = False) -> Engine:
    """Engine for the owner role, or (`reader=True`) the assistant's read-only role."""
    # Pin the session time zone to UTC: otherwise extract()/date_trunc() on timestamptz
    # use the server's local zone and e.g. 2025-12-31 23:00Z would count as year 2026.
    return create_engine(
        db_settings(database, reader=reader).url,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=UTC"},
    )


MATERIALIZED_VIEWS = ("daily_station",)


def init_schema(engine: Engine) -> None:
    """Create tables (schema.sql), analytical views (views.sql) and BI layer (bi.sql)."""
    with engine.begin() as conn:
        raw = conn.connection.driver_connection  # no parameters -> '%' in SQL needs no escaping
        for name in ("schema.sql", "views.sql", "bi.sql"):
            raw.execute((SQL_DIR / name).read_text(encoding="utf-8"))


def refresh_views(engine: Engine) -> None:
    """Refresh materialized views; CONCURRENTLY (no read lock) once they hold data."""
    with engine.begin() as conn:
        for view in MATERIALIZED_VIEWS:
            populated = conn.execute(
                text("SELECT ispopulated FROM pg_matviews WHERE matviewname = :v"), {"v": view}
            ).scalar_one()
            mode = "CONCURRENTLY " if populated else ""
            conn.exec_driver_sql(f"REFRESH MATERIALIZED VIEW {mode}{view}")


def _require_utc(df: pd.DataFrame) -> None:
    ts = df["ts_utc"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
        raise ValueError("ts_utc must be a tz-aware pandas datetime column in UTC")


def _copy_to_temp(conn, temp: str, like: str, df: pd.DataFrame, cols: list[str]) -> None:
    """Fast bulk load of a DataFrame into a session temp table via COPY."""
    conn.exec_driver_sql(
        f"CREATE TEMP TABLE {temp} (LIKE {like} INCLUDING DEFAULTS) ON COMMIT DROP"
    )
    raw = conn.connection.driver_connection  # psycopg3 connection
    with raw.cursor() as cur, cur.copy(f"COPY {temp} ({', '.join(cols)}) FROM STDIN") as copy:
        for row in df[cols].itertuples(index=False, name=None):
            copy.write_row([None if pd.isna(v) else v for v in row])


def upsert_measurements(engine: Engine, df: pd.DataFrame) -> int:
    """Upsert rows (station_code, pollutant, ts_utc, value, source).

    A row only replaces an existing one if its source is at least as trustworthy
    (source_rank >=), so provisional live data never overwrites official archives.
    Returns the number of rows inserted or updated.
    """
    if df.empty:
        return 0
    _require_utc(df)
    unknown = set(df["source"]) - SOURCE_RANK.keys()
    if unknown:
        raise ValueError(f"unknown source(s): {unknown}")
    df = df.drop_duplicates(["station_code", "pollutant", "ts_utc"], keep="last")
    with engine.begin() as conn:
        _copy_to_temp(
            conn,
            "tmp_meas",
            "measurements",
            df.assign(source_rank=df["source"].map(SOURCE_RANK)),
            [*MEASUREMENT_COLS, "source_rank"],
        )
        result = conn.exec_driver_sql(
            """
            INSERT INTO measurements AS m
                (station_code, pollutant, ts_utc, value, source, source_rank)
            SELECT station_code, pollutant, ts_utc, value, source, source_rank FROM tmp_meas
            ON CONFLICT (station_code, pollutant, ts_utc) DO UPDATE
               SET value = EXCLUDED.value, source = EXCLUDED.source,
                   source_rank = EXCLUDED.source_rank, ingested_at = now()
             WHERE EXCLUDED.source_rank >= m.source_rank
               AND (m.value IS DISTINCT FROM EXCLUDED.value OR m.source <> EXCLUDED.source)
            """
        )
        return result.rowcount


def upsert_weather(engine: Engine, df: pd.DataFrame) -> int:
    """Upsert hourly weather rows. 'archive' (reanalysis) is never overwritten by 'forecast'."""
    if df.empty:
        return 0
    _require_utc(df)
    df = df.drop_duplicates(["station_code", "ts_utc"], keep="last")
    value_cols = [c for c in WEATHER_COLS if c not in ("station_code", "ts_utc")]
    with engine.begin() as conn:
        _copy_to_temp(conn, "tmp_weather", "weather_hourly", df, WEATHER_COLS)
        result = conn.exec_driver_sql(
            f"""
            INSERT INTO weather_hourly AS w ({", ".join(WEATHER_COLS)})
            SELECT {", ".join(WEATHER_COLS)} FROM tmp_weather
            ON CONFLICT (station_code, ts_utc) DO UPDATE
               SET {", ".join(f"{c} = EXCLUDED.{c}" for c in value_cols)}, ingested_at = now()
             WHERE NOT (w.source = 'archive' AND EXCLUDED.source = 'forecast')
            """
        )
        return result.rowcount


FORECAST_COLS = [
    "model", "run_type", "station_code", "issue_ts", "horizon_h", "target_ts",
    "predicted", "p_episode", "uses_weather",
]


def upsert_forecasts(engine: Engine, df: pd.DataFrame) -> int:
    """Insert or replace model forecasts (key: model, run_type, station, issue, horizon)."""
    if df.empty:
        return 0
    for col in ("issue_ts", "target_ts"):
        _require_utc(df.rename(columns={col: "ts_utc"}))
    key = ["model", "run_type", "station_code", "issue_ts", "horizon_h"]
    df = df.drop_duplicates(key, keep="last")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in FORECAST_COLS if c not in key)
    with engine.begin() as conn:
        _copy_to_temp(conn, "tmp_fc", "forecasts", df, FORECAST_COLS)
        result = conn.exec_driver_sql(
            f"INSERT INTO forecasts ({', '.join(FORECAST_COLS)}) "
            f"SELECT {', '.join(FORECAST_COLS)} FROM tmp_fc "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}, created_at = now()"
        )
        return result.rowcount


@contextmanager
def ingestion_run(engine: Engine, job: str, params: dict | None = None) -> Iterator[dict]:
    """Log a job to ingestion_runs. Set run['rows'] inside the block."""
    with engine.begin() as conn:
        run_id = conn.execute(
            text(
                "INSERT INTO ingestion_runs (job, params) VALUES (:j, CAST(:p AS jsonb)) "
                "RETURNING run_id"
            ),
            {"j": job, "p": json.dumps(params or {}, default=str)},
        ).scalar_one()
    run = {"run_id": run_id, "rows": 0}
    try:
        yield run
    except BaseException as exc:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE ingestion_runs SET status='failed', finished_at=now(), "
                    "rows_upserted=:r, error=:e WHERE run_id=:id"
                ),
                {"r": run["rows"], "e": repr(exc)[:2000], "id": run_id},
            )
        raise
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE ingestion_runs SET status='success', finished_at=now(), "
                "rows_upserted=:r WHERE run_id=:id"
            ),
            {"r": run["rows"], "id": run_id},
        )
