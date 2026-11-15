"""Extract modelling data from PostgreSQL into pandas (with a local Parquet cache).

Output frames are on a *complete hourly grid* per station (missing hours = NaN rows),
so that a shift of k rows within a station is always exactly k hours.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
from sqlalchemy import Engine

from airquality.config import DATA_DIR

log = logging.getLogger(__name__)
CACHE_DIR = DATA_DIR / "cache"

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "surface_pressure", "precipitation", "cloud_cover", "boundary_layer_height",
]

HOURLY_SQL = """
    SELECT station_code, ts_utc,
           max(value) FILTER (WHERE pollutant = 'PM2.5') AS pm25,
           max(value) FILTER (WHERE pollutant = 'PM10')  AS pm10,
           max(value) FILTER (WHERE pollutant = 'NO2')   AS no2
      FROM measurements
     WHERE station_code IN (SELECT DISTINCT station_code FROM measurements
                             WHERE pollutant = 'PM2.5' AND value IS NOT NULL)
     GROUP BY station_code, ts_utc
"""

# 'archive' (ERA5 reanalysis) wins over 'forecast' - the PK already guarantees one row.
WEATHER_SQL = f"SELECT station_code, ts_utc, {', '.join(WEATHER_VARS)} FROM weather_hourly"

STATIONS_SQL = "SELECT station_code, latitude, longitude FROM stations"


def read_sql_fast(engine: Engine, sql: str) -> pd.DataFrame:
    """Read a query via COPY ... TO STDOUT (CSV): several times faster than read_sql."""
    buf = io.BytesIO()
    with engine.connect() as conn:
        raw = conn.connection.driver_connection
        with raw.cursor() as cur, cur.copy(f"COPY ({sql}) TO STDOUT WITH CSV HEADER") as copy:
            for chunk in copy:
                buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf)


def _to_float32(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.astype({c: "float32" for c in cols})


def complete_hourly_grid(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Reindex each station to every hour between its first and last row."""
    parts = []
    for code, g in df.groupby("station_code", sort=True):
        hours = pd.date_range(g["ts_utc"].min(), g["ts_utc"].max(), freq="h")
        full = g.set_index("ts_utc")[value_cols].reindex(hours)
        full.index.name = "ts_utc"
        parts.append(full.reset_index().assign(station_code=code))
    out = pd.concat(parts, ignore_index=True)
    out["station_code"] = out["station_code"].astype("category")
    return out[["station_code", "ts_utc", *value_cols]]


def _cached(name: str, refresh: bool, build) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = build()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("cached %s: %d rows -> %s", name, len(df), path)
    return df


def load_hourly(engine: Engine, refresh: bool = False) -> pd.DataFrame:
    """station_code, ts_utc, pm25, pm10, no2 for every station with PM2.5 data."""
    def build() -> pd.DataFrame:
        df = read_sql_fast(engine, HOURLY_SQL)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df[df["pm25"].notna() | df["pm10"].notna() | df["no2"].notna()]
        # Grid spans the station's PM2.5 record only (the forecast target).
        span = df[df["pm25"].notna()].groupby("station_code")["ts_utc"].agg(["min", "max"])
        df = df.merge(span, left_on="station_code", right_index=True)
        df = df[(df["ts_utc"] >= df["min"]) & (df["ts_utc"] <= df["max"])]
        cols = ["pm25", "pm10", "no2"]
        return complete_hourly_grid(_to_float32(df, cols), cols)
    return _cached("hourly", refresh, build)


def load_weather(engine: Engine, refresh: bool = False) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        df = read_sql_fast(engine, WEATHER_SQL)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        return _to_float32(df, WEATHER_VARS)
    return _cached("weather", refresh, build)


def load_stations(engine: Engine) -> pd.DataFrame:
    return read_sql_fast(engine, STATIONS_SQL)
