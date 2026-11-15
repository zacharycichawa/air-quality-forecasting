"""Open-Meteo weather ingestion into ``weather_hourly``.

Two jobs:

* ``load_weather_archive``  - ERA5 reanalysis (training features, ``source='archive'``).
* ``load_weather_forecast`` - forecast API, last 7 days + next 3 days (``source='forecast'``,
  used to serve the model). It never overwrites 'archive' rows (see ``db.upsert_weather``).

The free tier is cost-weighted: cost = locations * days * variables / 140 per request
(measured via the ``X-RateLimit-Cost`` header), limit 600/min and 10,000/day. To keep a
multi-year backfill affordable the archive job

1. fetches each 0.25 degree ERA5 grid cell once and fans the result out to its stations,
2. batches several cells per request (bounded by ``MAX_REQUEST_COST``),
3. is resumable: only days that are not yet complete (24 rows) for every station of a cell
   are requested,
4. backs off on per-minute HTTP 429s and fails fast when the hourly/daily quota is exhausted
   (retrying cannot help within the run; rerun later to resume).

Time: requests use ``timezone=UTC``; the naive ISO timestamps are localised to UTC.
Values are instantaneous at the timestamp, except ``precipitation`` which is the sum over
the preceding hour (same end-of-hour convention as the GIOS data).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import date, timedelta

import httpx
import pandas as pd
from sqlalchemy import Engine, text
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from airquality.config import TARGET_POLLUTANT
from airquality.db import WEATHER_COLS, ingestion_run, upsert_weather

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "boundary_layer_height",
]
COMMON_PARAMS = {"hourly": ",".join(VARIABLES), "timezone": "UTC", "wind_speed_unit": "ms"}

CELL_DEG = 0.25  # ERA5 grid resolution
ERA5_LAG_DAYS = 7  # ERA5 is ~5-6 days behind; stay clear of provisional values
MAX_SLICE_DAYS = 366  # longest date range per archive request
MAX_REQUEST_COST = 100.0  # weighted cost budget per request (keeps well under 600/min)
DAILY_LIMIT = 10_000  # free-tier weighted calls per day
MAX_RETRY_SECONDS = 300  # overall cap on retrying one request
HTTP_TIMEOUT = httpx.Timeout(60, connect=10)
FORECAST_PAST_DAYS = 7
FORECAST_DAYS = 3

Cell = tuple[float, float]


class RateLimited(Exception):
    """HTTP 429 that may clear after a short wait (per-minute limit)."""


class DailyLimitExceeded(Exception):
    """HTTP 429 for the hourly or daily quota: waiting in-process is pointless; rerun later."""


# ---------------------------------------------------------------- planning (pure functions)


def request_cost(n_locations: int, n_days: int) -> float:
    """Open-Meteo weighted cost of one request."""
    return max(1.0, n_locations * n_days * len(VARIABLES) / 140)


def grid_cell(lat: float, lon: float) -> Cell:
    """Centre of the 0.25 degree ERA5 cell containing the point."""
    return (round(lat / CELL_DEG) * CELL_DEG, round(lon / CELL_DEG) * CELL_DEG)


def date_runs(days: list[date], max_len: int = MAX_SLICE_DAYS) -> list[tuple[date, date]]:
    """Group sorted days into contiguous (start, end) runs of at most ``max_len`` days."""
    runs: list[tuple[date, date]] = []
    for d in sorted(days):
        if runs and d - runs[-1][1] == timedelta(days=1) and (d - runs[-1][0]).days < max_len:
            runs[-1] = (runs[-1][0], d)
        else:
            runs.append((d, d))
    return runs


def plan_archive(
    cell_stations: dict[Cell, list[str]],
    complete_days: dict[str, set[date]],
    start: date,
    end: date,
) -> list[tuple[date, date, list[Cell]]]:
    """Split the missing work into requests: (start, end, cells) with bounded cost."""
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    cells_by_range: dict[tuple[date, date], list[Cell]] = defaultdict(list)
    for cell, codes in cell_stations.items():
        missing = [d for d in all_days if any(d not in complete_days.get(c, ()) for c in codes)]
        for run in date_runs(missing):
            cells_by_range[run].append(cell)

    requests = []
    for (s, e), cells in sorted(cells_by_range.items()):
        n_days = (e - s).days + 1
        per_request = max(1, int(MAX_REQUEST_COST // request_cost(1, n_days)))
        for i in range(0, len(cells), per_request):
            requests.append((s, e, cells[i : i + per_request]))
    return requests


def estimate_cost(requests: list[tuple[date, date, list[Cell]]]) -> float:
    return sum(request_cost(len(cells), (e - s).days + 1) for s, e, cells in requests)


# ---------------------------------------------------------------- database


def read_locations(engine: Engine, station_codes: list[str] | None = None) -> pd.DataFrame:
    """Stations with coordinates that actually have hourly PM2.5 data in `measurements`.

    Restricting to stations with real target data (not just a registered sensor) keeps
    the Open-Meteo request budget down (optionally filtered further by code).
    """
    df = pd.read_sql(
        text(
            "SELECT s.station_code, s.latitude, s.longitude FROM stations s "
            "WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL AND EXISTS ("
            "  SELECT 1 FROM measurements m WHERE m.station_code = s.station_code"
            "  AND m.pollutant = :p AND m.value IS NOT NULL) "
            "ORDER BY s.station_code"
        ),
        engine,
        params={"p": TARGET_POLLUTANT},
    )
    if station_codes is not None:
        df = df[df["station_code"].isin(station_codes)]
    return df.reset_index(drop=True)


def read_complete_days(engine: Engine, codes: list[str]) -> dict[str, set[date]]:
    """Per station, the UTC days that already hold 24 'archive' rows.

    Archive rows are stored even when every variable is null (see ``_to_rows``), so a
    fetched day always counts as complete and is never re-requested.
    """
    sql = text(
        "SELECT station_code, (ts_utc AT TIME ZONE 'UTC')::date AS d FROM weather_hourly "
        "WHERE source = 'archive' AND station_code = ANY(:codes) "
        "GROUP BY station_code, d HAVING count(*) = 24"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"codes": codes}).all()
    out: dict[str, set[date]] = defaultdict(set)
    for code, d in rows:
        out[code].add(d)
    return out


# ---------------------------------------------------------------- HTTP


@retry(
    retry=retry_if_exception_type((RateLimited, httpx.TransportError)),
    wait=wait_exponential(multiplier=5, max=60),
    stop=stop_after_attempt(6) | stop_after_delay(MAX_RETRY_SECONDS),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def get_json(client: httpx.Client, url: str, params: dict) -> tuple[list[dict], httpx.Headers]:
    """GET with backoff on 429 / network errors. Always returns a list of location objects."""
    resp = client.get(url, params=params)
    if resp.status_code == 429:
        reason = resp.text.lower()
        if "daily" in reason or "hourly" in reason:  # minutely limits clear within a minute
            raise DailyLimitExceeded(f"{resp.text} - rerun later to resume (progress is saved)")
        raise RateLimited(resp.text)
    resp.raise_for_status()
    payload = resp.json()
    return (payload if isinstance(payload, list) else [payload]), resp.headers


def _pause_if_budget_low(headers: httpx.Headers) -> None:
    """Sleep until the per-minute window resets when little budget remains."""
    remaining, reset = headers.get("X-RateLimit-Remaining"), headers.get("X-RateLimit-Reset")
    if remaining and reset and float(remaining) < MAX_REQUEST_COST:
        wait = max(0.0, float(reset) - time.time()) + 1
        log.info("rate-limit budget low (%s left), sleeping %.0fs", remaining, wait)
        time.sleep(wait)


def parse_location(obj: dict) -> pd.DataFrame:
    """One Open-Meteo location object -> DataFrame with tz-aware UTC ``ts_utc``."""
    hourly = obj["hourly"]
    df = pd.DataFrame({v: hourly[v] for v in VARIABLES})
    df.insert(0, "ts_utc", pd.to_datetime(hourly["time"], utc=True))
    return df.astype({v: "float64" for v in VARIABLES})


def fetch(client: httpx.Client, url: str, coords: list[Cell], params: dict) -> list[pd.DataFrame]:
    """Fetch several coordinates in one request; result order matches ``coords``."""
    query = {
        **COMMON_PARAMS,
        **params,
        "latitude": ",".join(f"{lat:g}" for lat, _ in coords),
        "longitude": ",".join(f"{lon:g}" for _, lon in coords),
    }
    objs, headers = get_json(client, url, query)
    if len(objs) != len(coords):
        raise ValueError(f"expected {len(coords)} locations, got {len(objs)}")
    _pause_if_budget_low(headers)
    return [parse_location(o) for o in objs]


def _to_rows(df: pd.DataFrame, codes: list[str], source: str) -> pd.DataFrame:
    """Fan one location's series out to its stations.

    Archive keeps all-null hours as rows so a day always has 24 rows once fetched (the
    resumability check relies on it). Forecast drops them: empty future hours are noise.
    """
    if source == "forecast":
        df = df.dropna(subset=VARIABLES, how="all")
    out = pd.concat([df.assign(station_code=c) for c in codes], ignore_index=True)
    return out.assign(source=source)[WEATHER_COLS]


def _batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------- public jobs


def load_weather_archive(
    engine: Engine,
    start: date,
    end: date | None = None,
    station_codes: list[str] | None = None,
) -> int:
    """Backfill ERA5 weather for start..end (default today - 7 days). Returns rows upserted."""
    latest = date.today() - timedelta(days=ERA5_LAG_DAYS)
    if end is None or end > latest:
        if end is not None:
            log.warning("end %s is within the ERA5 lag; clamped to %s", end, latest)
        end = latest
    params = {"start": start, "end": end, "stations": station_codes}
    with ingestion_run(engine, "weather_archive", params) as run:
        locations = read_locations(engine, station_codes)
        cell_stations: dict[Cell, list[str]] = defaultdict(list)
        for row in locations.itertuples():
            cell_stations[grid_cell(row.latitude, row.longitude)].append(row.station_code)

        complete = read_complete_days(engine, list(locations["station_code"]))
        requests = plan_archive(cell_stations, complete, start, end)
        cost = estimate_cost(requests)
        log.info(
            "archive %s..%s: %d stations in %d grid cells, %d requests, estimated cost %.0f "
            "weighted calls (daily limit %d)",
            start,
            end,
            len(locations),
            len(cell_stations),
            len(requests),
            cost,
            DAILY_LIMIT,
        )
        if cost > DAILY_LIMIT:
            log.warning("estimate exceeds the daily limit; the run will stop and can be resumed")

        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            for s, e, cells in requests:
                series = fetch(
                    client,
                    ARCHIVE_URL,
                    cells,
                    {"start_date": s.isoformat(), "end_date": e.isoformat(), "models": "era5"},
                )
                frames = zip(cells, series, strict=True)
                rows = pd.concat(
                    [_to_rows(df, cell_stations[c], "archive") for c, df in frames],
                    ignore_index=True,
                )
                run["rows"] += upsert_weather(engine, rows)
                log.info("archive %s..%s, %d cells: %d rows total", s, e, len(cells), run["rows"])
        return run["rows"]


def load_weather_forecast(engine: Engine, station_codes: list[str] | None = None) -> int:
    """Load the last 7 days + next 3 days from the forecast API. Returns rows upserted."""
    with ingestion_run(engine, "weather_forecast", {"stations": station_codes}) as run:
        locations = read_locations(engine, station_codes)
        unit_cost = request_cost(1, FORECAST_PAST_DAYS + FORECAST_DAYS)
        batch_size = max(1, int(MAX_REQUEST_COST // unit_cost))
        log.info(
            "forecast: %d stations, estimated cost %.0f weighted calls",
            len(locations),
            len(locations) * unit_cost,
        )
        params = {"past_days": FORECAST_PAST_DAYS, "forecast_days": FORECAST_DAYS}
        with httpx.Client(timeout=60) as client:
            for batch in _batches(list(locations.itertuples()), batch_size):
                coords = [(r.latitude, r.longitude) for r in batch]
                series = fetch(client, FORECAST_URL, coords, params)
                frames = zip(batch, series, strict=True)
                rows = pd.concat(
                    [_to_rows(df, [r.station_code], "forecast") for r, df in frames],
                    ignore_index=True,
                )
                run["rows"] += upsert_weather(engine, rows)
        return run["rows"]
