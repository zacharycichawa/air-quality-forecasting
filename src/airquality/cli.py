"""Command-line interface: `airquality <command>` (use `--db` to target another database)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Annotated, NamedTuple

# Windows consoles default to a legacy codepage; make output safe before typer/rich render.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402
import typer  # noqa: E402
from sqlalchemy import Engine, text  # noqa: E402

from airquality.config import POLLUTANTS  # noqa: E402
from airquality.db import (  # noqa: E402
    get_engine,
    ingestion_run,
    init_schema,
    refresh_views,
    upsert_measurements,
)
from airquality.ingest.gios_api import GiosClient  # noqa: E402
from airquality.ingest.gios_archive import (  # noqa: E402
    archive_measurements,
    download_metadata,
    read_old_to_new_codes,
)
from airquality.ingest.stations import refresh_stations  # noqa: E402

app = typer.Typer(help="Polish air quality data pipeline.", no_args_is_help=True)
log = logging.getLogger("airquality")

DEFAULT_UPDATE_LOOKBACK_DAYS = 30
HOUR = pd.Timedelta(hours=1)  # series step: starting one step after the watermark is contiguous
UPDATE_OVERLAP = pd.Timedelta(days=3)  # re-fetch recent hours: the API revises values


@app.callback()
def main(
    ctx: typer.Context,
    db: Annotated[
        str | None, typer.Option("--db", envvar="AIRQUALITY_DB", help="Database name override.")
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ctx.obj = get_engine(db)


def _engine(ctx: typer.Context) -> Engine:
    return ctx.obj


def parse_years(spec: str) -> list[int]:
    """'2019-2025' -> [2019..2025]; '2024' -> [2024]."""
    first, _, last = spec.partition("-")
    years = list(range(int(first), int(last or first) + 1))
    if not years:
        raise typer.BadParameter(f"empty year range: {spec}")
    return years


def _utc_timestamp(day: datetime) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def _now_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("h")


@app.command("init-db")
def init_db(ctx: typer.Context) -> None:
    """Create tables if they do not exist."""
    init_schema(_engine(ctx))
    log.info("schema ready")


@app.command("refresh-views")
def refresh_views_cmd(ctx: typer.Context) -> None:
    """Recompute the analytical materialized views (run after loads)."""
    engine = _engine(ctx)
    with ingestion_run(engine, "refresh-views"):
        refresh_views(engine)
    log.info("views refreshed")


@app.command("load-stations")
def load_stations(ctx: typer.Context) -> None:
    """Refresh stations and hourly PM2.5/PM10/NO2 sensors from the GIOŚ API."""
    engine = _engine(ctx)
    with ingestion_run(engine, "load-stations") as run:
        stations, sensors = refresh_stations(engine, GiosClient())
        run["rows"] = stations + sensors


@app.command("backfill-archive")
def backfill_archive(
    ctx: typer.Context,
    years: Annotated[str, typer.Option(help="Year or range, e.g. 2019-2025.")] = "2019-2025",
    pollutants: Annotated[list[str] | None, typer.Option(help="Repeat for several.")] = None,
) -> None:
    """Load the official yearly xlsx archives (downloaded once, cached in data/raw)."""
    engine = _engine(ctx)
    selected = pollutants or list(POLLUTANTS)
    if set(selected) - set(POLLUTANTS):
        raise typer.BadParameter(f"pollutants must be among {POLLUTANTS}")
    year_list = parse_years(years)
    params = {"years": year_list, "pollutants": selected}
    with ingestion_run(engine, "backfill-archive", params) as run:
        old_to_new = read_old_to_new_codes(download_metadata())
        for year in year_list:
            for pollutant in selected:
                df = archive_measurements(engine, year, pollutant, old_to_new)
                n = upsert_measurements(engine, df)
                run["rows"] += n
                log.info("%d %s: %d rows upserted", year, pollutant, n)


class SensorJob(NamedTuple):
    sensor_id: int
    station_code: str
    pollutant: str
    start: pd.Timestamp
    advance: bool  # move the sensor's loaded_until watermark to the run's end on success


def _sensors_with_watermark(engine: Engine, default: pd.Timestamp) -> list[tuple]:
    """(sensor_id, station, pollutant, effective watermark) per sensor.

    Watermark = loaded_until, else the end of its xlsx archive data, else `default`.
    """
    query = text(
        """
        SELECT s.gios_sensor_id, s.station_code, s.pollutant,
               coalesce(s.loaded_until,
                        (SELECT max(m.ts_utc) FROM measurements m
                          WHERE m.station_code = s.station_code AND m.pollutant = s.pollutant
                            AND m.source = 'archive_xlsx'),
                        :default)
          FROM sensors s ORDER BY s.gios_sensor_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"default": default.to_pydatetime()}).all()
    return [(r[0], r[1], r[2], pd.Timestamp(r[3]).tz_convert("UTC")) for r in rows]


def plan_update(engine: Engine, end: pd.Timestamp) -> list[SensorJob]:
    """Per sensor: from its watermark minus a small overlap; the watermark advances on success."""
    default = end - pd.Timedelta(days=DEFAULT_UPDATE_LOOKBACK_DAYS)
    return [
        SensorJob(sid, station, pollutant, mark - UPDATE_OVERLAP, True)
        for sid, station, pollutant, mark in _sensors_with_watermark(engine, default)
    ]


def plan_backfill(engine: Engine, start: pd.Timestamp, end: pd.Timestamp) -> list[SensorJob]:
    """Same start for every sensor; the watermark advances only if the range joins existing
    coverage (start <= watermark + 1 h), so a backfill island never hides a gap before it."""
    default = end - pd.Timedelta(days=DEFAULT_UPDATE_LOOKBACK_DAYS)
    return [
        SensorJob(sid, station, pollutant, start, start <= mark + HOUR)
        for sid, station, pollutant, mark in _sensors_with_watermark(engine, default)
    ]


def load_sensors(
    engine: Engine, client: GiosClient, jobs: list[SensorJob], end: pd.Timestamp
) -> tuple[int, int]:
    """Fetch and upsert each sensor's range. A failing sensor is logged and skipped (its
    watermark stays put). Returns (rows upserted, failed sensors)."""
    total = failed = 0
    for job in jobs:
        if job.start > end:
            continue
        try:
            df = client.archival(job.sensor_id, job.start, end).drop(columns="sensor_code")
            df = df.assign(station_code=job.station_code, pollutant=job.pollutant)
            total += upsert_measurements(engine, df.assign(source="api_archival"))
            if job.advance:
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            "UPDATE sensors SET loaded_until = GREATEST(loaded_until, :e) "
                            "WHERE gios_sensor_id = :id"
                        ),
                        {"e": end.to_pydatetime(), "id": job.sensor_id},
                    )
        except Exception:
            failed += 1
            log.exception("sensor %d (%s %s) failed", *job[:3])
    return total, failed


def _run_jobs(engine: Engine, name: str, jobs: list[SensorJob], end: pd.Timestamp, params: dict):
    with ingestion_run(engine, name, params) as run:
        run["rows"], failed = load_sensors(engine, GiosClient(), jobs, end)
        log.info("%s: %d rows upserted, %d sensors failed", name, run["rows"], failed)
        if failed:
            raise RuntimeError(f"{failed} of {len(jobs)} sensors failed")


@app.command("backfill-api")
def backfill_api(
    ctx: typer.Context,
    start: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="First day (UTC).")],
    end: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Last day (UTC, inclusive). Default: now."),
    ] = None,
) -> None:
    """Load a date range per sensor from the archival API (e.g. the current year to date)."""
    engine = _engine(ctx)
    now = _now_hour()
    start_ts = _utc_timestamp(start)
    end_ts = min(_utc_timestamp(end) + pd.Timedelta(days=1), now) if end else now
    jobs = plan_backfill(engine, start_ts, end_ts)
    _run_jobs(engine, "backfill-api", jobs, end_ts, {"start": start_ts, "end": end_ts})


@app.command()
def update(ctx: typer.Context) -> None:
    """Incremental load: per sensor, from its loaded_until watermark (minus 3 days) to now."""
    engine = _engine(ctx)
    end = _now_hour()
    _run_jobs(engine, "update", plan_update(engine, end), end, {"end": end})


@app.command("weather-archive")
def weather_archive(
    ctx: typer.Context,
    start: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="First day.")],
    end: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
) -> None:
    """Load Open-Meteo reanalysis weather for all stations."""
    from airquality.ingest.weather import load_weather_archive

    # load_weather_archive logs its own ingestion_run.
    load_weather_archive(_engine(ctx), start=start.date(), end=end.date() if end else None)


@app.command("weather-forecast")
def weather_forecast(ctx: typer.Context) -> None:
    """Load the Open-Meteo forecast for all stations."""
    from airquality.ingest.weather import load_weather_forecast

    load_weather_forecast(_engine(ctx))  # logs its own ingestion_run


@app.command("train-forecast")
def train_forecast(
    ctx: typer.Context,
    horizon: Annotated[int, typer.Option(help="Forecast horizon in hours.")] = 24,
    weather: Annotated[bool, typer.Option(help="Use target-hour weather features.")] = False,
    refresh_cache: Annotated[bool, typer.Option(help="Re-extract data from PostgreSQL.")] = False,
) -> None:
    """Train and evaluate the PM2.5 forecaster (baselines + LightGBM) for one horizon."""
    from airquality.forecast.train import run_experiment

    run_experiment(_engine(ctx), horizon, weather, refresh_cache=refresh_cache)


@app.command()
def predict(
    ctx: typer.Context,
    weather: Annotated[bool, typer.Option(help="Use the weather model (needs forecasts).")] = False,
) -> None:
    """Issue live 24 h / 48 h PM2.5 forecasts from the latest data into `forecasts`."""
    from airquality.forecast.predict import predict_live

    engine = _engine(ctx)
    with ingestion_run(engine, "predict", {"weather": weather}) as run:
        run["rows"] = predict_live(engine, use_weather=weather)


@app.command("load-backtests")
def load_backtests_cmd(
    ctx: typer.Context,
    weather: Annotated[bool, typer.Option(help="Load the weather model's backtests.")] = False,
) -> None:
    """Copy the test-year predictions into `forecasts` (for evaluation dashboards)."""
    from airquality.forecast.predict import load_backtests

    engine = _engine(ctx)
    with ingestion_run(engine, "load-backtests", {"weather": weather}) as run:
        run["rows"] = load_backtests(engine, use_weather=weather)


@app.command("export-snapshot")
def export_snapshot_cmd(ctx: typer.Context) -> None:
    """Write the compact Parquet snapshot used by the Streamlit app (app/data/)."""
    from airquality.snapshot import export_snapshot

    export_snapshot(_engine(ctx))


def _assistant():
    from airquality.assistant.agent import Assistant

    return Assistant(get_engine(reader=True))


@app.command("build-index")
def build_index(ctx: typer.Context) -> None:
    """Create the read-only DB role and embed docs/knowledge/ for the assistant (Ollama)."""
    from airquality.assistant.knowledge import KnowledgeIndex
    from airquality.assistant.llm import OllamaClient
    from airquality.assistant.sql_tool import ensure_reader_role

    ensure_reader_role(_engine(ctx))
    index = KnowledgeIndex.build(OllamaClient().embed)
    index.save()
    log.info("knowledge index: %d passages", len(index.chunks))


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Question in English or Polish.")]) -> None:
    """Ask the local LLM assistant a question about Polish air quality."""
    ans = _assistant().ask(question)
    for call in ans.tool_calls:
        typer.secho(f"  [{call['tool']}] {json.dumps(call['args'], ensure_ascii=False)}"
                    + (f"  -> error: {call['error']}" if call["error"] else ""), fg="cyan")
    typer.echo(f"\n{ans.answer}\n")
    typer.secho(f"({ans.seconds}s, {ans.steps} steps)", fg="bright_black")


@app.command("eval-assistant")
def eval_assistant(
    ctx: typer.Context,
    only: Annotated[str | None, typer.Option(help="One category or question id.")] = None,
    holdout: Annotated[bool, typer.Option(help="Use the held-out set (never tuned on).")] = False,
) -> None:
    """Run the assistant evaluation and save the report under reports/metrics/."""
    from airquality.assistant.evaluate import EVAL_FILE, HOLDOUT_FILE, run_eval

    s = run_eval(_assistant(), _engine(ctx), get_engine(reader=True),
                 path=HOLDOUT_FILE if holdout else EVAL_FILE, only=only)
    typer.echo(f"\naccuracy {s['accuracy']:.0%} on {s['n']} questions; by category: "
               + ", ".join(f"{c} {v['accuracy']:.0%} ({v['n']})"
                           for c, v in s["by_category"].items())
               + f"; median {s['median_seconds']}s per question")


if __name__ == "__main__":
    app()
