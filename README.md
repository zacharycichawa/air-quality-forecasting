# Air Quality Forecasting & AI Assistant (Poland)

End-to-end data project on Polish air quality: an automated pipeline that loads official
GIOŚ measurements and Open-Meteo weather into PostgreSQL, with PM2.5 forecasting,
a BI dashboard and an LLM assistant planned on top.

> **Status:** work in progress. The data ingestion layer is done and tested; analysis,
> forecasting, dashboard and assistant are next.

## What's built so far

- **PostgreSQL warehouse** ([sql/schema.sql](sql/schema.sql)): stations, sensors, hourly
  measurements (PM2.5, PM10, NO₂), hourly weather, and an `ingestion_runs` job log.
  All timestamps are UTC and mark the end of the averaging hour.
- **GIOŚ ingestion**:
  - Yearly official archives (2019–2025, ~20M hourly rows) parsed from xlsx.
  - REST API for the current year and daily incremental updates.
  - Each sensor keeps a `loaded_until` watermark, so incremental loads never leave gaps.
- **Source precedence**: official yearly files > archival API > live data. Provisional
  values can never overwrite verified ones (enforced in the upsert).
- **Weather ingestion** (ERA5 reanalysis + forecast, incl. boundary-layer height):
  - Stations in the same 0.25° grid cell share one request.
  - Loads are resumable and quota-aware (Open-Meteo free tier).
- **Engineering**:
  - Idempotent bulk upserts via `COPY`.
  - Retries with backoff and overall time caps.
  - 53 tests (unit tests with mocked HTTP + integration tests on a separate test database).

Data-source details and quirks (fixed-CET timestamps, rate limits, archive layout) are
documented in [docs/data_sources.md](docs/data_sources.md).

## Quick start

```bash
cp .env.example .env            # set the Postgres password
docker compose up -d            # or use a local PostgreSQL 17 install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

airquality init-db
airquality load-stations
airquality backfill-archive --years 2019-2025
airquality backfill-api --start 2026-01-01
airquality weather-archive --start 2019-01-01
airquality update               # daily incremental load
```

Run the tests (integration tests need a database named `airquality_test`):

```bash
pytest                          # or: pytest -m "not integration"
```

## Roadmap

1. ~~Data pipeline~~ ✅
2. Exploratory analysis: seasonality, heating-season smog, weather effects
3. PM2.5 24–48 h forecasting: baselines → LightGBM → deep learning, time-series CV
4. Power BI dashboard
5. LLM assistant: text-to-SQL, forecast tool, RAG over WHO/EU guidelines, with evaluation

## Data sources

- Air quality: [GIOŚ](https://powietrze.gios.gov.pl/) (Chief Inspectorate of Environmental Protection, Poland)
- Weather: [Open-Meteo](https://open-meteo.com/) (ERA5 reanalysis, CC BY 4.0)
