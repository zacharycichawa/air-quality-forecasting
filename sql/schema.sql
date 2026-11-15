-- Air quality warehouse schema. Idempotent: safe to run repeatedly.
-- Convention: every timestamp is stored as timestamptz in UTC and marks the END
-- of the hourly averaging period (GIOŚ convention; see docs/data_sources.md).

CREATE TABLE IF NOT EXISTS stations (
    station_code    text PRIMARY KEY,              -- GIOŚ "Kod stacji", e.g. 'LdLodzCzerni'
    gios_station_id integer UNIQUE,                -- API id (null if station unknown to API)
    name            text NOT NULL,
    city            text,
    voivodeship     text,
    latitude        double precision,
    longitude       double precision,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sensors (
    gios_sensor_id  integer PRIMARY KEY,           -- API measurement-position id
    station_code    text NOT NULL REFERENCES stations (station_code),
    pollutant       text NOT NULL,
    UNIQUE (station_code, pollutant)
);
-- Watermark: API data is complete up to here (contiguously). Used by `update`.
ALTER TABLE sensors ADD COLUMN IF NOT EXISTS loaded_until timestamptz;

-- source_rank: higher = more trustworthy. Lower-ranked data never overwrites higher.
--   1 = api_live (provisional), 2 = api_archival, 3 = archive_xlsx (official yearly files)
CREATE TABLE IF NOT EXISTS measurements (
    station_code    text NOT NULL REFERENCES stations (station_code),
    pollutant       text NOT NULL CHECK (pollutant IN ('PM2.5', 'PM10', 'NO2')),
    ts_utc          timestamptz NOT NULL,
    value           double precision CHECK (value IS NULL OR value >= 0),   -- µg/m³
    source          text NOT NULL CHECK (source IN ('api_live', 'api_archival', 'archive_xlsx')),
    source_rank     smallint NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (station_code, pollutant, ts_utc)
);
CREATE INDEX IF NOT EXISTS measurements_pollutant_ts_idx ON measurements (pollutant, ts_utc);

CREATE TABLE IF NOT EXISTS weather_hourly (
    station_code          text NOT NULL REFERENCES stations (station_code),
    ts_utc                timestamptz NOT NULL,
    temperature_2m        double precision,   -- °C
    relative_humidity_2m  double precision,   -- %
    wind_speed_10m        double precision,   -- m/s
    wind_direction_10m    double precision,   -- degrees
    surface_pressure      double precision,   -- hPa
    precipitation         double precision,   -- mm, preceding hour
    cloud_cover           double precision,   -- %
    boundary_layer_height double precision,   -- m
    source                text NOT NULL CHECK (source IN ('archive', 'forecast')),
    ingested_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (station_code, ts_utc)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id         bigserial PRIMARY KEY,
    job            text NOT NULL,
    params         jsonb,
    started_at     timestamptz NOT NULL DEFAULT now(),
    finished_at    timestamptz,
    status         text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'success', 'failed')),
    rows_upserted  bigint,
    error          text
);
