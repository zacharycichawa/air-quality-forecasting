-- Analytical layer (used by EDA and the BI dashboard). Idempotent.
-- Refresh after loads with:  airquality refresh-views

-- Daily mean per station and pollutant, by Polish local calendar day.
-- ts_utc marks the END of the hour, so we shift back 1 h to the hour's start
-- before taking the local date (hour 00:00-01:00 belongs to that day).
-- A daily mean is only considered valid with >= 18 hourly values (75% coverage,
-- the usual air-quality-directive data-capture rule).
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_station AS
SELECT
    m.station_code,
    m.pollutant,
    ((m.ts_utc - interval '1 hour') AT TIME ZONE 'Europe/Warsaw')::date AS day,
    avg(m.value)                    AS mean_value,
    max(m.value)                    AS max_hourly,
    count(m.value)                  AS n_hours,
    count(m.value) >= 18            AS is_valid
FROM measurements m
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS daily_station_pk
    ON daily_station (station_code, pollutant, day);

-- Reference thresholds for daily means (µg/m³). Kept in a table so the
-- dashboard and the LLM assistant can cite them rather than hard-code them.
CREATE TABLE IF NOT EXISTS air_quality_thresholds (
    pollutant   text NOT NULL,
    standard    text NOT NULL,      -- 'WHO 2021 guideline' / 'EU limit'
    averaging   text NOT NULL,      -- 'daily' / 'annual'
    value       double precision NOT NULL,
    note        text,
    PRIMARY KEY (pollutant, standard, averaging)
);

INSERT INTO air_quality_thresholds VALUES
    ('PM2.5', 'WHO 2021 guideline', 'daily',  15, '99th percentile, i.e. 3-4 exceedance days per year'),
    ('PM2.5', 'WHO 2021 guideline', 'annual',  5, NULL),
    ('PM10',  'WHO 2021 guideline', 'daily',  45, '99th percentile, i.e. 3-4 exceedance days per year'),
    ('PM10',  'WHO 2021 guideline', 'annual', 15, NULL),
    ('NO2',   'WHO 2021 guideline', 'daily',  25, '99th percentile, i.e. 3-4 exceedance days per year'),
    ('NO2',   'WHO 2021 guideline', 'annual', 10, NULL),
    ('PM10',  'EU limit',           'daily',  50, 'not to be exceeded on more than 35 days per year'),
    ('PM10',  'EU limit',           'annual', 40, NULL),
    ('PM2.5', 'EU limit',           'annual', 25, NULL),
    ('NO2',   'EU limit',           'annual', 40, NULL)
ON CONFLICT DO NOTHING;
