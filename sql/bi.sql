-- BI layer: a small star schema for Power BI (or any BI tool). Idempotent.
-- Dimensions: bi.dim_station, bi.dim_date, bi.dim_threshold
-- Facts:      bi.fact_daily_air_quality (valid daily means), bi.fact_forecast (hourly
--             forecasts joined to what was actually measured)
-- Timestamps are converted to Polish local time (timestamp without time zone),
-- which is what report users expect to see.

CREATE SCHEMA IF NOT EXISTS bi;

CREATE OR REPLACE VIEW bi.dim_station AS
SELECT s.station_code,
       s.name                                   AS station_name,
       coalesce(s.city, '(unknown)')            AS city,
       coalesce(s.voivodeship, '(unknown)')     AS voivodeship,
       s.latitude,
       s.longitude,
       s.station_code LIKE '%MOB'               AS is_mobile,
       EXISTS (SELECT 1 FROM sensors x WHERE x.station_code = s.station_code
                 AND x.pollutant = 'PM2.5')     AS measures_pm25
  FROM stations s;

CREATE OR REPLACE VIEW bi.dim_date AS
SELECT d::date                                  AS date,
       extract(year FROM d)::int                AS year,
       extract(quarter FROM d)::int             AS quarter,
       extract(month FROM d)::int               AS month,
       to_char(d, 'Mon')                        AS month_name,
       to_char(d, 'YYYY-MM')                    AS year_month,
       extract(isodow FROM d)::int              AS day_of_week,     -- 1 = Monday
       to_char(d, 'Dy')                         AS day_name,
       extract(isodow FROM d) >= 6              AS is_weekend,
       extract(month FROM d) IN (10, 11, 12, 1, 2, 3) AS is_heating_season
  FROM generate_series(date '2019-01-01', current_date + 3, interval '1 day') AS d;

CREATE OR REPLACE VIEW bi.dim_threshold AS
SELECT pollutant, standard, averaging, value AS threshold_ug_m3, note
  FROM air_quality_thresholds;

CREATE OR REPLACE VIEW bi.fact_daily_air_quality AS
SELECT d.station_code,
       d.pollutant,
       d.day                                    AS date,
       d.mean_value                             AS mean_ug_m3,
       d.max_hourly                             AS max_hourly_ug_m3,
       d.n_hours,
       d.mean_value > who.value                 AS exceeds_who_daily,
       CASE WHEN eu.value IS NULL THEN NULL
            ELSE d.mean_value > eu.value END    AS exceeds_eu_daily
  FROM daily_station d
  LEFT JOIN air_quality_thresholds who
         ON who.pollutant = d.pollutant AND who.standard = 'WHO 2021 guideline'
        AND who.averaging = 'daily'
  LEFT JOIN air_quality_thresholds eu
         ON eu.pollutant = d.pollutant AND eu.standard = 'EU limit' AND eu.averaging = 'daily'
 WHERE d.is_valid;

CREATE OR REPLACE VIEW bi.fact_forecast AS
SELECT f.model,
       f.run_type,
       f.uses_weather,
       f.station_code,
       f.horizon_h,
       (f.issue_ts  AT TIME ZONE 'Europe/Warsaw')                       AS issue_local,
       (f.target_ts AT TIME ZONE 'Europe/Warsaw')                       AS target_local,
       ((f.target_ts - interval '1 hour') AT TIME ZONE 'Europe/Warsaw')::date AS target_date,
       f.predicted                                                      AS predicted_ug_m3,
       m.value                                                          AS actual_ug_m3,
       abs(f.predicted - m.value)                                       AS abs_error,
       f.p_episode
  FROM forecasts f
  LEFT JOIN measurements m
         ON m.station_code = f.station_code AND m.pollutant = 'PM2.5'
        AND m.ts_utc = f.target_ts;
