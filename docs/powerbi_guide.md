# Building the Power BI report

This guide walks through building the Power BI report on top of the project's
PostgreSQL database. The database already contains a ready-made **star schema** in
the `bi` schema, so Power BI only has to model relationships and measures. Expect
2–4 hours for a first version.

## 1. Before you start

1. Install **Power BI Desktop** (free, Microsoft Store or microsoft.com).
2. Make sure the data is fresh. In the project folder, with the virtual environment active:
   ```bash
   airquality update          # latest measurements
   airquality refresh-views   # recompute daily means
   airquality predict         # latest 24/48 h forecasts
   ```

## 2. Connect to PostgreSQL

1. *Home → Get data → More… → PostgreSQL database*.
2. Server `localhost`, Database `airquality`, Data connectivity mode **Import**.
3. Credentials: *Database* tab, user `postgres` and the password from your `.env`.
   If you get an encryption error, untick *Encrypt connections* for this local server.
4. In the navigator, tick these views from the `bi` schema, then *Load*:

| View | Rows (approx.) | What it is |
|---|---|---|
| `bi.dim_station` | ~290 | one row per station: name, city, voivodeship, coordinates, `is_mobile` |
| `bi.dim_date` | ~2,900 | calendar 2019 → today + 3 days, incl. `is_heating_season` |
| `bi.dim_threshold` | 10 | WHO 2021 guidelines and EU limits |
| `bi.fact_daily_air_quality` | ~0.9 M | valid daily means per station and pollutant, with exceedance flags |
| `bi.fact_forecast` | ~3.1 M | hourly forecasts (backtest year 2025 + live) with the measured value |

## 3. Model the relationships

Open *Model view* and create (or check) these relationships, all **many-to-one, single direction**:

| From (many) | To (one) |
|---|---|
| `fact_daily_air_quality[station_code]` | `dim_station[station_code]` |
| `fact_daily_air_quality[date]` | `dim_date[date]` |
| `fact_forecast[station_code]` | `dim_station[station_code]` |
| `fact_forecast[target_date]` | `dim_date[date]` |

Then:
- Select `dim_date` → *Table tools → Mark as date table* → column `date`.
- Set data categories: `dim_station[latitude]` = Latitude, `[longitude]` = Longitude,
  `[city]` = City. This makes the map visual work.
- Sort `dim_date[month_name]` by `month` and `day_name` by `day_of_week`
  (*Column tools → Sort by column*).
- Hide the foreign-key columns in the fact tables, so report users pick fields from the dimensions.

## 4. Measures (DAX)

Create a table for measures (*Home → Enter data*, name it `_Measures`), then add:

```DAX
Avg Concentration = AVERAGE ( fact_daily_air_quality[mean_ug_m3] )

Station Days = COUNTROWS ( fact_daily_air_quality )

Days Over WHO =
    CALCULATE ( [Station Days], fact_daily_air_quality[exceeds_who_daily] = TRUE () )

% Days Over WHO = DIVIDE ( [Days Over WHO], [Station Days] )

Avg Concentration PY =
    CALCULATE ( [Avg Concentration], SAMEPERIODLASTYEAR ( dim_date[date] ) )

YoY Change % = DIVIDE ( [Avg Concentration] - [Avg Concentration PY], [Avg Concentration PY] )

-- EU rule: PM10 daily mean > 50 µg/m³ on more than 35 days a year is a breach
PM10 EU Exceedance Days =
    CALCULATE ( [Station Days],
        fact_daily_air_quality[pollutant] = "PM10",
        fact_daily_air_quality[exceeds_eu_daily] = TRUE () )

Forecast MAE =
    CALCULATE ( AVERAGE ( fact_forecast[abs_error] ), fact_forecast[run_type] = "backtest" )

MAE LightGBM    = CALCULATE ( [Forecast MAE], fact_forecast[model] = "lightgbm" )
MAE Persistence = CALCULATE ( [Forecast MAE], fact_forecast[model] = "persistence" )
Skill vs Persistence = 1 - DIVIDE ( [MAE LightGBM], [MAE Persistence] )
```

Format the `%` measures as percentages and the concentrations with one decimal.
**Never sum concentrations:** always average them, or count days.

## 5. Report pages (suggested)

Add a **Pollutant** slicer (`fact_daily_air_quality[pollutant]`, single select,
default PM2.5) and a **Year** slicer to the first three pages. Sync them via
*View → Sync slicers*.

1. **Overview**
   - Cards: `Avg Concentration`, `% Days Over WHO`, `YoY Change %`.
   - Line chart: `Avg Concentration` by `dim_date[year_month]`, with a constant line at
     the WHO guideline (*Analytics pane → Constant line*: 15 for daily PM2.5).
   - Map: `dim_station[latitude]/[longitude]`, bubble size or colour = `Avg Concentration`.
2. **Seasonality**
   - Column chart: `Avg Concentration` by `month_name`.
   - Matrix: rows `month_name`, columns `day_name`, values `Avg Concentration` with
     conditional formatting (background colour scale) → a heat map.
3. **Station ranking**
   - Table: `city`, `station_name`, `Avg Concentration`, `% Days Over WHO`,
     `PM10 EU Exceedance Days`. Filter `is_mobile = False`; sort descending.
   - A *Top N* filter (10) on a bar chart of cities.
4. **Forecast performance** (filter `run_type = backtest`)
   - Cards: `MAE LightGBM`, `MAE Persistence`, `Skill vs Persistence`, with a slicer
     on `horizon_h` (24 / 48).
   - Line chart: `MAE LightGBM` and `MAE Persistence` by `month_name`.
   - Line chart: `actual_ug_m3` vs `predicted_ug_m3` by `target_local`, filtered to
     one station and a two-week winter window (a smog episode tells the story best).
5. **Live forecast** (filter `run_type = live`)
   - Table: `city`, `station_name`, `horizon_h`, `target_local`, `predicted_ug_m3`,
     `p_episode` (as %), with conditional formatting on `p_episode`.

## 6. Finish and publish to the repo

- Save the file as `powerbi/air_quality.pbix`. A `.pbix` stores no database password
  (Power BI keeps credentials separately), but check before committing.
- Export each page as an image (*File → Export → Export to PDF*, or screenshots) into
  `reports/powerbi/`, and link them from the README. Most recruiters won't open a
  `.pbix`, but they will look at screenshots.
- To refresh later: run the commands from step 1, then *Home → Refresh*.

## Interview talking points

- Why a star schema: facts at one grain (station-day, station-hour), shared
  dimensions, simple single-direction relationships. Fast, and measures stay correct.
- Why views in the database instead of Power Query transformations: one source of
  truth that both the BI report and the Python code use, and it's version-controlled.
- Valid-day rule (≥ 18 hourly values) and local-time days (data stored in UTC).
- Why the forecast page compares against persistence: a model is only useful if it
  beats the naive "same as now" forecast.
