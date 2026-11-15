---
title: How this project's data and forecasts work
source: Project documentation (README.md, docs/data_sources.md, notebooks/02_forecasting.ipynb)
url: https://github.com/zacharycichawa/air-quality-forecasting
---

## Data sources

Air quality data comes from GIOŚ, the Chief Inspectorate of Environmental Protection
in Poland, which runs the national monitoring network. The project loads the official
yearly archives for 2019–2025 and the GIOŚ REST API for the current year, for three
pollutants: PM2.5, PM10 and NO2. All values are hourly means in µg/m³. Weather data
(temperature, wind, humidity, pressure, precipitation, cloud cover and boundary layer
height) comes from Open-Meteo, using ERA5 reanalysis for history.

## Timestamps and daily means

GIOŚ timestamps mark the end of the averaging hour in fixed CET (UTC+1). The project
stores everything in UTC. Daily means use Polish local calendar days and are valid
only with at least 18 hourly values.

## How the PM2.5 forecast is made

A LightGBM gradient-boosting model forecasts hourly PM2.5 at each station 24 and 48
hours ahead (one model per horizon). Its inputs are recent PM2.5 values and averages
(last hours, last day, last week), recent PM10 and NO2, the hour, weekday and month of
the target time, and station location. A weather variant also uses the weather
forecast for the target hour. The model was trained on 2019–2023, tuned on 2024 and
tested once on 2025.

## How good the forecast is

On the 2025 test year, without weather features, the mean absolute error (MAE) was
6.55 µg/m³ at 24 hours (persistence baseline: 7.72) and 7.13 µg/m³ at 48 hours
(persistence: 8.86), i.e. 15 % and 20 % better than assuming tomorrow equals today. The
main weakness is severe smog: above 50 µg/m³ the model under-predicts and is worse than
persistence, because without weather information it cannot anticipate an inversion.

## Smog alert probability

A separate classifier estimates the probability that PM2.5 exceeds 50 µg/m³ at the
target hour. With its alert cut-off chosen on 2024 data, it caught about half of all
high-pollution hours in 2025 (F1 0.40 at 24 hours, versus 0.35 for the rule "it is
above 50 now").
