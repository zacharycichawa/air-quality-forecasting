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
the target time, station location, and the weather forecast for the target hour
(temperature, wind, humidity, pressure, precipitation and cloud cover). The model was
trained on 2019-2023, tuned on 2024 and tested once on 2025.

## How good the forecast is

The model was tested on 2025 using real archived weather forecasts issued 1-2 days
earlier, i.e. exactly what would have been available at the time. Its mean absolute
error (MAE) was 6.15 µg/m³ at 24 hours and 6.40 µg/m³ at 48 hours, compared with 7.72
and 8.86 µg/m³ for the persistence baseline ("tomorrow equals today"): 20 % and 28 %
better. Without weather information the errors were 6.55 and 7.13 µg/m³. With perfect
knowledge of the weather (an unrealistic upper bound) they would be about 5.3 and 5.6.

The model is most useful on polluted days: above 35 µg/m³ it is clearly better than
the alternatives. It still under-predicts the most severe smog peaks, and on clean days
it sometimes over-predicts when the forecast weather looks smog-prone.

## Smog alert probability

A separate classifier estimates the probability that PM2.5 exceeds 50 µg/m³ at the
target hour. From pollution history alone, it caught about half of all high-pollution
hours in 2025 (F1 0.40 at 24 hours, versus 0.35 for the rule "it is above 50 now").
Weather information improves it further.
