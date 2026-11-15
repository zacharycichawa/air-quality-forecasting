# Data sources spec: GIOS air quality + Open-Meteo

Verified live on 2026-11-15 with real HTTP calls unless marked **unverified**. Samples are truncated.

---
## 1. GIOS (Glowny Inspektorat Ochrony Srodowiska) API

### 1.1 Base URL, versions, docs
- Base: `https://api.gios.gov.pl/pjp-api/v1/rest/` (v1, current). Responses are `application/ld+json` (plain JSON parseable).
- Legacy `https://api.gios.gov.pl/pjp-api/rest/station/findAll` (old v0, English field names) -> connection fails (curl code 000/empty). Treat as dead. (The OpenAPI file lists some unversioned `/rest/...` paths for archivalData/metadata; not needed, use `/v1/`.)
- OpenAPI JSON: `https://api.gios.gov.pl/pjp-api/v3/api-docs`; Swagger UI: `https://api.gios.gov.pl/pjp-api/swagger-ui/index.html`; human docs: `https://powietrze.gios.gov.pl/pjp/content/api`. Contact: api@gios.gov.pl.
- All field names are POLISH with spaces/diacritics (exact keys below). Top-level data key is also Polish and differs per endpoint (e.g. `"Lista stacji pomiarowych"`, `"Lista danych pomiarowych"`). Robust approach: take the first key starting with `"Lista"`.
- Envelope: `{"@context":..., "<Lista ...>":[...], "meta":{...}, "links":{self,first,prev,next,last}, "totalPages":N}`. Note `links.next`/`last` are unreliable (on a single page they still point to page 1/0); use `totalPages` and loop `page=0..totalPages-1`.

### 1.2 Endpoints
| Purpose | Path (under /v1/rest) | Params | Notes |
|---|---|---|---|
| All stations | `station/findAll` | `page`,`size`,`sort` | `size=500` returns all 288 current stations in ONE page. |
| Sensors of a station | `station/sensors/{stationId}` | page,size,sort | Sensor = "stanowisko" (one pollutant at one station). |
| Current measurements | `data/getData/{sensorId}` | page,size,sort | Last ~3 days only (see 1.4). |
| Archival measurements, 1 sensor | `archivalData/getDataBySensor/{sensorId}` | `dateFrom`,`dateTo` (`YYYY-MM-DD HH:mm`) or `dayNumber`; page,size,sort | Max range 366 days (HTTP 400 otherwise). |
| Archival, all stations | `archivalData/getDataForAllStations` | `dateFrom`,`dateTo` or `dayNumber`; page,size,sort | Max 31 days; ALL pollutants/sensors (huge). |
| Archival, year x voivodeship | `archivalData/getDataForAllStationsByYearAndVoivodeship` | `year`,`voivodeship` (e.g. `ŚLĄSKIE`), `pollution` (e.g. `PM2.5`), page,size | Documented; **not executed** (unverified in practice). |
| Sensor metadata (incl. closed sensors, start/close dates) | `metadata/sensors` | page,size | 5745 rows total; **strongly throttled** (1.5). |
| Station metadata | `metadata/stations` | page,size | not called. |
| Air quality index | `aqindex/getIndex/{stationId}` | - | Documented; not called (not needed). |
| Version | `/pjp-api/version` | - | listed in OpenAPI, not called. |

Pagination (all list endpoints): `page` 0-based, default `size=20`, **documented max 500**. `size=501` and `size=1000` returned HTTP 200 (did not verify whether silently clamped) - just use 500.

### 1.3 Samples and fields
`GET station/findAll?size=2` ->
```json
{"Lista stacji pomiarowych":[
 {"Identyfikator stacji":11,"Kod stacji":"DsCzerStraza","Nazwa stacji":"Czerniawa",
  "WGS84 φ N":"50.912475","WGS84 λ E":"15.312190","Identyfikator miasta":142,"Nazwa miasta":"Czerniawa",
  "Gmina":"Świeradów-Zdrój","Powiat":"lubański","Województwo":"DOLNOŚLĄSKIE","Ulica":"ul. Strażacka 7"}],
 "totalPages":144,...}
```
Lat/lon are STRINGS (`"WGS84 φ N"`, `"WGS84 λ E"`). Station id = `Identyfikator stacji` (int); station code = `Kod stacji`.

`GET station/sensors/11` ->
```json
{"Lista stanowisk pomiarowych dla podanej stacji":[
 {"Identyfikator stanowiska":54,"Identyfikator stacji":11,"Wskaźnik":"pył zawieszony PM10",
  "Wskaźnik - wzór":"PM10","Wskaźnik - kod":"PM10","Id wskaźnika":3}, ...]}
```
Pollutant codes seen (`Wskaźnik - kod`): PM10, PM2.5, NO2, NO, NOx, O3, SO2, CO, C6H6 (+ metals/BaP etc. on 24h manual sensors). Id wskaźnika: PM10=3, NO2=6, NOx=7, O3=5, NO=16 (PM2.5 id not recorded). Units: ug/m3 (CO is also reported in ug/m3 in these services; per `meta`: "w jednostce μg/m3"). Sensor ids are what the data endpoints take.

`GET data/getData/54?size=3` ->
```json
{"Lista danych pomiarowych":[
 {"Kod stanowiska":"DsCzerStraza-PM10-1g","Data":"2026-11-15 14:00:00","Wartość":null},
 {"Kod stanowiska":"DsCzerStraza-PM10-1g","Data":"2026-11-15 13:00:00","Wartość":15.0},
 {"Kod stanowiska":"DsCzerStraza-PM10-1g","Data":"2026-11-15 12:00:00","Wartość":10.7}],"totalPages":...}
```
Fields: `Kod stanowiska` (string, `<StationCode>-<Pollutant>-<avg>`; e.g. `-1g` = 1 hour, `-24g` = 24 h), `Data` (string `YYYY-MM-DD HH:mm:ss`), `Wartość` (float or null). Default sort: `Data` descending.

`GET archivalData/getDataBySensor/14731?dateFrom=2025-01-01 00:00&dateTo=2025-01-02 00:00` -> same plus `Nazwa stacji`:
```json
{"Nazwa stacji":"Jelenia Góra, ul. Ogińskiego","Kod stanowiska":"DsJelGorOgin-PM2.5-1g","Data":"2025-01-01 00:00:00","Wartość":67.9}
```
Archival default sort is `Data` descending for getDataBySensor (ascending by code+date for the AllStations variants). Range is inclusive on both ends (00:00..00:00 next day = 25 rows). Use `sort=Data` for ascending. Data is sparse-by-sensor: it only returns rows for hours that exist.

`GET metadata/sensors` row (use to know which sensors are open and when they started):
```json
{"Nr":173,"Kod stanowiska":"DsJelGorOgin-PM2.5-1g","Kod stacji":"DsJelGorOgin","Nazwa stacji":"Jelenia Góra, ul. Ogińskiego",
 "Stary Kod stacji":null,"Wskaźnik - kod":"PM2.5","Wskaźnik":"pył zawieszony PM2.5","Czas uśredniania":"1-godzinny",
 "Typ pomiaru":"automatyczny","Data uruchomienia":"10/20/2014","Data zamknięcia":null,
 "Województwo":"DOLNOŚLĄSKIE","Nazwa strefy":"strefa dolnośląska"}
```
Dates here are `MM/DD/YYYY` strings; `Data zamknięcia` null = open. Filter `Czas uśredniania=="1-godzinny"` and `Typ pomiaru=="automatyczny"`.

Error body (HTTP 400 for >366 days): `{"error_result":"...","error_reason":"Zakres czasu nie może być większy niż 366 dni.","error_code":"API-ERR-100"}`.

### 1.4 History depth
- `data/getData`: meta says "od ostatniego dostępnego wyniku do trzeciej doby wstecz" = ~3 days (observed: 63 rows from 2026-11-13 00:00 to now for one sensor). Includes the current (not yet published) hour as a null row.
- Manual 24h sensors: results appear 4-8 weeks after sampling (docs). Manual PM2.5 24h sensors (105) are irrelevant for hourly forecast.
- `archivalData/getDataBySensor`: works for old dates (tested 2015 and 2019 for an existing PM2.5 sensor; returns data), range <=366 days per request; no "oldest date" limit found. Returns `[]` (HTTP 200, totalPages 0) when the sensor had no data in the window (e.g. sensor 54 for 2019 and 2022 windows - sensor/period without data, not an error). Latency: data up to ~hours ago is available (tested 2026-11-01..03 OK).
- A full year for one sensor = ~8760 rows = 18 pages at size=500.

### 1.5 Rate limits / throttling
- HTTP 429, JSON body `{"error":"Zbyt wiele żądań","message":"Przekroczono limit wywołań dla endpointu:/v1/rest/metadata/sensors ","status":429}`. No `Retry-After` header. Limits are PER ENDPOINT.
- `metadata/sensors`: ~3 requests then 429; blocks cleared after ~10-30 s. Fetching 13 pages needed ~3 min with 10 s backoff. Cache this; call rarely.
- `data/getData` (30 rapid sequential), `archivalData/getDataBySensor` (120+ rapid sequential), `station/findAll`, `station/sensors` (6 each): no 429 observed. No documented numeric limit found (**unverified** what the true ceiling is under parallel load). Be polite: <=5 req/s, honor 429 with exponential backoff (10 s start).
- `Cache-Control: max-age=30` on getData, 120 on findAll. Data refreshes hourly.

### 1.6 Timestamps (IMPORTANT)
- Format: naive string `YYYY-MM-DD HH:mm:ss`, no offset.
- Archival endpoint `meta` explicitly says: "Wyniki pomiarów jednogodzinnych udostępniane są w czasie CET (UTC+1)". Verified: for 2025-03-30 00:00..06:00 and 2025-10-26 00:00..06:00 (EU DST change days) the API returned 7 consecutive hourly rows each -> **fixed UTC+1 all year, no DST gap or duplicate**. Yearly files also have exactly 8760 data rows for 2025 (consistent).
- `getData` (current) meta says "czas lokalny" (local time). At test time (November, local = CET) it cannot be distinguished; behaviour in summer (CEST) is **unverified**. Assumption to validate in summer: same CET (UTC+1) convention. Safest: after first summer run compare `getData` with `archivalData` for the same hours and assert equality.
- Label = END of averaging hour (inferred, strongly supported): the 2025 yearly file's first row is `2025-01-01 01:00` and last is `2026-01-01 00:00`; API and file agree on values (JelGora PM2.5: 2025-01-01 01:00 = 50.1). The newest row in `getData` is the current hour with `null` (value not yet published). Treat `Data` as period-ending; convert to UTC with `utc = Data - 1h` (fixed CET), then optionally define `hour_start_utc = utc - 1h`. Pick one convention in the DB and document it (see section 3).
- Parse as `pytz`/`zoneinfo` NOT Europe/Warsaw; use fixed `timezone(timedelta(hours=1))`.

### 1.7 Nulls and quirks
- Missing = JSON `null` in `Wartość` (API) / empty cell in xlsx. Negative or zero values occasionally occur (instrument noise): do not assume >0.
- Values are floats with varying precision (e.g. 96.7489 vs 5.5).
- Station list (`findAll`) returns only current stations (288); `metadata/sensors` also lists closed sensors.

### 1.8 Counts (from `metadata/sensors`, 5745 sensors total; "open" = no `Data zamknięcia`)
- Stations in `station/findAll`: 288.
- PM2.5 sensors: 278 total; 172 hourly-automatic (**91 open at 91 stations**), 105 manual 24h, 1 automatic 24h.
- Open hourly-automatic: PM10 176, NO2 134, PM2.5 91.
- PM2.5 hourly sensor start years: 2019:27, 2020:19, 2021:18, 2022:15, 2023:15, 2024:11 (older ones 2004-2018: 60). So PM2.5 network grew sharply after 2018: for 2019+ expect roughly 60-90 stations with PM2.5.
- Yearly 2025 PM2.5 1h file has 99 station columns (includes some later-closed/mobile).

### 1.9 Bulk yearly archive files
- Page: `https://powietrze.gios.gov.pl/pjp/archives`. Direct links `https://powietrze.gios.gov.pl/pjp/archives/downloadFile/{id}` (HTTP 200, `Content-Disposition: attachment; filename="2025.zip"`, no auth, session cookie not required). Ids: 2019=322, 2020=424, 2021=486, 2022=524, 2023=564, 2024=582, 2025=644 (ids scraped from page HTML, re-scrape the page each run since new year ids are added; 2000-2018 also exist, e.g. 2017=262, 2018=603). Extras: 643 = metadata xlsx (sheets `STACJE`, `STANOWISKA`; the file is an .xlsx served as .zip name), 642 = statistics 2000-2025.
- **Latest year available: 2025 (full year)**. No 2026 file yet (published after year end). Current-year data must come from the API.
- 2025.zip = 50 MB, contains one xlsx per pollutant x averaging time, e.g. `2025_PM25_1g.xlsx` (4.8 MB), `2025_PM10_1g.xlsx` (9.2 MB), `2025_NO2_1g.xlsx` (6.5 MB), `2025_O3_1g.xlsx`, `2025_SO2_1g.xlsx`, `2025_CO_1g.xlsx`, `2025_C6H6_1g.xlsx`, `*_24g.xlsx`, `Depozycja_1m` etc. Naming: `<year>_<pollutant>_<1g|24g|1m>.xlsx` (PM2.5 is `PM25`; some others contain parentheses like `Pb(PM10)`).
- Layout of `2025_PM25_1g.xlsx` (sheet name `2025_PM25_1g`, 8766 rows x 100 cols):
  - row 1 `Nr`, 1..99
  - row 2 `Kod stacji` (e.g. DsJelGorOgin)
  - row 3 `Wskaźnik` (PM2.5)
  - row 4 `Czas uśredniania` (1g)
  - row 5 `Jednostka` (ug/m3)
  - row 6 `Kod stanowiska` (e.g. DsJelGorOgin-PM2.5-1g)
  - rows 7+: column A = timestamp, columns B.. = value per sensor (wide format, one column per sensor).
  - 8760 data rows (2025-01-01 01:00 .. 2026-01-01 00:00 -> end-of-hour, CET).
  - **Gotchas**: early rows' values are STRINGS (`'50.1'`) while later rows are floats; some timestamps carry float junk (`2025-12-31 22:00:43.78` or `'2025-01-01 03:00:00.00'`) and are mixed str/datetime -> parse then `round('h')`. Empty cells for missing. Header layout of older years (2019-2024) **not inspected** (unverified; likely similar but verify the 6-row header per year).
  - Use `openpyxl` read_only mode (about 100 columns x 8.7k rows, fine) or `pandas.read_excel(header=None)`.

### 1.10 Backfill recommendation (GIOS)
For ~2019-2025 hourly PM2.5/PM10/NO2: **use the yearly xlsx files** (7 zips x ~50 MB, 3 files each needed, no throttling, already wide-format and complete for all stations incl. closed ones, 1 download per year). The archival API would need ~(stations x sensors) = ~400 sensors x 7 years x ~18 pages = ~50k requests, plus per-sensor-per-year chunking at 366 days; workable (no 429 at 120 rapid calls) but 50x more requests and fiddly. Use the API only for (a) the current year 2026 YTD (sensors by `metadata`, windows <=366 d, size=500), (b) incremental updates (`getData` every hour/few hours, 3-day window gives overlap/self-healing), (c) gap repair. Check yearly files for restatements (GIOS validates data; API archival values may be revised vs. getData realtime values).

---
## 2. Open-Meteo

### 2.1 Endpoints
| API | URL | Use |
|---|---|---|
| Historical (ERA5 reanalysis) | `https://archive-api.open-meteo.com/v1/archive` | training features, 1940-present |
| Forecast | `https://api.open-meteo.com/v1/forecast` | future features for deployed model (+ `past_days`) |
| Historical Forecast | `https://historical-forecast-api.open-meteo.com/v1/forecast` | archived NWP forecasts (what the forecast API said then), incl. `boundary_layer_height`; useful to avoid reanalysis-vs-forecast train/serve skew |

### 2.2 Archive API
- Params: `latitude`,`longitude`,`start_date`,`end_date` (YYYY-MM-DD), `hourly=...`, `timezone`, `wind_speed_unit`, `models` (optional).
- Max range: no hard limit hit; a **7-year (2019-01-01..2025-12-31, 61,368 hourly rows) single-location request with 8 variables succeeded** in one call (HTTP 200). Larger requests cost more rate-limit units.
- Latency: with `models=era5` the last non-null hour on 2026-11-15 was 2026-11-09T23:00 (i.e. **~5-6 days lag**). The default (no `models`, = best_match) fills the recent days with other reanalysis/forecast model data and returned non-null up to the END of the current day (2026-11-15T23:00, includes future hours of today) -> do not treat default-model recent values as final ERA5; for training restrict `end_date` <= today-7 or use `models=era5`.
- Data source: default best_match (ERA5/ERA5-Land/IFS mix; grid ~9-11 km, e.g. requested 52.23,21.01 -> snapped 52.267,20.961). `models=era5` -> 0.25° (snapped 52.25,21.0). `models=era5_land` ok (~0.1°, 11 km) but `boundary_layer_height` is **not** available for era5_land (all null).
- Free tier (from open-meteo.com/en/pricing and `X-RateLimit-*` headers): non-commercial use only; **600 calls/min, 5,000/hour, 10,000/day**; no uptime guarantee. Calls are weighted: header `X-RateLimit-Cost: 1.0` for 1 location x small request, `2.0` for 2 locations; larger date ranges/more variables cost more. 429 returned when exceeded (not triggered).

### 2.3 Variables (exact hourly names)
`temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `surface_pressure`, `precipitation`, `cloud_cover`, `boundary_layer_height` (+ optional `wind_gusts_10m`, `dew_point_2m`, `pressure_msl`, `shortwave_radiation`, `snowfall`, `is_day`; not tested).
- **`boundary_layer_height` IS available**: in the archive API (default model: verified non-null, e.g. 2020-06-01 midday 1660-2345 m; 2026-11-14 night 145-300 m, day ~1000 m), in the forecast API (verified), and in the Historical Forecast API (verified returns series for 2021 and 2023 dates; coverage start not probed). Units: m. Not available with `models=era5_land`.
- Units (with `wind_speed_unit=ms`): °C, %, m/s, °, hPa, mm, %, m. Default wind unit is km/h - pass `wind_speed_unit=ms` explicitly.

### 2.4 Sample (archive, UTC, ms)
`archive?latitude=52.23&longitude=21.01&start_date=2026-11-01&end_date=2026-11-02&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,cloud_cover&timezone=UTC&wind_speed_unit=ms`
```json
{"latitude":52.26713,"longitude":20.961182,"generationtime_ms":0.149,"utc_offset_seconds":0,
 "timezone":"GMT","timezone_abbreviation":"GMT","elevation":113.0,
 "hourly_units":{"time":"iso8601","temperature_2m":"°C","relative_humidity_2m":"%","wind_speed_10m":"m/s",
   "wind_direction_10m":"°","surface_pressure":"hPa","precipitation":"mm","cloud_cover":"%"},
 "hourly":{"time":["2026-11-01T00:00","2026-11-01T01:00",...],"temperature_2m":[5.2,4.6,4.1,...],...}}
```
Response headers: `X-RateLimit-Cost/Limit(600)/Remaining/Reset/Window(60)`.

### 2.5 Forecast API
`forecast?latitude=..&longitude=..&hourly=<vars incl. boundary_layer_height>&past_days=2&forecast_days=3&timezone=UTC&wind_speed_unit=ms` -> verified 200; `past_days=2&forecast_days=3` gave series from 2026-11-13T00:00 (midnight of day-2) through end of day+3... (5 days). `forecast_days=16` -> 384 hourly rows (default max horizon 16 d). `past_days=92` OK (2232 rows with forecast_days=1); `past_days=93` returned a normal 200 (max documented 92; clamping behaviour not examined). Past part of the forecast response is a stitched analysis/short-lead forecast, not ERA5.

### 2.6 Timezones/format
- `timezone=UTC` -> `"timezone":"GMT"`, `utc_offset_seconds:0`, times like `2026-11-01T00:00` (ISO8601 without offset, naive). `timezone=GMT` identical. `timezone=Europe/Warsaw` -> `utc_offset_seconds:3600` and local naive times (DST-shifted) - avoid. Always send `timezone=UTC`.
- Open-Meteo hourly values for non-instantaneous variables (e.g. `precipitation`) are **preceding-hour** aggregates (value at 14:00 = 13:00-14:00; per Open-Meteo docs, not independently tested); `temperature_2m` etc. are instantaneous at the timestamp. This matches GIOS end-of-hour labelling: after converting GIOS (CET) to UTC, the same UTC timestamp joins directly.

### 2.7 Multiple coordinates
Comma-separated `latitude=52.23,50.06&longitude=21.01,19.94` works on archive and forecast (verified). The response then is a **JSON list** (one object per location, same order as input; each object has the same shape as the single response, with its own snapped `latitude`/`longitude`). Single coordinate = plain object, not a list. Cost header scales with locations (2 coords = cost 2.0). Keep batches modest (e.g. 10-20 locations, <=1-year slices) to stay well within 600/min.

---
## 3. Recommendations for ingestion design

**Time normalisation.** Store everything as `timestamptz` UTC with one convention. Suggested: `ts_utc` = END of the averaging hour (GIOS label) converted: `ts_utc = to_utc(Data, tz=fixed +01:00)`; e.g. GIOS `2025-07-01 13:00:00` -> `2025-07-01 12:00:00Z`. Open-Meteo hour timestamps (UTC) join 1:1 on that column (precipitation = preceding hour, same semantic). Never use Europe/Warsaw for GIOS (no DST in the data). Round xlsx timestamps to the hour first.

**Backfill (2019 .. 2025).** Download yearly zips (`/pjp/archives/downloadFile/{id}`; scrape ids from the archives page), extract only `*_PM25_1g.xlsx`, `*_PM10_1g.xlsx`, `*_NO2_1g.xlsx` (optionally O3/SO2), melt wide->long (`kod_stanowiska`, `ts`, `value`), coerce strings to float, drop nulls only at load time (store missing explicitly or omit consistently), upsert. Inspect header rows of each year's file (2019-2024 unverified). Cross-check one year against the API (verified equal for one sample).
**Backfill 2026 YTD**: API `archivalData/getDataBySensor` per open hourly sensor (`metadata/sensors`), `dateFrom=2026-01-01 00:00`, `dateTo=now`, `sort=Data`, `size=500`, loop pages (~18 pages/sensor/year).
**Weather backfill**: per station (or per city; dedupe stations by city centroid to reduce locations), archive API with `timezone=UTC`, `wind_speed_unit=ms`, whole date range in one request per location (or yearly), `models` default for boundary_layer_height consistency with the forecast API; for the last ~7 days re-pull later because best_match recent values are provisional (re-fetch with a 7-10 day overlap each run).

**Incremental.** Hourly or every 3 h: for each sensor `getData/{id}?size=500` (covers 3 days, self-heals gaps), upsert, ignore the trailing null current-hour row (or upsert as NULL and later overwrite). Weather: forecast API `past_days=3&forecast_days=3` for all stations in one multi-coordinate call; keep forecast issuance separately (table `weather_forecast` keyed by (location, valid_ts, issued_at)) if you want to train on forecast-like features (Historical Forecast API can seed that for training).
**Metadata**: refresh `station/findAll?size=500` + `station/sensors/{id}` weekly; `metadata/sensors` monthly at most with 10 s sleep + backoff on 429 (cache the result on disk).

**Rate limits.** GIOS: global client throttle ~5 req/s, retry on 429/5xx with exponential backoff (10 s, 20 s, 40 s..., max ~5 tries) since there is no Retry-After; treat `metadata/*` as 1 req / 15 s. Open-Meteo: stay under 600 calls/min, 5000/h, 10000/day (cost-weighted by locations and range); read `X-RateLimit-Remaining`; non-commercial use only (portfolio OK).

**Upsert keys.** `measurements(sensor_code TEXT, ts_utc timestamptz, pollutant, value NUMERIC NULL, source TEXT)` PK `(sensor_code, ts_utc)` with `ON CONFLICT DO UPDATE` (API later revises vs. realtime). Use `Kod stanowiska` (string) as the natural key (present in every source: API, xlsx header row 6) rather than the integer `Identyfikator stanowiska` (needs lookup, API only). `stations(station_code PK, station_id UNIQUE, lat, lon, ...)`. `weather(location_id, ts_utc)` PK; forecast table `(location_id, valid_ts_utc, issued_at)`.

**Unverified / open items.**
1. `getData` timestamp convention during CEST (summer): not testable in November; assumed CET fixed like archival; validate by comparing getData vs archival in summer.
2. Header layout of 2019-2024 yearly xlsx files (only 2025 inspected).
3. Whether `size>500` is clamped or honoured; numeric GIOS rate ceilings for non-metadata endpoints.
4. `getDataForAllStationsByYearAndVoivodeship` and `aqindex` not executed.
5. End-of-hour label is inferred (first row 01:00 of year, null current-hour row), not stated in docs.
6. Open-Meteo Historical Forecast API start date of `boundary_layer_height` coverage; `precipitation` preceding-hour semantic taken from docs.
