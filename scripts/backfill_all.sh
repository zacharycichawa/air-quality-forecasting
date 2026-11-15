#!/usr/bin/env bash
# One-off full historical load: stations, 2019-2025 yearly archives, then 2026
# year-to-date (GIOŚ API) and weather history (Open-Meteo) in parallel.
# Safe to re-run: every step is an idempotent upsert / resumable.
set -u
cd "$(dirname "$0")/.."
AQ=${AQ:-airquality}            # CLI from `pip install -e .` (activate the venv first)
L=data/logs; mkdir -p "$L"

{ $AQ init-db && $AQ load-stations && $AQ backfill-archive --years 2019-2025; } \
    > "$L/01_core.log" 2>&1 || { echo "core load failed, see $L/01_core.log"; exit 1; }
echo "core done $(date)"

$AQ backfill-api --start 2026-01-01 > "$L/02_api2026.log" 2>&1 & P1=$!
$AQ weather-archive --start 2019-01-01 > "$L/03_weather.log" 2>&1 & P2=$!
wait $P1; echo "api2026 exit $? $(date)"
wait $P2; echo "weather exit $? $(date)"
