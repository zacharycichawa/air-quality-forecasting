"""Refresh the `stations` and `sensors` tables from the GIOŚ API."""

from __future__ import annotations

import logging

import httpx
import pandas as pd
from sqlalchemy import Engine, text

from airquality.ingest.gios_api import GiosClient

log = logging.getLogger(__name__)

# GIOŚ parameter code (normalised: upper-case, no dots) -> our pollutant name.
POLLUTANT_BY_CODE = {"PM25": "PM2.5", "PM10": "PM10", "NO2": "NO2"}

UPSERT_STATIONS = text(
    """
    INSERT INTO stations (station_code, gios_station_id, name, city, voivodeship,
                          latitude, longitude)
    VALUES (:station_code, :gios_station_id, :name, :city, :voivodeship, :latitude, :longitude)
    ON CONFLICT (station_code) DO UPDATE SET
        gios_station_id = EXCLUDED.gios_station_id, name = EXCLUDED.name, city = EXCLUDED.city,
        voivodeship = EXCLUDED.voivodeship, latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude, updated_at = now()
    """
)
# A replaced sensor gets a new id for the same station/pollutant (UNIQUE). Delete the stale
# row first and carry its watermark over: the series is the same, only the API id changed.
UPSERT_SENSORS = text(
    """
    WITH stale AS (
        DELETE FROM sensors
         WHERE station_code = :station_code AND pollutant = :pollutant
           AND gios_sensor_id <> :gios_sensor_id
        RETURNING loaded_until
    )
    INSERT INTO sensors (gios_sensor_id, station_code, pollutant, loaded_until)
    VALUES (:gios_sensor_id, :station_code, :pollutant, (SELECT max(loaded_until) FROM stale))
    ON CONFLICT (gios_sensor_id) DO UPDATE SET
        station_code = EXCLUDED.station_code, pollutant = EXCLUDED.pollutant
    """
)


def pollutant_for(param_code: str) -> str | None:
    """Map a GIOŚ parameter code to 'PM2.5' / 'PM10' / 'NO2', or None if not tracked."""
    return POLLUTANT_BY_CODE.get(param_code.upper().replace(".", ""))


def live_row_count(client: GiosClient, sensor_id: int) -> int | None:
    """Number of live rows of a sensor, or None for a manual (24h lab) sensor.

    The sensors endpoint does not say whether a sensor is automatic or manual (a station
    can have both for PM10/PM2.5), but the live endpoint answers HTTP 400 for manual ones.
    """
    try:
        return len(client.live(sensor_id))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 400:
            return None
        raise


def collect_sensors(client: GiosClient, stations: pd.DataFrame) -> pd.DataFrame:
    """Automatic PM2.5/PM10/NO2 sensors for the given stations.

    If a station has several automatic sensors for one pollutant, the one with live data wins.
    """
    code_by_id = dict(zip(stations["gios_station_id"], stations["station_code"], strict=True))
    rows = []
    for station_id in stations["gios_station_id"]:
        for s in client.station_sensors(int(station_id)).itertuples():
            pollutant = pollutant_for(s.param_code)
            n_live = live_row_count(client, int(s.gios_sensor_id)) if pollutant else None
            if n_live is not None:
                rows.append(
                    {
                        "gios_sensor_id": int(s.gios_sensor_id),
                        "station_code": code_by_id[station_id],
                        "pollutant": pollutant,
                        "n_live": n_live,
                    }
                )
    sensors = pd.DataFrame(rows, columns=["gios_sensor_id", "station_code", "pollutant", "n_live"])
    sensors = sensors.sort_values("n_live", ascending=False, kind="stable")
    return sensors.drop_duplicates(["station_code", "pollutant"]).drop(columns="n_live")


def refresh_stations(engine: Engine, client: GiosClient) -> tuple[int, int]:
    """Upsert all stations and their hourly PM2.5/PM10/NO2 sensors. Returns the counts."""
    stations = client.stations()
    sensors = collect_sensors(client, stations)
    # NaN would be stored as 'NaN' floats; send NULL instead.
    station_rows = stations.astype(object).where(stations.notna(), None).to_dict("records")
    with engine.begin() as conn:
        conn.execute(UPSERT_STATIONS, station_rows)
        if not sensors.empty:
            conn.execute(UPSERT_SENSORS, sensors.to_dict("records"))
    log.info("stations: %d, sensors: %d", len(stations), len(sensors))
    return len(stations), len(sensors)
