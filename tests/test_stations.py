"""Station/sensor refresh with a mocked API and the test database."""

from __future__ import annotations

import httpx
import pytest
import respx

from airquality.ingest.gios_api import BASE_URL, GiosClient
from airquality.ingest.stations import pollutant_for, refresh_stations

STATION = {
    "Identyfikator stacji": 1,
    "Kod stacji": "StA",
    "Nazwa stacji": "Station A",
    "Nazwa miasta": "Town",
    "Województwo": "ŁÓDZKIE",
    "WGS84 φ N": "51.5",
    "WGS84 λ E": "19.25",
}
LIVE_ROW = {"Kod stanowiska": "StA-X-1g", "Data": "2025-01-01 01:00:00", "Wartość": 1.0}


def test_pollutant_code_mapping():
    assert pollutant_for("PM2.5") == "PM2.5"
    assert pollutant_for("PM25") == "PM2.5"
    assert pollutant_for("pm10") == "PM10"
    assert pollutant_for("NO2") == "NO2"
    assert pollutant_for("O3") is None
    assert pollutant_for("BaP(PM10)") is None


def sensor(sensor_id: int, code: str) -> dict:
    return {
        "Identyfikator stanowiska": sensor_id,
        "Identyfikator stacji": 1,
        "Wskaźnik - kod": code,
    }


def mock_api() -> None:
    respx.get(f"{BASE_URL}/station/findAll").mock(
        return_value=httpx.Response(
            200, json={"Lista stacji pomiarowych": [STATION], "totalPages": 1}
        )
    )
    sensors = [sensor(10, "PM10"), sensor(11, "PM10"), sensor(12, "PM2.5"), sensor(13, "O3")]
    respx.get(f"{BASE_URL}/station/sensors/1").mock(
        return_value=httpx.Response(
            200,
            json={"Lista stanowisk pomiarowych dla podanej stacji": sensors, "totalPages": 1},
        )
    )
    live = httpx.Response(200, json={"Lista danych pomiarowych": [LIVE_ROW], "totalPages": 1})
    respx.get(f"{BASE_URL}/data/getData/10").mock(return_value=httpx.Response(400))  # manual
    respx.get(f"{BASE_URL}/data/getData/11").mock(return_value=live)
    respx.get(f"{BASE_URL}/data/getData/12").mock(return_value=live)


@pytest.mark.integration
@respx.mock
def test_refresh_stations_upserts_only_tracked_automatic_sensors(engine):
    mock_api()
    client = GiosClient(delay=0, backoff=0)
    assert refresh_stations(engine, client) == (1, 2)
    assert refresh_stations(engine, client) == (1, 2)  # idempotent
    with engine.connect() as conn:
        sensors = conn.exec_driver_sql(
            "SELECT gios_sensor_id, pollutant FROM sensors ORDER BY 1"
        ).all()
        lat = conn.exec_driver_sql("SELECT latitude FROM stations").scalar_one()
    assert sensors == [(11, "PM10"), (12, "PM2.5")]
    assert lat == 51.5


@pytest.mark.integration
@respx.mock
def test_replaced_sensor_keeps_watermark_and_does_not_crash(engine):
    mock_api()
    client = GiosClient(delay=0, backoff=0)
    refresh_stations(engine, client)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE sensors SET loaded_until = '2025-06-01Z' WHERE gios_sensor_id = 11"
        )
    # GIOŚ replaces PM10 sensor 11 by 21: same station/pollutant, new id.
    listing = [sensor(21, "PM10"), sensor(12, "PM2.5")]
    respx.get(f"{BASE_URL}/station/sensors/1").mock(
        return_value=httpx.Response(
            200,
            json={"Lista stanowisk pomiarowych dla podanej stacji": listing, "totalPages": 1},
        )
    )
    live = {"Lista danych pomiarowych": [LIVE_ROW], "totalPages": 1}
    respx.get(f"{BASE_URL}/data/getData/21").mock(return_value=httpx.Response(200, json=live))
    assert refresh_stations(engine, client) == (1, 2)
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT gios_sensor_id, pollutant, loaded_until FROM sensors ORDER BY 1"
        ).all()
    assert [(r[0], r[1]) for r in rows] == [(12, "PM2.5"), (21, "PM10")]
    assert rows[1][2].isoformat().startswith("2025-06-01")  # watermark carried over
    assert rows[0][2] is None
