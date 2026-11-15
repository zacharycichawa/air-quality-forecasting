"""GIOŚ API client and time/value helpers. No network: HTTP is mocked with respx."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from airquality.ingest.gios_api import (
    BASE_URL,
    GiosClient,
    clean_values,
    gios_label_to_utc,
    utc_to_gios_label,
)


@pytest.fixture
def client() -> GiosClient:
    return GiosClient(delay=0, backoff=0, max_attempts=3)


def labels_to_utc(*labels: str) -> list[str]:
    return [t.strftime("%Y-%m-%d %H:%M") for t in gios_label_to_utc(pd.Series(labels))]


def test_label_is_end_of_hour_cet_shifted_to_utc():
    assert labels_to_utc("2025-01-01 01:00:00") == ["2025-01-01 00:00"]
    assert labels_to_utc("2025-07-01 13:00:00") == ["2025-07-01 12:00"]  # no summer-time shift


@pytest.mark.parametrize("day", ["2025-03-30", "2025-10-26"])
def test_dst_change_days_have_no_gaps_or_duplicates(day):
    hours = pd.date_range(f"{day} 01:00", periods=24, freq="h")
    utc = gios_label_to_utc(pd.Series(hours))
    assert utc.is_unique
    assert (utc.diff().dropna() == pd.Timedelta(hours=1)).all()
    assert str(utc.dt.tz) == "UTC"


def test_utc_to_label_roundtrip():
    ts = pd.Timestamp("2025-07-01 12:00", tz="UTC")
    assert utc_to_gios_label(ts) == "2025-07-01 13:00"


def test_clean_values_handles_strings_commas_and_negatives():
    out = clean_values(pd.Series(["50,1", "7.5", None, "abc", -3.0, 4]))
    assert out.iloc[:2].tolist() == [50.1, 7.5]
    assert out.iloc[2:5].isna().all()
    assert out.iloc[5] == 4.0


def station_page(n: int) -> httpx.Response:
    row = {
        "Identyfikator stacji": n,
        "Kod stacji": f"S{n}",
        "Nazwa stacji": f"Name {n}",
        "Nazwa miasta": "Town",
        "Województwo": "ŁÓDZKIE",
        "WGS84 φ N": "51.5",
        "WGS84 λ E": "19.25",
    }
    return httpx.Response(200, json={"Lista stacji pomiarowych": [row], "totalPages": 2})


@respx.mock
def test_stations_paginates_and_normalises(client):
    route = respx.get(f"{BASE_URL}/station/findAll")
    route.side_effect = [station_page(1), station_page(2)]
    df = client.stations()
    assert df["station_code"].tolist() == ["S1", "S2"]
    assert df["latitude"].tolist() == [51.5, 51.5]
    assert route.call_count == 2


@respx.mock
def test_retries_on_429_and_network_errors_then_succeeds(client):
    route = respx.get(f"{BASE_URL}/data/getData/1")
    ok = httpx.Response(200, json={"Lista danych pomiarowych": [], "totalPages": 0})
    route.side_effect = [httpx.Response(429), httpx.ConnectError("boom"), ok]
    assert client.live(1).empty
    assert route.call_count == 3


@respx.mock
def test_gives_up_after_max_attempts(client):
    route = respx.get(f"{BASE_URL}/data/getData/1").mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        client.live(1)
    assert route.call_count == 3


@respx.mock
def test_client_error_is_not_retried(client):
    route = respx.get(f"{BASE_URL}/data/getData/1").mock(return_value=httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        client.live(1)
    assert route.call_count == 1


@respx.mock
def test_archival_splits_long_ranges_and_converts_time(client):
    rows = [
        {"Kod stanowiska": "S-PM10-1g", "Data": "2024-01-01 01:00:00", "Wartość": 5.5},
        {"Kod stanowiska": "S-PM10-1g", "Data": "2024-01-01 02:00:00", "Wartość": None},
    ]
    body = {"Lista archiwalnych danych pomiarowych": rows, "totalPages": 1}
    route = respx.get(f"{BASE_URL}/archivalData/getDataBySensor/7").mock(
        return_value=httpx.Response(200, json=body)
    )
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2024-06-01", tz="UTC")  # ~880 days -> 3 windows of <=365 days
    df = client.archival(7, start, end)
    assert route.call_count == 3
    assert route.calls[0].request.url.params["dateFrom"] == "2022-01-01 01:00"  # UTC -> CET label
    assert df["ts_utc"].tolist() == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
    ]
    assert df["value"].iloc[0] == 5.5
    assert pd.isna(df["value"].iloc[1])


@respx.mock
def test_empty_archival_window_returns_empty_frame(client):
    body = {"Lista archiwalnych danych pomiarowych": [], "totalPages": 0}
    respx.get(f"{BASE_URL}/archivalData/getDataBySensor/7").mock(
        return_value=httpx.Response(200, json=body)
    )
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    df = client.archival(7, ts, ts + pd.Timedelta(days=1))
    assert df.empty
    assert list(df.columns) == ["sensor_code", "ts_utc", "value"]


@respx.mock
def test_archival_windows_are_contiguous_without_gap_or_overlap(client):
    """Both dateFrom and dateTo are inclusive, so window k+1 must start 1 h after window k ends."""
    body = {"Lista archiwalnych danych pomiarowych": [], "totalPages": 0}
    route = respx.get(f"{BASE_URL}/archivalData/getDataBySensor/7").mock(
        return_value=httpx.Response(200, json=body)
    )
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2024-06-01", tz="UTC")
    client.archival(7, start, end)
    spans = [
        (
            pd.Timestamp(c.request.url.params["dateFrom"]),
            pd.Timestamp(c.request.url.params["dateTo"]),
        )
        for c in route.calls
    ]
    assert len(spans) == 3
    for (_, prev_to), (next_from, _) in zip(spans, spans[1:], strict=False):
        assert next_from - prev_to == pd.Timedelta(hours=1)
    assert spans[0][0] == pd.Timestamp(utc_to_gios_label(start))
    assert spans[-1][1] == pd.Timestamp(utc_to_gios_label(end))
    assert all(c.request.url.params["sort"] == "Data" for c in route.calls)


@respx.mock
def test_multi_page_archival_collects_every_row_once(client):
    def page(first_hour: int, n: int) -> httpx.Response:
        rows = [
            {
                "Kod stanowiska": "S-PM10-1g",
                "Data": f"2024-01-01 {h:02d}:00:00",
                "Wartość": float(h),
            }
            for h in range(first_hour, first_hour + n)
        ]
        body = {"Lista archiwalnych danych pomiarowych": rows, "totalPages": 3}
        return httpx.Response(200, json=body)

    route = respx.get(f"{BASE_URL}/archivalData/getDataBySensor/7")
    route.side_effect = [page(1, 10), page(11, 10), page(21, 3)]
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    df = client.archival(7, ts, ts + pd.Timedelta(days=1))
    assert [c.request.url.params["page"] for c in route.calls] == ["0", "1", "2"]
    assert len(df) == 23
    assert df["ts_utc"].is_unique
    assert df["value"].tolist() == [float(h) for h in range(1, 24)]
