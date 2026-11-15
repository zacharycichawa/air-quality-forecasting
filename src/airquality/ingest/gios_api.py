"""HTTP client for the GIOŚ air-quality REST API (v1) plus shared GIOŚ time/value helpers.

Timestamps: GIOŚ publishes naive hourly timestamps in fixed CET (UTC+1) all year,
labelling the END of the hour. `gios_label_to_utc` is the single place that converts them.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import httpx
import pandas as pd
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

BASE_URL = "https://api.gios.gov.pl/pjp-api/v1/rest"
PAGE_SIZE = 500
MAX_RETRY_SECONDS = 300  # overall cap per request, including backoff sleeps
MAX_WINDOW = timedelta(days=365)  # API limit is 366 days; stay safely below
CET_OFFSET = pd.Timedelta(hours=1)

log = logging.getLogger(__name__)


def gios_label_to_utc(labels: pd.Series) -> pd.Series:
    """Convert naive GIOŚ end-of-hour labels (fixed CET, UTC+1) to tz-aware UTC.

    There is no DST in this convention, so the mapping is a plain -1 h shift:
    no gaps or duplicates on the spring/autumn change days.
    """
    return (pd.to_datetime(labels) - CET_OFFSET).dt.tz_localize("UTC")


def utc_to_gios_label(ts: pd.Timestamp) -> str:
    """Inverse of `gios_label_to_utc` for API query parameters (`YYYY-MM-DD HH:mm`)."""
    return (ts.tz_convert("UTC") + CET_OFFSET).strftime("%Y-%m-%d %H:%M")


def clean_values(values: pd.Series) -> pd.Series:
    """Coerce to float (accepting strings with comma decimals); negatives become NaN."""
    as_text = values.astype("string").str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(as_text, errors="coerce").astype("float64")
    return numeric.where(numeric >= 0)


def _empty_measurements() -> pd.DataFrame:
    return pd.DataFrame({"sensor_code": [], "ts_utc": [], "value": []}).astype(
        {"sensor_code": "object", "ts_utc": "datetime64[ns, UTC]", "value": "float64"}
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.TransportError)


def _log_retry(path: str, state: RetryCallState) -> None:
    wait = state.next_action.sleep if state.next_action else 0.0
    log.warning("retrying %s in %.0f s (attempt %d failed)", path, wait, state.attempt_number)


class GiosClient:
    """Thin client: pagination, retries with exponential backoff, polite delay."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        delay: float = 0.2,
        backoff: float = 5.0,
        max_attempts: int = 6,
    ) -> None:
        self._http = client or httpx.Client(
            base_url=BASE_URL, timeout=httpx.Timeout(30.0, connect=10.0)
        )
        self._delay = delay
        self._backoff = backoff
        self._max_attempts = max_attempts

    def _get(self, path: str, **params: object) -> dict:
        def attempt() -> dict:
            resp = self._http.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=self._backoff, max=120),
            stop=stop_after_attempt(self._max_attempts) | stop_after_delay(MAX_RETRY_SECONDS),
            before_sleep=lambda s: _log_retry(path, s),
            reraise=True,
        )
        result = retrying(attempt)
        time.sleep(self._delay)
        return result

    def _paginate(self, path: str, **params: object) -> list[dict]:
        """Return all records of a list endpoint (top-level key starts with 'Lista')."""
        rows: list[dict] = []
        page, total = 0, 1
        while page < total:
            body = self._get(path, page=page, size=PAGE_SIZE, **params)
            key = next((k for k in body if k.startswith("Lista")), None)
            if key is None:
                break
            rows.extend(body[key])
            total = int(body.get("totalPages", 0))
            page += 1
        return rows

    def stations(self) -> pd.DataFrame:
        rows = self._paginate("station/findAll")
        df = pd.DataFrame(
            {
                "gios_station_id": [r["Identyfikator stacji"] for r in rows],
                "station_code": [r["Kod stacji"] for r in rows],
                "name": [r["Nazwa stacji"] for r in rows],
                "city": [r.get("Nazwa miasta") for r in rows],
                "voivodeship": [r.get("Województwo") for r in rows],
                "latitude": [r.get("WGS84 φ N") for r in rows],
                "longitude": [r.get("WGS84 λ E") for r in rows],
            }
        )
        df[["latitude", "longitude"]] = df[["latitude", "longitude"]].apply(
            pd.to_numeric, errors="coerce"
        )
        return df

    def station_sensors(self, station_id: int) -> pd.DataFrame:
        rows = self._paginate(f"station/sensors/{station_id}")
        return pd.DataFrame(
            {
                "gios_sensor_id": [r["Identyfikator stanowiska"] for r in rows],
                "gios_station_id": [r["Identyfikator stacji"] for r in rows],
                "param_code": [r["Wskaźnik - kod"] for r in rows],
            }
        )

    def _measurements(self, path: str, **params: object) -> pd.DataFrame:
        rows = self._paginate(path, **params)
        if not rows:
            return _empty_measurements()
        df = pd.DataFrame(rows)
        return pd.DataFrame(
            {
                "sensor_code": df["Kod stanowiska"],
                "ts_utc": gios_label_to_utc(df["Data"]),
                "value": clean_values(df["Wartość"]),
            }
        )

    def live(self, sensor_id: int) -> pd.DataFrame:
        """Last ~3 days. Summer timestamp convention is unverified; treated as CET like archival."""
        return self._measurements(f"data/getData/{sensor_id}")

    def archival(self, sensor_id: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Archival hourly data for ts_utc in [start, end], split into <=365-day windows."""
        frames = []
        window_start = start
        while window_start <= end:
            window_end = min(window_start + MAX_WINDOW, end)
            frames.append(
                self._measurements(
                    f"archivalData/getDataBySensor/{sensor_id}",
                    dateFrom=utc_to_gios_label(window_start),
                    dateTo=utc_to_gios_label(window_end),
                    sort="Data",  # ascending
                )
            )
            window_start = window_end + pd.Timedelta(hours=1)
        if not frames:
            return _empty_measurements()
        return pd.concat(frames, ignore_index=True).drop_duplicates(["ts_utc"])
