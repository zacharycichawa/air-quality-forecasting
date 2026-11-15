"""Download and parse the GIOŚ yearly archives (zip of one wide xlsx per pollutant).

Layout (checked for 2019-2025): a few header rows (first cell 'Kod stacji' holds the
station codes), then one row per hour: timestamp in column A, one value column per
station. The header row is located by its label, not by a fixed row number.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import httpx
import openpyxl
import pandas as pd
from sqlalchemy import Engine, text

from airquality.config import RAW_DIR
from airquality.ingest.gios_api import clean_values, gios_label_to_utc

ARCHIVES_URL = "https://powietrze.gios.gov.pl/pjp/archives"
DOWNLOAD_URL = ARCHIVES_URL + "/downloadFile/{file_id}"
METADATA_FILE = "metadata.xlsx"
STATION_CODE_LABEL = "Kod stacji"
OLD_CODE_LABEL = "Stary Kod stacji"
FILE_SUFFIX = {"PM2.5": "PM25", "PM10": "PM10", "NO2": "NO2"}

log = logging.getLogger(__name__)


def _link_pattern(label: str) -> str:
    """Regex capturing the download id of the link whose text matches `label`."""
    return r'downloadFile/(\d+)"(?:(?!downloadFile).)*?' + label


def scrape_file_ids(html: str) -> dict[int, int]:
    """Map year -> download id from the archives page ('Wyniki pomiarów z <year> roku')."""
    pattern = _link_pattern(r"Wyniki pomiar\w+ z (\d{4}) roku")
    return {int(y): int(i) for i, y in re.findall(pattern, html, re.S)}


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def download_year(year: int, raw_dir: Path = RAW_DIR) -> Path:
    """Return the cached zip for `year`, downloading it first if needed."""
    dest = raw_dir / f"{year}.zip"
    if dest.exists():
        return dest
    html = httpx.get(ARCHIVES_URL, timeout=60.0).text
    ids = scrape_file_ids(html)
    if year not in ids:
        raise ValueError(f"no archive for {year} on {ARCHIVES_URL}")
    return _download(DOWNLOAD_URL.format(file_id=ids[year]), dest)


def download_metadata(raw_dir: Path = RAW_DIR) -> Path:
    """Station metadata workbook ('Metadane oraz kody stacji i stanowisk')."""
    dest = raw_dir / METADATA_FILE
    if dest.exists():
        return dest
    html = httpx.get(ARCHIVES_URL, timeout=60.0).text
    match = re.search(_link_pattern("Metadane oraz kody"), html, re.S)
    if match is None:
        raise ValueError("metadata file not found on archives page")
    return _download(DOWNLOAD_URL.format(file_id=match.group(1)), dest)


def read_old_to_new_codes(metadata_xlsx: Path) -> dict[str, str]:
    """Old station code -> current code, from the STACJE sheet of the metadata workbook."""
    wb = openpyxl.load_workbook(metadata_xlsx, read_only=True)
    rows = wb["STACJE"].iter_rows(values_only=True)
    header = next(rows)
    new_idx = header.index(STATION_CODE_LABEL)
    old_idx = next(i for i, h in enumerate(header) if str(h).startswith(OLD_CODE_LABEL))
    mapping = {}
    for row in rows:
        if row[old_idx] and row[new_idx]:
            # One cell may list several old codes: 'Pm.a08a, PmGdaLeczk08, PmGdaLecz08m'.
            # Old codes are merged into the current code even when the site relocated
            # (e.g. MpKrakDietla -> MpKrakKamien, ~4 km): intentional, gives one continuous
            # city series per current station code.
            for old in str(row[old_idx]).split(","):
                if old.strip():
                    mapping[old.strip()] = str(row[new_idx]).strip()
    return mapping


def _normalise(name: str) -> str:
    return name.replace("PM2.5", "PM25").upper()


def find_member(names: list[str], year: int, pollutant: str) -> str:
    """Zip member holding the hourly file, e.g. '2024_PM25_1g.xlsx' (tolerates 'PM2.5')."""
    wanted = f"{year}_{FILE_SUFFIX[pollutant]}_1g.xlsx".upper()
    for name in names:
        if _normalise(Path(name).name) == wanted:
            return name
    raise FileNotFoundError(f"{wanted} not in archive")


def parse_wide_sheet(rows: list[tuple]) -> pd.DataFrame:
    """Wide sheet rows -> long frame [station_code, ts_utc, value] (no mapping/filtering)."""
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == STATION_CODE_LABEL)
    codes = rows[header_idx][1:]
    wide = pd.DataFrame([r[1:] for r in rows[header_idx + 1 :]], columns=range(len(codes)))
    stamps = pd.to_datetime(
        pd.Series([r[0] for r in rows[header_idx + 1 :]]), errors="coerce", format="mixed"
    )
    keep = stamps.notna()  # drops the remaining header rows (units, sensor codes, ...)
    wide, stamps = wide[keep], stamps[keep].dt.round("h")
    long = wide.assign(ts=stamps.values).melt(id_vars="ts", var_name="col", value_name="raw")
    long["station_code"] = long["col"].map(dict(enumerate(codes)))
    long["ts_utc"] = gios_label_to_utc(long["ts"])
    long["value"] = clean_values(long["raw"])
    return long.dropna(subset=["station_code"])[["station_code", "ts_utc", "value"]]


def read_archive_file(zip_path: Path, year: int, pollutant: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(find_member(zf.namelist(), year, pollutant))
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
    rows = [tuple(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    return parse_wide_sheet(rows)


def match_stations(
    long: pd.DataFrame, known: set[str], old_to_new: dict[str, str]
) -> tuple[pd.DataFrame, int, int]:
    """Keep rows of known stations (renaming old codes). Returns (rows, kept, dropped) codes."""
    long = long.assign(station_code=long["station_code"].replace(old_to_new))
    is_known = long["station_code"].isin(known)
    codes = long["station_code"].drop_duplicates()
    kept = int(codes.isin(known).sum())
    return long[is_known], kept, len(codes) - kept


def load_known_stations(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text("SELECT station_code FROM stations"))}


def archive_measurements(
    engine: Engine, year: int, pollutant: str, old_to_new: dict[str, str]
) -> pd.DataFrame:
    """Long-format rows ready for `upsert_measurements` for one year and pollutant."""
    long = read_archive_file(download_year(year), year, pollutant)
    rows, kept, dropped = match_stations(long, load_known_stations(engine), old_to_new)
    log.info("%d %s: stations matched=%d dropped=%d", year, pollutant, kept, dropped)
    return rows.assign(pollutant=pollutant, source="archive_xlsx")
