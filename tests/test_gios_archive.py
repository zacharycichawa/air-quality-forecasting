"""Archive parsing with tiny synthetic xlsx/zip files. No network."""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from airquality.db import upsert_measurements
from airquality.ingest import gios_archive as ga

HEADER = [
    ["Nr", 1, 2, 3],
    ["Kod stacji", "StA", "OldB", "Gone"],
    ["Wskaźnik", "PM10", "PM10", "PM10"],
    ["Czas uśredniania", "1g", "1g", "1g"],
    ["Jednostka", "ug/m3", "ug/m3", "ug/m3"],
    ["Kod stanowiska", "StA-PM10-1g", "OldB-PM10-1g", "Gone-PM10-1g"],
]

ROWS = [
    [dt.datetime(2024, 1, 1, 1, 0), "50,1", 3.0, None],  # comma-decimal string
    ["2024-01-01 02:00:00.00", 7, -1.0, "x"],  # string timestamp, negative, junk
    [dt.datetime(2024, 1, 1, 2, 59, 59, 995000), 1.5, 2.5, 3.5],  # float-junk timestamp
]


def make_zip(tmp_path: Path, member: str, extra_title_rows: int = 0) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    for _ in range(extra_title_rows):
        ws.append(["some title"])
    for row in [*HEADER, *ROWS]:
        ws.append(row)
    xlsx = tmp_path / "x.xlsx"
    wb.save(xlsx)
    zip_path = tmp_path / "2024.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(xlsx, member)
    return zip_path


def test_scrape_file_ids_pairs_each_link_with_its_own_label():
    html = (
        '<a href = "/pjp/archives/downloadFile/643">Metadane oraz kody</a>'
        '<a href = "/pjp/archives/downloadFile/644">Wyniki pomiarów z 2025 roku</a>'
        '<a href = "/pjp/archives/downloadFile/582">Wyniki pomiarów z 2024 roku</a>'
    )
    assert ga.scrape_file_ids(html) == {2025: 644, 2024: 582}


def test_find_member_tolerates_dotted_pollutant_name():
    names = ["2024/2024_PM2.5_1g.xlsx", "2024_PM10_24g.xlsx"]
    assert ga.find_member(names, 2024, "PM2.5") == "2024/2024_PM2.5_1g.xlsx"
    with pytest.raises(FileNotFoundError):
        ga.find_member(names, 2024, "NO2")


@pytest.mark.parametrize("title_rows", [0, 3])
def test_header_is_detected_by_label_not_row_number(tmp_path, title_rows):
    zip_path = make_zip(tmp_path, "2024_PM10_1g.xlsx", title_rows)
    df = ga.read_archive_file(zip_path, 2024, "PM10")
    assert len(df) == 3 * 3  # 3 hours x 3 stations; header rows ignored
    a = df[df["station_code"] == "StA"].sort_values("ts_utc")
    assert a["value"].tolist() == [50.1, 7.0, 1.5]
    # labels 01:00, 02:00, 03:00 (rounded) -> UTC 00:00, 01:00, 02:00
    assert [t.hour for t in a["ts_utc"]] == [0, 1, 2]
    assert str(a["ts_utc"].dt.tz) == "UTC"


def test_negative_and_junk_values_become_null(tmp_path):
    df = ga.read_archive_file(make_zip(tmp_path, "2024_PM10_1g.xlsx"), 2024, "PM10")
    b = df[df["station_code"] == "OldB"].sort_values("ts_utc")
    assert b["value"].iloc[0] == 3.0
    assert pd.isna(b["value"].iloc[1])
    gone = df[df["station_code"] == "Gone"].sort_values("ts_utc")
    assert gone["value"].iloc[:2].isna().all()


def test_match_stations_maps_old_codes_and_reports_counts(tmp_path):
    long = ga.read_archive_file(make_zip(tmp_path, "2024_PM10_1g.xlsx"), 2024, "PM10")
    rows, kept, dropped = ga.match_stations(long, {"StA", "NewB"}, {"OldB": "NewB"})
    assert set(rows["station_code"]) == {"StA", "NewB"}
    assert (kept, dropped) == (2, 1)


def test_download_year_uses_cache(tmp_path):
    (tmp_path / "2024.zip").write_bytes(b"cached")
    assert ga.download_year(2024, tmp_path) == tmp_path / "2024.zip"  # no network needed


@pytest.mark.integration
def test_archive_rows_load_into_db(engine, tmp_path, monkeypatch):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO stations (station_code, name) VALUES ('StA', 'A'), ('NewB', 'B')"
        )
    zip_path = make_zip(tmp_path, "2024_PM10_1g.xlsx")
    monkeypatch.setattr(ga, "download_year", lambda year: zip_path)
    df = ga.archive_measurements(engine, 2024, "PM10", {"OldB": "NewB"})
    assert upsert_measurements(engine, df) == 6
    assert upsert_measurements(engine, df) == 0  # idempotent
    with engine.connect() as conn:
        n, nulls = conn.exec_driver_sql(
            "SELECT count(*), count(*) FILTER (WHERE value IS NULL) FROM measurements"
        ).one()
    assert (n, nulls) == (6, 1)  # the negative value is stored as NULL


def test_read_old_to_new_codes_splits_comma_separated_old_codes(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "STACJE"
    ws.append(["Nr", "Kod stacji", "Stary Kod stacji"])
    ws.append([1, "NewA", "Pm.a08a, PmGdaLeczk08 ,, PmGdaLecz08m"])
    ws.append([2, "NewB", "OldB"])
    ws.append([3, "NewC", None])
    path = tmp_path / "metadata.xlsx"
    wb.save(path)
    assert ga.read_old_to_new_codes(path) == {
        "Pm.a08a": "NewA",
        "PmGdaLeczk08": "NewA",
        "PmGdaLecz08m": "NewA",
        "OldB": "NewB",
    }
