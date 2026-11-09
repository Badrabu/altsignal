"""Tests for src/ingest/msrs.py using small made-up CSV files (test fixtures)."""

import pandas as pd
import pytest

import config
from src import db
from src.ingest import msrs
from tests.fakes import FakeResponse, FakeSession

MONTHS = pd.date_range("2020-01-01", periods=14, freq="MS").strftime("%Y%m")


def make_csv(prefix, rows):
    """Build a wide MSRS-style CSV. ``rows`` = [(fips, abbr, naics, [values...])]."""
    header = "fips,stateabbr,naics," + ",".join(f"{prefix}{m}" for m in MONTHS)
    lines = [header] + [f"{f},{a},{n}," + ",".join(str(v) for v in vals) for f, a, n, vals in rows]
    return "\n".join(lines) + "\n"


YY = make_csv("yy", [
    ("06", "CA", "TOTAL", [float(i) for i in range(13)] + ["S"]),   # last month suppressed
    ("06", "CA", "441", [1.0] * 14),
    ("48", "TX", "TOTAL", [9.0] * 14),
    ("00", "USA", "TOTAL", [2.0] * 14),
])
COV = make_csv("cov", [
    ("06", "CA", "TOTAL", ["B"] * 13 + ["S"]),
    ("06", "CA", "441", ["D"] * 14),
    ("48", "TX", "TOTAL", ["A"] * 14),
    ("00", "USA", "TOTAL", ["A"] * 14),
])


def test_parse_keeps_only_requested_state():
    """Only the chosen state's rows come out, with correct ids and geography."""
    rows = msrs.parse_msrs(YY, COV, "CA")
    assert set(rows["geo_name"]) == {"California"}
    assert set(rows["signal_id"]) == {"msrs_yoy_TOTAL", "msrs_yoy_441"}
    assert set(rows["grain"]) == {"monthly"}


def test_parse_state_switch_needs_no_other_change():
    """Asking for Texas returns Texas rows from the same files."""
    rows = msrs.parse_msrs(YY, COV, "TX")
    assert set(rows["geo_name"]) == {"Texas"}
    assert (rows["value"] == 9.0).all()


def test_suppressed_cells_are_skipped_not_imputed():
    """'S' cells produce no row."""
    rows = msrs.parse_msrs(YY, COV, "CA")
    total = rows[rows["signal_id"] == "msrs_yoy_TOTAL"]
    assert len(total) == 13
    assert pd.Timestamp("2021-02-01") not in set(total["period_start"])


def test_coverage_class_becomes_quality_score():
    """Coverage letters map to numeric quality scores."""
    rows = msrs.parse_msrs(YY, COV, "CA")
    quality = rows.groupby("signal_id")["quality_score"].first()
    assert quality["msrs_yoy_TOTAL"] == msrs.COVERAGE_SCORE["B"]
    assert quality["msrs_yoy_441"] == msrs.COVERAGE_SCORE["D"]


def test_normalized_value_needs_twelve_prior_months():
    """The 13th month gets a z-score; earlier months do not."""
    rows = msrs.parse_msrs(YY, COV, "CA")
    total = rows[rows["signal_id"] == "msrs_yoy_TOTAL"].sort_values("period_start")
    assert total["normalized_value"].iloc[:12].isna().all()
    assert pd.notna(total["normalized_value"].iloc[12])


def test_period_end_is_last_day_of_month():
    """Monthly rows span the whole calendar month."""
    rows = msrs.parse_msrs(YY, COV, "CA")
    first = rows.sort_values("period_start").iloc[0]
    assert first["period_start"] == pd.Timestamp("2020-01-01")
    assert first["period_end"] == pd.Timestamp("2020-01-31")


def test_ingest_success_writes_signals_and_scrape_row():
    """A good download fills the signals table and logs a success."""
    def handler(url, params):
        text = YY if "_yy" in url else COV
        return FakeResponse(text=text)

    result = msrs.ingest("California", session=FakeSession(handler))
    assert result.status == "success"
    assert result.rows_returned == len(db.read_signals(source="msrs"))
    log = db.read_scrapes(source="msrs")
    assert log.loc[0, "status"] == "success"
    assert log.loc[0, "rows_returned"] == result.rows_returned


def test_ingest_download_failure_is_logged_with_clear_error():
    """A failed download still writes a scrapes row with the error message."""
    def handler(url, params):
        return FakeResponse(status_code=503)

    result = msrs.ingest("California", session=FakeSession(handler))
    assert result.status == "failed"
    log = db.read_scrapes(source="msrs")
    assert log.loc[0, "status"] == "failed"
    assert "Could not download MSRS file" in log.loc[0, "error_message"]
    assert len(db.read_signals()) == 0


def test_ingest_html_instead_of_csv_is_a_failure():
    """If census.gov changes and returns a web page, we fail clearly."""
    result = msrs.ingest("California", session=FakeSession(lambda u, p: FakeResponse(text="<html>oops</html>")))
    assert result.status == "failed"
    assert "does not look like a CSV" in result.error_message


def test_ingest_unknown_state_is_logged_as_failure():
    """A bad STATE value is logged rather than crashing."""
    result = msrs.ingest("Atlantis", session=FakeSession(lambda u, p: FakeResponse(text=YY)))
    assert result.status == "failed"
    assert "Unknown state" in result.error_message
