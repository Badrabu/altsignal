"""Tests for src/process/align.py (monthly alignment)."""

from datetime import date

import pandas as pd

import config
from src.process import align

METROS = [m for m in config.get_state_info("California").metros if m.name in ("Los Angeles", "San Diego")]


def postings(*rows):
    """Build a postings table from (job_id, first_seen, metro, sector)."""
    return pd.DataFrame(rows, columns=["job_id", "first_seen", "metro", "sector"])


def test_indeed_monthly_counts_by_calendar_month_with_zero_fill():
    """Counts are per calendar month; a metro/sector without postings is a real zero."""
    posts = postings(
        ("a", "2026-10-30", "Los Angeles", "Healthcare Occupations"),
        ("b", "2026-11-02", "Los Angeles", "Healthcare Occupations"),
        ("c", "2026-11-03", "San Diego", "Retail Occupations"),
    )
    observed = {date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 2), date(2026, 11, 3)}
    rows = align.indeed_monthly_rows(posts, "California", METROS, observed)

    def value(signal, geo, month):
        match = rows[(rows.signal_id == signal) & (rows.geo_name == geo) & (rows.period_start == month)]
        assert len(match) == 1
        return match["value"].iloc[0]

    oct_, nov = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")
    assert value("indeed_postings_healthcare", "Los Angeles, CA", oct_) == 1
    assert value("indeed_postings_healthcare", "Los Angeles, CA", nov) == 1
    assert value("indeed_postings_retail", "San Diego, CA", oct_) == 0
    assert value("indeed_postings_total", "California", nov) == 2
    assert set(rows["grain"]) == {"monthly"} and set(rows["unit"]) == {"postings"}


def test_indeed_quality_is_share_of_days_scraped():
    """Quality = observed days / days in the month."""
    posts = postings(("a", "2026-11-02", "Los Angeles", "Healthcare Occupations"))
    observed = {date(2026, 11, d) for d in range(1, 16)}      # 15 of 30 days
    rows = align.indeed_monthly_rows(posts, "California", METROS, observed)
    assert (rows["quality_score"] == 0.5).all()
    assert (rows["period_end"] == pd.Timestamp("2026-11-30")).all()


def test_unobserved_months_get_no_rows():
    """A month with no scrape at all is absent, not zero."""
    posts = postings(("a", "2026-09-05", "Los Angeles", "Healthcare Occupations"))
    rows = align.indeed_monthly_rows(posts, "California", METROS, {date(2026, 11, 2)})
    assert set(rows["period_start"]) == {pd.Timestamp("2026-11-01")}


def test_no_scrapes_gives_empty_result():
    """Never scraped -> nothing to build."""
    assert align.indeed_monthly_rows(postings(), "California", METROS, set()).empty


def quarterly(values):
    """Quarterly rows (test data) starting 2023-Q1."""
    starts = pd.date_range("2023-01-01", periods=len(values), freq="QS")
    return pd.DataFrame({
        "signal_id": "q", "source": "x", "geo_level": "national", "geo_name": "United States",
        "period_start": starts, "period_end": starts + pd.offsets.QuarterEnd(0), "value": values,
        "unit": "index", "quality_score": 1.0, "grain": "quarterly"})


def test_quarterly_values_are_repeated_and_tagged():
    """Each quarter becomes three months, all tagged 'quarterly_repeated'."""
    out = align.repeat_to_monthly(quarterly([10.0, 20.0]))
    assert len(out) == 6
    assert set(out["grain"]) == {"quarterly_repeated"}
    first = out[out["value"] == 10.0].sort_values("period_start")
    assert list(first["period_start"]) == list(pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
    assert first["period_end"].iloc[-1] == pd.Timestamp("2023-03-31")


def test_repeated_rows_share_the_native_zscore_and_do_not_extend_history():
    """The z-score comes from the quarterly series (4-quarter baseline), not from repeated months."""
    out = align.repeat_to_monthly(quarterly([1.0, 2.0, 3.0, 4.0, 10.0]))
    last_quarter = out[out["value"] == 10.0]
    assert last_quarter["normalized_value"].notna().all()
    assert last_quarter["normalized_value"].nunique() == 1
    assert out[out["value"] == 1.0]["normalized_value"].isna().all()
    assert out["period_start"].max() == pd.Timestamp("2024-03-01")     # nothing carried past the last quarter


def test_repeat_of_empty_input_is_empty():
    """No quarterly data -> nothing."""
    assert align.repeat_to_monthly(quarterly([]).iloc[0:0]).empty
