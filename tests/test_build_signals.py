"""End-to-end tests for src/process/build_signals.py.

Each test first ingests with the same fakes used by the phase-one tests (fake
Census files, fake Indeed server, fake Earth Engine, fake FRED), then builds.
"""

import pandas as pd
import pytest

import config
from src import db
from src.ingest import fred, indeed, msrs, viirs
from src.process import build_signals as bs
from tests.fakes import FakeResponse, FakeSession
from tests.test_fred import FakeFred
from tests.test_indeed import all_metros_ok
from tests.test_msrs import COV, YY
from tests.test_viirs import FakeEE, good_months, patch_ee

CONTENT = ["signal_id", "source", "geo_level", "geo_name", "period_start", "period_end", "value",
           "normalized_value", "unit", "quality_score", "grain"]


def ingest_msrs():
    """Ingest the small MSRS test files."""
    return msrs.ingest("California", session=FakeSession(
        lambda url, params: FakeResponse(text=YY if "_yy" in url else COV)))


def snapshot():
    """All signal rows as a sorted table, without ingestion timestamps."""
    table = db.read_signals()[CONTENT]
    return table.sort_values(["signal_id", "geo_name", "grain", "period_start"]).reset_index(drop=True)


def by_source(reports):
    """Index build reports by source name."""
    return {r.source: r for r in reports}


def test_build_with_nothing_ingested_reports_no_data():
    """An empty database builds nothing and says what to run."""
    reports, problems = bs.build_signals("California")
    status = {r.source: r.status for r in reports}
    assert status == {"msrs": "no data", "viirs": "no data", "fred_targets": "no data",
                      "indeed": "no data"}
    assert problems == [] and db.read_signals().empty
    assert bs.main() == 1       # nothing was built


def test_build_msrs_fills_zscore_quality_and_grain():
    """MSRS rows keep raw values and get a 12-month z-score, quality and grain."""
    ingest_msrs()
    reports, problems = bs.build_signals("California")
    assert by_source(reports)["msrs"].status == "built" and problems == []
    rows = db.read_signals(source="msrs", signal_id="msrs_yoy_TOTAL").sort_values("period_start")
    assert (rows["grain"] == "monthly").all() and (rows["unit"] == "percent_yoy").all()
    assert rows["quality_score"].notna().all()
    assert rows["normalized_value"].iloc[:12].isna().all()
    assert rows["normalized_value"].iloc[12:].notna().all()


def test_build_is_idempotent(monkeypatch):
    """Running the build repeatedly changes nothing and creates no duplicates."""
    ingest_msrs()
    indeed.ingest("California", session=all_metros_ok(), pause=0)
    patch_ee(monkeypatch, good_months)
    viirs.ingest("California", ee=FakeEE(), start_date="2023-01-01")
    fred.ingest_targets("California", fred=FakeFred())
    bs.build_signals("California")
    first = snapshot()
    count = len(first)
    for _ in range(2):
        bs.build_signals("California")
    pd.testing.assert_frame_equal(snapshot(), first)
    assert len(snapshot()) == count
    with db.get_connection() as conn:
        dupes = conn.execute("""SELECT COUNT(*) FROM (SELECT 1 FROM signals
            GROUP BY signal_id, geo_level, geo_name, period_start, grain HAVING COUNT(*) > 1)""").fetchone()[0]
    assert dupes == 0


def test_build_indeed_adds_monthly_rows_and_keeps_weekly():
    """Indeed gets monthly rows (same signal ids); weekly rows stay, tagged weekly."""
    indeed.ingest("California", session=all_metros_ok(), pause=0)
    reports, _ = bs.build_signals("California")
    assert by_source(reports)["indeed"].status == "built"
    table = db.read_signals(source="indeed", signal_id="indeed_postings_total", geo_level="state")
    assert set(table["grain"]) == {"monthly", "weekly"}
    monthly = table[table["grain"] == "monthly"].iloc[0]
    assert monthly["unit"] == "postings"
    assert 0 < monthly["quality_score"] <= 1
    assert pd.isna(monthly["normalized_value"])           # no 12 months of history yet


def test_partly_scraped_months_stay_out_of_zscore_baselines():
    """A month scraped on few days never gets a z-score, nor feeds a later one."""
    months = pd.date_range("2025-01-01", periods=14, freq="MS")
    quality = [1.0] * 13 + [0.2]                           # latest month barely scraped
    rows = pd.DataFrame({
        "signal_id": "s", "source": "indeed", "geo_level": "state", "geo_name": "California",
        "period_start": months, "period_end": months + pd.offsets.MonthEnd(0),
        "value": [float(i) for i in range(14)], "unit": "postings",
        "quality_score": quality, "grain": "monthly"})
    scored = bs.add_zscores(rows)
    assert scored["normalized_value"].iloc[:12].isna().all()
    assert pd.notna(scored["normalized_value"].iloc[12])
    assert pd.isna(scored["normalized_value"].iloc[13])    # weak month: no z-score
    assert (scored["value"] == rows["value"]).all()        # raw values untouched


def test_viirs_low_coverage_months_never_reach_the_final_table(monkeypatch):
    """Bad-coverage VIIRS rows already in the table are dropped and deleted."""
    patch_ee(monkeypatch, good_months)
    viirs.ingest("Wyoming", ee=FakeEE(), start_date="2023-01-01")
    # Simulate rows written by an older/looser ingest: 50% low coverage.
    bad = db.read_signals(source="viirs", geo_level="state").iloc[[3]].copy()
    bad["period_start"], bad["period_end"] = "2030-01-01", "2030-01-31"
    bad["quality_score"] = 0.5
    db.write_signals(bad)

    reports, problems = bs.build_signals("Wyoming")
    report = by_source(reports)["viirs"]
    assert report.status == "built" and problems == []
    assert report.rows_dropped == {"VIIRS low coverage (more than 20% of pixels)": 1}
    table = db.read_signals(source="viirs")
    assert "2030-01-01" not in set(table["period_start"])
    assert (table["quality_score"] >= 0.8).all()           # coverage kept as a feature
    assert table["normalized_value"].notna().any()
    assert set(table["unit"]) == {"nW/cm2/sr"}


def test_fred_targets_are_built_with_full_quality():
    """Official series get quality 1.0."""
    ingest_msrs()
    fred.ingest_targets("California", fred=FakeFred())
    reports, problems = bs.build_signals("California")
    status = {r.source: r.status for r in reports}
    assert status["fred_targets"] == "built"
    assert problems == [] and bs.main() == 0
    rows = db.read_signals(source="fred_targets")
    assert (rows["quality_score"] == 1.0).all()
    assert set(rows["unit"]) == {"percent", "millions_usd"}


def test_units_are_never_mixed_across_signals():
    """Every signal_id has one unit and z-scores are computed per series."""
    ingest_msrs()
    fred.ingest_targets("California", fred=FakeFred())
    bs.build_signals("California")
    table = db.read_signals()
    assert table.groupby("signal_id")["unit"].nunique().max() == 1
    # Same raw numbers (0..13) in two series with different units give the same
    # z-score: each is normalised against itself, not against the other.
    unrate = db.read_signals(signal_id="fred_unemployment_rate_national").sort_values("period_start")
    retail = db.read_signals(signal_id="fred_retail_sales_national").sort_values("period_start")
    assert unrate["unit"].iloc[0] != retail["unit"].iloc[0]
    assert unrate["normalized_value"].iloc[-1] == pytest.approx(retail["normalized_value"].iloc[-1])


def test_mixed_units_stop_the_source_with_an_error():
    """A series with two units is reported as an error and not written."""
    ingest_msrs()
    with db.get_connection() as conn:
        conn.execute("UPDATE signals SET unit = 'percent' WHERE signal_id = 'msrs_yoy_TOTAL' "
                     "AND period_start = '2020-01-01'")
        before = conn.execute("SELECT COUNT(*) FROM signals WHERE normalized_value IS NOT NULL").fetchone()[0]
    reports, _ = bs.build_signals("California")
    report = by_source(reports)["msrs"]
    assert report.status == "error" and "Mixed units" in report.message
    with db.get_connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM signals WHERE normalized_value IS NOT NULL").fetchone()[0]
    assert after == before
    assert bs.main() == 1


def test_other_states_rows_are_ignored():
    """Building California ignores Texas rows in the same database."""
    ingest_msrs()
    msrs.ingest("Texas", session=FakeSession(lambda url, params: FakeResponse(text=YY if "_yy" in url else COV)))
    texas_before = db.read_signals(geo_name="Texas").sort_values("period_start").reset_index(drop=True)
    texas_before = texas_before.drop(columns="fetched_at")
    reports, _ = bs.build_signals("California")
    assert by_source(reports)["msrs"].rows_read == len(db.read_signals(geo_name="California"))
    pd.testing.assert_frame_equal(
        db.read_signals(geo_name="Texas").sort_values("period_start").reset_index(drop=True).drop(columns="fetched_at"),
        texas_before)


def test_same_named_metros_in_different_states_stay_separate():
    """Springfield MA and Springfield MO never share rows."""
    ma = {m.geo_name for m in config.get_state_info("Massachusetts").metros}
    mo = {m.geo_name for m in config.get_state_info("Missouri").metros}
    assert "Springfield, MA" in ma and "Springfield, MO" in mo and not ma & mo
    info = config.get_state_info("Massachusetts")
    table = pd.DataFrame({"geo_level": ["metro", "metro", "state", "national"],
                          "geo_name": ["Springfield, MA", "Springfield, MO", "Massachusetts", "United States"]})
    assert list(bs.in_scope(table, info)["geo_name"]) == ["Springfield, MA", "Massachusetts", "United States"]


def test_switching_state_needs_no_code_change(monkeypatch):
    """Changing config.STATE alone builds the other state."""
    msrs.ingest("Texas", session=FakeSession(lambda url, params: FakeResponse(text=YY if "_yy" in url else COV)))
    monkeypatch.setattr(config, "STATE", "Texas")
    reports, _ = bs.build_signals()
    assert by_source(reports)["msrs"].status == "built"
    assert bs.main() == 0
