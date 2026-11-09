"""Tests for src/ingest/indeed.py using a fake local scraper server."""

from datetime import date

import pandas as pd
import requests

import config
from src import db
from src.ingest import indeed
from tests.fakes import FakeResponse, FakeSession


def job(job_id, occupations=("Healthcare Occupations",), published="2026-11-09"):
    """Build a posting shaped like the omkarcloud scraper's output."""
    return {"id": job_id, "occupations": list(occupations), "date_published": published}


def page(jobs, total_pages=1, count=None, current=1):
    """Build a scraper search response."""
    return {"jobs": jobs, "total_pages": total_pages, "count": count or len(jobs), "current_page": current}


def test_sector_uses_first_occupation_group():
    """The sector is the first name ending in 'Occupations'."""
    assert indeed.sector_of(job("a", ["Nurses", "Healthcare Occupations", "X Occupations"])) == "Healthcare Occupations"
    assert indeed.sector_of(job("b", [])) == indeed.UNCLASSIFIED
    assert indeed.sector_of({"id": "c"}) == indeed.UNCLASSIFIED


def test_sector_slug():
    """Slugs are short, lowercase and safe for ids."""
    assert indeed.sector_slug("Food & Beverage Occupations") == "food_beverage"


def test_dedup_by_job_id_within_and_across_runs():
    """A job ID is stored once, even if it appears in two metros or on two days."""
    empty = pd.DataFrame(columns=indeed.POSTING_COLUMNS)
    day1 = indeed.add_new_postings(
        empty, {"Los Angeles": [job("a"), job("b")], "San Diego": [job("b"), job("c")]}, date(2026, 11, 9))
    assert sorted(day1["job_id"]) == ["a", "b", "c"]
    assert day1.set_index("job_id").loc["b", "metro"] == "Los Angeles"   # first metro wins

    day2 = indeed.add_new_postings(day1, {"Los Angeles": [job("a"), job("d")]}, date(2026, 11, 10))
    assert list(day2["job_id"]) == ["d"]


def test_postings_without_id_are_ignored():
    """A posting with no id cannot be deduplicated, so it is skipped."""
    empty = pd.DataFrame(columns=indeed.POSTING_COLUMNS)
    out = indeed.add_new_postings(empty, {"X": [{"occupations": []}]}, date(2026, 11, 9))
    assert out.empty


def test_week_start_is_monday():
    """Weeks run Monday to Sunday."""
    assert indeed.week_start(date(2026, 11, 9)) == date(2026, 11, 9)    # a Monday
    assert indeed.week_start(date(2026, 11, 15)) == date(2026, 11, 9)   # Sunday


def test_weekly_counts_by_sector_metro_and_zero_fill():
    """Counts per week/metro/sector, with real zeros for observed weeks."""
    postings = pd.DataFrame([
        {"job_id": "a", "first_seen": "2026-11-09", "metro": "Los Angeles", "sector": "Healthcare Occupations", "date_published": ""},
        {"job_id": "b", "first_seen": "2026-11-10", "metro": "Los Angeles", "sector": "Healthcare Occupations", "date_published": ""},
        {"job_id": "c", "first_seen": "2026-11-16", "metro": "San Diego", "sector": "Retail Occupations", "date_published": ""},
    ])
    observed = {date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 16)}
    metros = [m for m in config.get_state_info("California").metros if m.name in ("Los Angeles", "San Diego")]
    rows = indeed.weekly_counts(postings, "California", metros, observed)
    rows["period_start"] = pd.to_datetime(rows["period_start"])

    def value(signal, geo, week):
        match = rows[(rows.signal_id == signal) & (rows.geo_name == geo) & (rows.period_start == week)]
        assert len(match) == 1, (signal, geo, week)
        return match["value"].iloc[0]

    assert value("indeed_postings_healthcare", "Los Angeles, CA", pd.Timestamp("2026-11-09")) == 2
    assert value("indeed_postings_healthcare", "Los Angeles, CA", pd.Timestamp("2026-11-16")) == 0
    assert value("indeed_postings_retail", "San Diego, CA", pd.Timestamp("2026-11-16")) == 1
    assert value("indeed_postings_total", "California", pd.Timestamp("2026-11-09")) == 2
    assert set(rows["grain"]) == {"weekly"}
    week1 = rows[rows.period_start == pd.Timestamp("2026-11-09")]
    assert week1["quality_score"].iloc[0] == 2 / 7     # scraped on 2 of 7 days


def make_server(metro_pages):
    """Return a fake session whose answers depend on the requested location."""
    def handler(url, params):
        location = params["location"]
        result = metro_pages[location]
        if isinstance(result, Exception):
            raise result
        if isinstance(result, FakeResponse):
            return result
        return FakeResponse(json_data=result[params["page"] - 1])
    return FakeSession(handler)


def test_collect_metro_follows_pagination():
    """All pages are fetched until total_pages."""
    pages = [page([job("a")], total_pages=2, count=2), page([job("b")], total_pages=2, count=2, current=2)]
    session = make_server({"Los Angeles, CA": pages})
    jobs, reported = indeed.collect_metro(session, "Los Angeles, CA", max_pages=50)
    assert [j["id"] for j in jobs] == ["a", "b"]
    assert reported == 2


def test_collect_metro_respects_max_pages():
    """max_pages caps the number of requests."""
    pages = [page([job(str(i))], total_pages=5, count=5, current=i + 1) for i in range(5)]
    session = make_server({"Los Angeles, CA": pages})
    jobs, _ = indeed.collect_metro(session, "Los Angeles, CA", max_pages=2)
    assert len(jobs) == 2


def all_metros_ok(extra=None):
    """Fake server where every California metro returns the same two postings."""
    responses = {m.indeed_location: [page([job(f"{m.name}-1"), job(f"{m.name}-2", ["Retail Occupations"])])]
                 for m in config.get_state_info("California").metros}
    responses.update(extra or {})
    return make_server(responses)


def test_ingest_success_end_to_end():
    """A good run stores postings, weekly signals and a success scrape row."""
    n_metros = len(config.get_state_info("California").metros)
    result = indeed.ingest("California", session=all_metros_ok(), pause=0)
    assert result.status == "success"
    assert result.rows_returned == 2 * n_metros
    assert result.metro_coverage == 1.0

    signals = db.read_signals(source="indeed")
    assert not signals.empty
    state_total = signals[(signals.signal_id == "indeed_postings_total") & (signals.geo_level == "state")]
    assert state_total["value"].iloc[0] == 2 * n_metros

    log = db.read_scrapes(source="indeed")
    assert log.loc[0, "status"] == "success" and log.loc[0, "metro_coverage"] == 1.0


def test_second_run_adds_no_duplicates():
    """Re-running with the same postings leaves counts unchanged."""
    indeed.ingest("California", session=all_metros_ok(), pause=0)
    before = db.read_signals(source="indeed")
    result = indeed.ingest("California", session=all_metros_ok(), pause=0)
    after = db.read_signals(source="indeed")
    assert result.rows_returned == 0
    assert before["value"].sum() == after["value"].sum()
    n_metros = len(config.get_state_info("California").metros)
    assert len(pd.read_csv(indeed.postings_path("CA"))) == 2 * n_metros   # stored once


def test_partial_failure_is_logged_as_partial_with_coverage():
    """If one metro is blocked the run is 'partial' and says which one."""
    blocked = FakeResponse(status_code=500, json_data={"error": "indeed jobs search failed: blocked"})
    session = all_metros_ok({"Fresno, CA": blocked})
    result = indeed.ingest("California", session=session, pause=0)
    assert result.status == "partial"
    assert "Fresno" in result.error_message and "blocking" in result.error_message
    n = len(config.get_state_info("California").metros)
    assert result.metro_coverage == (n - 1) / n
    assert db.last_successful_scrape("indeed") is None     # partial is not a success


def test_total_failure_server_down_is_logged_clearly():
    """No scraper server -> failed row that tells the user what to start."""
    def refuse(url, params):
        raise requests.exceptions.ConnectionError("connection refused")

    result = indeed.ingest("California", session=FakeSession(refuse), pause=0)
    assert result.status == "failed"
    log = db.read_scrapes(source="indeed")
    assert log.loc[0, "status"] == "failed"
    assert "Cannot reach the Indeed scraper server" in log.loc[0, "error_message"]
    assert log.loc[0, "metro_coverage"] == 0.0
    assert len(db.read_signals()) == 0


def test_blocked_by_indeed_failure_message():
    """An HTTP 500 from the scraper is reported as a possible IP block."""
    blocked = FakeResponse(status_code=500, json_data={"error": "blocked"})
    result = indeed.ingest("Wyoming", session=FakeSession(lambda u, p: blocked), pause=0)
    assert result.status == "failed"
    assert "blocking" in result.error_message


def test_shortfall_is_noted_but_still_success():
    """If Indeed reports more postings than we could fetch, the run says so."""
    big = {m.indeed_location: [page([job("x" + m.name)], total_pages=1, count=999)]
           for m in config.get_state_info("Wyoming").metros}
    result = indeed.ingest("Wyoming", session=make_server(big), pause=0)
    assert result.status == "success"
    assert "fewer postings were fetched" in result.error_message
