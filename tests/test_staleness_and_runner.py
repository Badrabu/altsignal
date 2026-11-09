"""Tests for the staleness alert, scrape logging and the run-all runner."""

from datetime import datetime, timedelta, timezone

import config
from src import db
from src.ingest import common, run_all, staleness


def record(source, status, hours_ago=0.0):
    """Insert a finished scrape row that ended ``hours_ago`` hours ago."""
    db.init_db()
    scrape_id = db.start_scrape(source)
    db.finish_scrape(scrape_id, status, rows_returned=1)
    finished = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    with db.get_connection() as conn:
        conn.execute("UPDATE scrapes SET finished_at = ? WHERE scrape_id = ?", (finished, scrape_id))


def test_recent_success_is_fresh():
    """A success 10 hours ago is fine."""
    record("indeed", "success", hours_ago=10)
    assert staleness.check_staleness(("indeed",)) == []


def test_success_older_than_48_hours_is_stale():
    """A success 49 hours ago triggers the alert."""
    record("indeed", "success", hours_ago=49)
    stale = staleness.check_staleness(("indeed",))
    assert [s.source for s in stale] == ["indeed"]
    assert "49" in stale[0].message()


def test_never_succeeded_is_stale():
    """A source with no successful run is reported."""
    stale = staleness.check_staleness(("viirs",))
    assert stale[0].last_success is None
    assert "ever been recorded" in stale[0].message()


def test_failed_and_partial_runs_do_not_count():
    """Only status 'success' resets the clock."""
    record("msrs", "failed", hours_ago=1)
    record("msrs", "partial", hours_ago=1)
    assert len(staleness.check_staleness(("msrs",))) == 1


def test_old_success_plus_recent_failure_is_stale():
    """A recent failure does not hide an old success."""
    record("msrs", "success", hours_ago=60)
    record("msrs", "failed", hours_ago=1)
    assert len(staleness.check_staleness(("msrs",))) == 1


def test_all_configured_sources_are_checked_by_default():
    """With an empty log, every configured source is reported."""
    assert {s.source for s in staleness.check_staleness()} == set(config.ALL_SOURCES)


def test_run_logged_records_success():
    """A job that fills the result is logged as success."""
    def job(result):
        result.rows_returned = 7
        result.metro_coverage = 0.5

    result = common.run_logged("msrs", job)
    log = db.read_scrapes(source="msrs").iloc[0]
    assert (result.status, log["status"], log["rows_returned"], log["metro_coverage"]) == ("success", "success", 7, 0.5)


def test_run_logged_records_unexpected_exception_as_failure():
    """Any exception becomes a failed row; run_logged itself does not raise."""
    def job(result):
        raise KeyError("boom")

    result = common.run_logged("msrs", job)
    log = db.read_scrapes(source="msrs").iloc[0]
    assert result.status == "failed"
    assert log["status"] == "failed" and "KeyError" in log["error_message"]
    assert log["finished_at"] is not None


def test_run_all_runs_every_source_even_when_some_fail(monkeypatch):
    """One failing source does not stop the others; each logs its own row."""
    calls = []

    def fake(name, succeed):
        """Build a fake ingest function that logs success or an IngestError."""
        def job(result):
            if not succeed:
                raise common.IngestError("no creds")

        def run(*args, **kwargs):
            calls.append(name)
            return common.run_logged(name, job)
        return run

    monkeypatch.setattr(run_all.msrs, "ingest", fake("msrs", True))
    monkeypatch.setattr(run_all.indeed, "ingest", fake("indeed", False))
    monkeypatch.setattr(run_all.viirs, "ingest", fake("viirs", False))
    monkeypatch.setattr(run_all.fred, "ingest_card_spending", fake("fred_card", False))
    monkeypatch.setattr(run_all.fred, "ingest_targets", fake("fred_targets", True))

    results, stale = run_all.run_all()
    assert calls == ["msrs", "indeed", "viirs", "fred_card", "fred_targets"]
    assert [r.status for r in results] == ["success", "failed", "failed", "failed", "success"]
    assert {s.source for s in stale} == {"indeed", "viirs", "fred_card"}
    assert run_all.main([]) == 1
