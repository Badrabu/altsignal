"""Helpers shared by every ingestion pipeline.

The most important one is ``run_logged``: it runs an ingest job and ALWAYS
writes a row to the ``scrapes`` table afterwards, whether the job succeeded or
failed. That is a hard rule of the project (spec: "write a row to a scrapes
log table on every run with status, rows returned, metro coverage, and error
message, even on failure").
"""

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

import config
from src import db

logger = logging.getLogger("altsignal")


class IngestError(Exception):
    """A problem with a clear, user-facing message (missing key, blocked, ...).

    The message is stored as-is in ``scrapes.error_message``.
    """


@dataclass
class ScrapeResult:
    """What a job reports back to ``run_logged`` (and what the caller gets).

    Attributes:
        source: Source name, e.g. "msrs".
        status: "success", "partial" or "failed".
        rows_returned: Number of signal rows (or postings) collected.
        error_message: Error text for failures, or a note for partial runs.
        metro_coverage: Share (0-1) of metros that returned data, if relevant.
    """

    source: str
    status: str = db.STATUS_SUCCESS
    rows_returned: int = 0
    error_message: str | None = None
    metro_coverage: float | None = None


def run_logged(source: str, job: Callable[[ScrapeResult], None]) -> ScrapeResult:
    """Run an ingest job and record the outcome in the ``scrapes`` table.

    The job receives a ``ScrapeResult`` and fills in ``rows_returned``,
    ``metro_coverage`` and, for partial runs, ``status`` and ``error_message``.
    If the job raises an exception the run is recorded as "failed" and the
    exception message is stored. This function never raises: callers inspect
    the returned result instead.

    Args:
        source: Source name used in the ``scrapes`` table.
        job: A function taking a ``ScrapeResult`` and doing the real work.

    Returns:
        The final ``ScrapeResult``.
    """
    db.init_db()
    scrape_id = db.start_scrape(source)
    result = ScrapeResult(source=source)
    try:
        job(result)
    except IngestError as exc:
        result.status = db.STATUS_FAILED
        result.error_message = str(exc)
    except Exception as exc:  # noqa: BLE001 - we want to log ANY failure
        logger.debug("Unexpected error in %s:\n%s", source, traceback.format_exc())
        result.status = db.STATUS_FAILED
        result.error_message = f"{type(exc).__name__}: {exc}"
    db.finish_scrape(
        scrape_id,
        status=result.status,
        rows_returned=result.rows_returned,
        error_message=result.error_message,
        metro_coverage=result.metro_coverage,
    )
    log = logger.info if result.status == db.STATUS_SUCCESS else logger.error
    log("[%s] %s: %d rows%s", source, result.status, result.rows_returned,
        f" - {result.error_message}" if result.error_message else "")
    return result


def make_session():
    """Create a ``requests`` session with the project's User-Agent header.

    Returns:
        A ``requests.Session``.
    """
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = config.HTTP_USER_AGENT
    return session


def setup_logging() -> None:
    """Send log messages to the terminal in a simple readable format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def start_daily_schedule(job: Callable[[], object], hour: int, minute: int) -> None:
    """Run ``job`` once now and then every day at ``hour:minute`` (blocking).

    Uses APScheduler's ``BlockingScheduler``. Stop it with Ctrl+C.

    Args:
        job: A function with no arguments.
        hour: Hour of day (0-23, local time).
        minute: Minute of the hour (0-59).
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "cron", hour=hour, minute=minute,
                      misfire_grace_time=3600, coalesce=True)
    logger.info("Scheduled daily at %02d:%02d. Running once now first.", hour, minute)
    job()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def exit_code(*results: ScrapeResult) -> int:
    """Turn results into a process exit code for command-line entry points.

    Args:
        results: One or more ``ScrapeResult`` objects.

    Returns:
        0 if every run succeeded, otherwise 1 (partial counts as a problem).
    """
    return 0 if all(r.status == db.STATUS_SUCCESS for r in results) else 1
