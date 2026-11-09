"""Run every ingestion pipeline for the active state, then the staleness check.

    python -m src.ingest.run_all                 # one run
    python -m src.ingest.run_all --schedule      # now, then daily (APScheduler)

Each source runs independently: if one fails (for example Earth Engine has no
credentials) the others still run, and the failure is recorded in the
``scrapes`` table. The exit code is 0 only if every source succeeded and no
source is stale.
"""

from __future__ import annotations

import argparse
import logging
import sys

import config
from src.ingest import fred, indeed, msrs, staleness, viirs
from src.ingest.common import ScrapeResult, setup_logging, start_daily_schedule

logger = logging.getLogger("altsignal")


def run_all() -> tuple[list[ScrapeResult], list[staleness.StaleSource]]:
    """Run all sources in turn, then check for stale sources.

    Returns:
        ``(results, stale)``: one ``ScrapeResult`` per run and the list of
        stale sources found afterwards.
    """
    logger.info("Running all ingestion pipelines for %s", config.get_state_info().name)
    results = [
        msrs.ingest(),
        indeed.ingest(),
        viirs.ingest(),
        fred.ingest_card_spending(),
        fred.ingest_targets(),
    ]
    stale = staleness.check_staleness()
    staleness.report_staleness(stale)
    return results, stale


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m src.ingest.run_all [--schedule]``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        0 if every source succeeded and nothing is stale, otherwise 1.
    """
    parser = argparse.ArgumentParser(description="Run all AltSignal ingestion pipelines.")
    parser.add_argument("--schedule", action="store_true",
                        help="run now, then every day (APScheduler); stop with Ctrl+C")
    args = parser.parse_args(argv)
    setup_logging()

    if args.schedule:
        start_daily_schedule(run_all, config.INDEED_SCHEDULE_HOUR, config.INDEED_SCHEDULE_MINUTE)
        return 0

    results, stale = run_all()
    failed = [r for r in results if r.status != "success"]
    for r in results:
        print(f"{r.source:14s} {r.status:8s} rows={r.rows_returned}"
              + (f"  {r.error_message[:150]}" if r.error_message else ""))
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
