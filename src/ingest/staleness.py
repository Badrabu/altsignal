"""Staleness alert: has every source scraped successfully in the last 48 hours?

The spec requires an alert if any source has not had a successful scrape in 48
hours. A source that has NEVER succeeded is also reported as stale. A "partial"
run does not count as a success.

Run it:  python -m src.ingest.staleness      (exit code 1 if anything is stale)

For the alert to stay quiet, every source must succeed at least every 48 hours,
so schedule ``python -m src.ingest.run_all --schedule`` (or run it from cron).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import config
from src import db
from src.ingest.common import setup_logging

logger = logging.getLogger("altsignal")


@dataclass
class StaleSource:
    """A source that needs attention.

    Attributes:
        source: Source name, e.g. "indeed".
        last_success: When it last succeeded, or ``None`` if never.
        age_hours: Hours since the last success (``None`` if never).
    """

    source: str
    last_success: datetime | None
    age_hours: float | None

    def message(self) -> str:
        """Describe the problem in one readable sentence.

        Returns:
            For example "ALERT [indeed]: no successful scrape for 60.0 hours".
        """
        if self.last_success is None:
            return f"ALERT [{self.source}]: no successful scrape has ever been recorded"
        return (f"ALERT [{self.source}]: no successful scrape for {self.age_hours:.1f} hours "
                f"(last success {self.last_success.isoformat(timespec='minutes')})")


def check_staleness(
    sources: tuple[str, ...] | None = None,
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> list[StaleSource]:
    """Find sources without a successful scrape inside the time limit.

    Args:
        sources: Sources to check. Defaults to ``config.ALL_SOURCES``.
        max_age_hours: Allowed age. Defaults to ``config.STALENESS_HOURS`` (48).
        now: Current time (UTC); only set by tests.

    Returns:
        One ``StaleSource`` per problem source (empty list = all fresh).
    """
    sources = sources or config.ALL_SOURCES
    limit = timedelta(hours=max_age_hours if max_age_hours is not None else config.STALENESS_HOURS)
    now = now or datetime.now(timezone.utc)
    db.init_db()

    stale = []
    for source in sources:
        last = db.last_successful_scrape(source)
        if last is None:
            stale.append(StaleSource(source, None, None))
        elif now - last > limit:
            stale.append(StaleSource(source, last, (now - last).total_seconds() / 3600))
    return stale


def report_staleness(stale: list[StaleSource]) -> None:
    """Log one error line per stale source (or an all-clear message).

    Args:
        stale: Output of ``check_staleness``.
    """
    if not stale:
        logger.info("Staleness check OK: every source succeeded within %d hours.", config.STALENESS_HOURS)
    for item in stale:
        logger.error(item.message())


def main() -> int:
    """Command-line entry point: ``python -m src.ingest.staleness``.

    Returns:
        1 if any source is stale, otherwise 0.
    """
    setup_logging()
    stale = check_staleness()
    report_staleness(stale)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
