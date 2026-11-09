"""Ingest Indeed job postings for the active state's metro areas.

There is no public Indeed API. We use the open-source project
omkarcloud/indeed-scraper (https://github.com/omkarcloud/indeed-scraper). It
runs on YOUR machine as a small local web server and needs no API key:

    git clone https://github.com/omkarcloud/indeed-scraper
    cd indeed-scraper && python -m pip install -r requirements.txt
    python run.py                      # serves http://localhost:8000

This module calls that server (``config.INDEED_API_URL``). If the server is
not running, or Indeed blocks it (Indeed often blocks datacenter IP addresses;
a home/residential connection works best), the run is logged as failed with a
clear message in the ``scrapes`` table.

How the counts are built
------------------------
1. For every metro of the active state we ask for postings from the last 24
   hours (newest first), page by page.
2. Each posting is labelled with an Indeed "occupation group" (the first
   occupation name ending in "Occupations") which we call its SECTOR.
3. Postings are DEDUPLICATED BY JOB ID. A job ID that was ever stored before
   is ignored, so a posting is counted once, in the week we first saw it.
   The deduplicated postings live in ``data/raw/indeed/<ST>_postings.csv``.
4. Postings are counted per week (Monday-Sunday), metro and sector and stored
   in the ``signals`` table. ``quality_score`` is the share of the week's 7
   days on which we actually scraped.

Limits to keep in mind: the scraper returns at most 50 pages of 20 postings
per search and Indeed's own pagination often ends sooner, so large metros are
only partly captured: the fetched count is below the count Indeed reports (the
run notes this in ``error_message``), so the weekly counts are a SAMPLE of new
postings, not a census. A
metro is searched by city name plus a radius, which only approximates the
official metro area.

DISCLAIMER: do not build anything commercial on scraped Indeed data. See the
README.

Run once:      python -m src.ingest.indeed
Run daily:     python -m src.ingest.indeed --schedule
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import config
from src import db
from src.ingest.common import (IngestError, ScrapeResult, exit_code, make_session,
                               run_logged, setup_logging, start_daily_schedule)
from src.normalize import add_normalized_values

logger = logging.getLogger("altsignal")

UNCLASSIFIED = "Unclassified"
POSTING_COLUMNS = ["job_id", "first_seen", "metro", "sector", "date_published"]


def sector_of(job: dict) -> str:
    """Pick the sector label for one Indeed posting.

    Args:
        job: One job dict from the scraper (has an ``occupations`` list).

    Returns:
        The first occupation name that ends in "Occupations" (Indeed's broad
        occupation group), or "Unclassified" if there is none.
    """
    for name in job.get("occupations") or []:
        if name.endswith("Occupations"):
            return name
    return UNCLASSIFIED


def sector_slug(sector: str) -> str:
    """Turn a sector name into a short id fragment for ``signal_id``.

    Args:
        sector: e.g. "Food Preparation Occupations".

    Returns:
        A lowercase snake_case slug, e.g. "food_preparation".
    """
    name = re.sub(r"\s*Occupations$", "", sector)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fetch_page(session, location: str, page: int) -> dict:
    """Ask the local Indeed scraper for one page of postings.

    Args:
        session: A ``requests`` session.
        location: Search location, e.g. "Los Angeles, CA".
        page: Page number, starting at 1.

    Returns:
        The decoded JSON response (keys include ``jobs``, ``count``,
        ``total_pages``).

    Raises:
        IngestError: If the scraper cannot be reached or returns an error
            (for example because Indeed blocked the request).
    """
    import requests

    params = {
        "location": location,
        "radius": config.INDEED_RADIUS_MILES,
        "date_posted": "last_24_hours",
        "sort": "date",
        "page": page,
    }
    url = f"{config.INDEED_API_URL}/jobs/search"
    try:
        response = session.get(url, params=params, timeout=config.INDEED_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise IngestError(
            f"Cannot reach the Indeed scraper server at {config.INDEED_API_URL}. "
            "Start it first (clone omkarcloud/indeed-scraper and run `python run.py`), "
            f"or set INDEED_API_URL. Details: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise IngestError(f"Indeed scraper request failed for {location!r}: {exc}") from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        raise IngestError(
            f"Indeed scraper returned HTTP {response.status_code} for {location!r}: "
            f"{str(detail)[:300]} (Indeed may be blocking this IP address)"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise IngestError(f"Indeed scraper returned invalid JSON for {location!r}") from exc


def collect_metro(session, location: str, max_pages: int, pause: float = 0.0) -> tuple[list[dict], int | None]:
    """Fetch all available pages of recent postings for one metro.

    Args:
        session: A ``requests`` session.
        location: Search location, e.g. "Los Angeles, CA".
        max_pages: Never fetch more pages than this.
        pause: Seconds to wait between page requests.

    Returns:
        ``(jobs, reported_count)``: the postings fetched and the total number
        Indeed says match (``None`` if not reported).
    """
    jobs: list[dict] = []
    reported = None
    page = 1
    while page <= max_pages:
        data = fetch_page(session, location, page)
        reported = data.get("count", reported)
        page_jobs = data.get("jobs") or []
        if not page_jobs:
            break
        jobs.extend(page_jobs)
        if page >= int(data.get("total_pages") or 0):
            break
        page += 1
        if pause:
            time.sleep(pause)
    return jobs, reported


# ---------------------------------------------------------------------------
# Deduplicated posting store (a CSV file per state)
# ---------------------------------------------------------------------------
def postings_path(state_abbr: str):
    """Return the CSV path that stores deduplicated postings for a state.

    Args:
        state_abbr: Two-letter abbreviation, e.g. "CA".

    Returns:
        A ``pathlib.Path`` under ``config.INDEED_POSTINGS_DIR``.
    """
    return config.INDEED_POSTINGS_DIR / f"{state_abbr}_postings.csv"


def load_postings(state_abbr: str) -> pd.DataFrame:
    """Load every posting stored so far for a state.

    Args:
        state_abbr: Two-letter abbreviation.

    Returns:
        DataFrame with ``POSTING_COLUMNS`` (empty if nothing is stored yet).
    """
    path = postings_path(state_abbr)
    if not path.exists():
        return pd.DataFrame(columns=POSTING_COLUMNS)
    return pd.read_csv(path, dtype=str)


def add_new_postings(
    existing: pd.DataFrame,
    metro_jobs: dict[str, list[dict]],
    today: date,
) -> pd.DataFrame:
    """Deduplicate fresh postings against the ones already stored.

    A job ID is kept only the first time it is seen (across all metros and all
    earlier runs).

    Args:
        existing: Postings stored so far (``POSTING_COLUMNS``).
        metro_jobs: ``{metro name: [job dicts]}`` from this run.
        today: The date of this run (stored as ``first_seen``).

    Returns:
        DataFrame of the NEW postings only.
    """
    seen = set(existing["job_id"])
    new_rows = []
    for metro, jobs in metro_jobs.items():
        for job in jobs:
            job_id = job.get("id")
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)
            new_rows.append({
                "job_id": job_id,
                "first_seen": today.isoformat(),
                "metro": metro,
                "sector": sector_of(job),
                "date_published": job.get("date_published") or "",
            })
    return pd.DataFrame(new_rows, columns=POSTING_COLUMNS)


def save_postings(state_abbr: str, new_postings: pd.DataFrame) -> None:
    """Append new postings to the state's CSV file.

    Args:
        state_abbr: Two-letter abbreviation.
        new_postings: Rows to append (``POSTING_COLUMNS``).
    """
    if new_postings.empty:
        return
    path = postings_path(state_abbr)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_postings.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------------------------
def week_start(day: date) -> date:
    """Return the Monday of the week containing ``day``.

    Args:
        day: Any date.

    Returns:
        The Monday of that week.
    """
    return day - timedelta(days=day.weekday())


def weekly_counts(
    postings: pd.DataFrame,
    state_name: str,
    metro_names: list[str],
    observed_days: set[date],
) -> pd.DataFrame:
    """Count postings per week, metro and sector as ``signals`` rows.

    Only weeks in which we scraped at least once get a row. For such a week a
    metro/sector with no new postings is a real count of zero (we looked and
    found none), so it is stored as 0 rather than left out.

    Series produced (``signal_id``):
        ``indeed_postings_<sector>``  per metro and sector
        ``indeed_postings_total``     per metro, and for the whole state

    Args:
        postings: Deduplicated postings (``POSTING_COLUMNS``).
        state_name: Full state name (geo_name of the state-level total).
        metro_names: All metros of the state.
        observed_days: Dates on which a scrape ran.

    Returns:
        A DataFrame of signal rows (with ``normalized_value``), without
        ``fetched_at``. Empty if no week has been observed.
    """
    if not observed_days:
        return pd.DataFrame()

    days_per_week: dict[date, int] = {}
    for day in observed_days:
        days_per_week[week_start(day)] = days_per_week.get(week_start(day), 0) + 1
    weeks = sorted(days_per_week)

    posts = postings.copy()
    posts["week"] = [week_start(date.fromisoformat(d)) for d in posts["first_seen"]]
    counts = posts.groupby(["week", "metro", "sector"]).size().to_dict()

    # Every (metro, sector) pair ever seen, plus the totals.
    pairs = sorted({(m, s) for (_, m, s) in counts})
    records = []
    for week in weeks:
        quality = min(days_per_week[week] / 7, 1.0)
        base = {
            "source": config.SOURCE_INDEED,
            "period_start": pd.Timestamp(week),
            "period_end": pd.Timestamp(week + timedelta(days=6)),
            "unit": "postings",
            "quality_score": quality,
            "grain": "weekly",
        }
        state_total = 0
        for metro in metro_names:
            metro_total = 0
            for (m, sector) in pairs:
                if m != metro:
                    continue
                value = counts.get((week, m, sector), 0)
                metro_total += value
                records.append({**base, "signal_id": f"indeed_postings_{sector_slug(sector)}",
                                "geo_level": "metro", "geo_name": metro, "value": value})
            state_total += metro_total
            records.append({**base, "signal_id": "indeed_postings_total",
                            "geo_level": "metro", "geo_name": metro, "value": metro_total})
        records.append({**base, "signal_id": "indeed_postings_total",
                        "geo_level": "state", "geo_name": state_name, "value": state_total})

    return add_normalized_values(pd.DataFrame(records))


def observed_scrape_days(today: date) -> set[date]:
    """List the days on which an Indeed scrape finished (plus today).

    Args:
        today: The date of the current run (counts as observed).

    Returns:
        A set of UTC dates.
    """
    days = {today}
    log = db.read_scrapes(source=config.SOURCE_INDEED)
    log = log[log["status"].isin([db.STATUS_SUCCESS, db.STATUS_PARTIAL])]
    for stamp in log["finished_at"].dropna():
        days.add(datetime.fromisoformat(stamp).astimezone(timezone.utc).date())
    return days


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def ingest(state: str | None = None, session=None, max_pages: int | None = None,
           pause: float | None = None) -> ScrapeResult:
    """Scrape Indeed for the active state, store counts, and log the scrape.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.
        session: Optional ``requests`` session (used by the tests).
        max_pages: Max pages per metro. Defaults to ``config.INDEED_MAX_PAGES``.
        pause: Seconds between page requests. Defaults to the config value.

    Returns:
        The ``ScrapeResult`` (also written to the ``scrapes`` table).
    """
    max_pages = max_pages if max_pages is not None else config.INDEED_MAX_PAGES
    pause = pause if pause is not None else config.INDEED_PAUSE_SECONDS

    def job(result: ScrapeResult) -> None:
        info = config.get_state_info(state)
        http = session or make_session()
        today = datetime.now(timezone.utc).date()

        metro_jobs: dict[str, list[dict]] = {}
        errors: dict[str, str] = {}
        notes: list[str] = []
        for metro in info.metros:
            try:
                jobs, reported = collect_metro(http, metro.indeed_location, max_pages, pause)
            except IngestError as exc:
                errors[metro.name] = str(exc)
                continue
            metro_jobs[metro.name] = jobs
            if reported is not None and len(jobs) < reported:
                notes.append(f"{metro.name}: fetched {len(jobs)} of {reported}")

        result.metro_coverage = len(metro_jobs) / len(info.metros)
        if not metro_jobs:
            first_error = next(iter(errors.values()))
            raise IngestError(f"All {len(info.metros)} metros failed. First error: {first_error}")

        existing = load_postings(info.abbr)
        new_postings = add_new_postings(existing, metro_jobs, today)
        save_postings(info.abbr, new_postings)
        all_postings = pd.concat([existing, new_postings], ignore_index=True)

        rows = weekly_counts(all_postings, info.name, [m.name for m in info.metros],
                             observed_scrape_days(today))
        db.write_signals(rows)
        result.rows_returned = len(new_postings)

        if errors:
            result.status = db.STATUS_PARTIAL
            failed = "; ".join(f"{name}: {msg}" for name, msg in errors.items())
            result.error_message = f"{len(errors)} of {len(info.metros)} metros failed. {failed}"[:1000]
        elif notes:
            result.error_message = ("note, fewer postings were fetched than Indeed reports "
                                    "(the scraper's pagination ends early or hit the page limit) - "
                                    + "; ".join(notes))[:1000]

    return run_logged(config.SOURCE_INDEED, job)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m src.ingest.indeed [--schedule]``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Scrape Indeed postings for the active state.")
    parser.add_argument("--schedule", action="store_true",
                        help="run now, then every day (APScheduler); stop with Ctrl+C")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="limit pages per metro (useful for a quick test)")
    args = parser.parse_args(argv)
    setup_logging()

    if args.schedule:
        start_daily_schedule(lambda: ingest(max_pages=args.max_pages),
                             config.INDEED_SCHEDULE_HOUR, config.INDEED_SCHEDULE_MINUTE)
        return 0
    return exit_code(ingest(max_pages=args.max_pages))


if __name__ == "__main__":
    sys.exit(main())
