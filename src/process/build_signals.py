"""Phase two: clean, align and finalise the ``signals`` table for the active state.

Run it after the ingestion pipelines:

    python -m src.process.build_signals

For the active ``STATE`` (and the national series) it:

1. Reads what ingestion stored (MSRS, VIIRS, FRED targets, optional card
   spending) and checks that no series mixes units.
2. Cleans it (``clean.py``): drops rows with no value, misaligned months,
   implausible values and VIIRS months with more than 20% low coverage. Dropped
   rows are also removed from the table. Nothing is imputed.
3. Aligns Indeed to the monthly grain (``align.py``). The weekly rows stay.
4. Recomputes ``normalized_value``: the z-score of each series against its OWN
   previous 12 months (never mixing signals or units).
5. Writes everything back with ``quality_score`` and ``grain`` filled in.

It is IDEMPOTENT: the table is unique on (signal_id, geo_level, geo_name,
period_start, grain), so running it again updates rows in place and never
creates duplicates. Sources that have no data yet (for example VIIRS before
Earth Engine is set up) are reported and skipped; the optional card spending
source is skipped quietly.

``quality_score`` is a number from 0 to 1 whose meaning depends on the source
(see the README). Scores from different sources are not comparable.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src import db
from src.ingest import indeed
from src.normalize import add_normalized_values
from src.process import align, clean

logger = logging.getLogger("altsignal")

# Sources whose rows are already monthly in the table.
NATIVE_MONTHLY_SOURCES = (
    config.SOURCE_MSRS,
    config.SOURCE_VIIRS,
    config.SOURCE_FRED_TARGETS,
    config.SOURCE_FRED_CARD,
)
# Sources that may legitimately be absent.
OPTIONAL_SOURCES = (config.SOURCE_FRED_CARD,)


@dataclass
class SourceReport:
    """What happened to one source during the build.

    Attributes:
        source: Source name.
        status: "built", "no data" (nothing ingested yet), "skipped" (optional
            source not configured) or "error".
        rows_read: Rows read from the table (or built, for Indeed).
        rows_dropped: Dropped rows per reason.
        rows_written: Rows written back.
        message: Extra explanation (errors, hints).
    """

    source: str
    status: str = "built"
    rows_read: int = 0
    rows_dropped: dict[str, int] = field(default_factory=dict)
    rows_written: int = 0
    message: str = ""


def in_scope(signals: pd.DataFrame, info: config.StateInfo) -> pd.DataFrame:
    """Keep the rows that belong to the active state, plus national rows.

    Args:
        signals: Rows read from the ``signals`` table.
        info: The active state's lookup information.

    Returns:
        The rows whose geography is national, the state itself or one of its
        metros. Rows of other states are ignored.
    """
    geo_names = {info.name} | {m.geo_name for m in info.metros}
    keep = (signals["geo_level"] == "national") | signals["geo_name"].isin(geo_names)
    return signals[keep].copy()


def add_zscores(signals: pd.DataFrame) -> pd.DataFrame:
    """Set ``normalized_value`` to each series' rolling z-score.

    Each series (signal_id + geography + grain) is compared with its own past
    only. Rows whose ``quality_score`` is below the source's minimum
    (``config.MIN_QUALITY_FOR_ZSCORE``, e.g. partly scraped Indeed months) are
    treated as missing, so they get no z-score and never enter a baseline.

    Args:
        signals: Cleaned rows with ``source``, ``value`` and ``quality_score``.

    Returns:
        A copy with ``normalized_value`` filled in (empty where there is no
        complete 12-month baseline).
    """
    if signals.empty:
        return signals.copy()
    masked = signals.copy()
    for source, minimum in config.MIN_QUALITY_FOR_ZSCORE.items():
        weak = (masked["source"] == source) & (masked["quality_score"] < minimum - 1e-9)
        masked.loc[weak, "value"] = np.nan
    scored = add_normalized_values(masked)
    result = signals.copy()
    result["normalized_value"] = scored["normalized_value"]
    return result


def finalise(rows: pd.DataFrame, report: SourceReport) -> pd.DataFrame:
    """Clean, drop bad rows from the table, add z-scores and write the result.

    Args:
        rows: Signal rows for one source.
        report: The source's report, updated in place.

    Returns:
        The rows that were written.

    Raises:
        clean.ProcessError: If a series mixes units.
    """
    clean.check_units(rows)
    cleaned = clean.clean_signals(rows)
    report.rows_dropped = cleaned.reasons()

    # Remove rows that broke a rule from the table too (duplicates excluded,
    # because the "duplicate" has the same key as the row we keep).
    to_delete = cleaned.dropped[cleaned.dropped["reason"] != "duplicate key"] if not cleaned.dropped.empty else cleaned.dropped
    db.delete_signals(to_delete)

    scored = add_zscores(cleaned.kept)
    report.rows_written = db.write_signals(scored)
    return scored


def build_native_source(source: str, info: config.StateInfo) -> SourceReport:
    """Process one source whose rows are already monthly in the table.

    Args:
        source: Source name (e.g. "msrs").
        info: The active state's lookup information.

    Returns:
        The source's ``SourceReport``.
    """
    report = SourceReport(source)
    rows = in_scope(db.read_signals(source=source), info)
    rows = rows[rows["grain"] == "monthly"]
    report.rows_read = len(rows)
    if rows.empty:
        if source in OPTIONAL_SOURCES:
            report.status, report.message = "skipped", "optional source, nothing ingested"
        else:
            report.status = "no data"
            report.message = f"nothing ingested yet; run `python -m src.ingest.{_ingest_module(source)}`"
        return report
    try:
        finalise(rows, report)
    except clean.ProcessError as exc:
        report.status, report.message = "error", str(exc)
    return report


def _ingest_module(source: str) -> str:
    """Name the ingest module that fills a source (used in hints).

    Args:
        source: Source name.

    Returns:
        The module name under ``src.ingest``.
    """
    return "fred" if source.startswith("fred") else source


def build_indeed(info: config.StateInfo) -> SourceReport:
    """Build the monthly Indeed series from the deduplicated postings.

    Args:
        info: The active state's lookup information.

    Returns:
        The source's ``SourceReport``.
    """
    report = SourceReport(config.SOURCE_INDEED)
    postings = indeed.load_postings(info.abbr)
    observed = indeed.observed_scrape_days()
    if postings.empty or not observed:
        report.status = "no data"
        report.message = "no Indeed postings or scrapes yet; run `python -m src.ingest.indeed`"
        return report
    rows = align.indeed_monthly_rows(postings, info.name, list(info.metros), observed)
    report.rows_read = len(rows)
    try:
        finalise(rows, report)
    except clean.ProcessError as exc:
        report.status, report.message = "error", str(exc)
    return report


def verify(info: config.StateInfo) -> list[str]:
    """Run final sanity checks on everything stored for the active state.

    Args:
        info: The active state's lookup information.

    Returns:
        A list of problems found (empty if all is well).
    """
    problems = []
    table = in_scope(db.read_signals(), info)
    try:
        clean.check_units(table)
    except clean.ProcessError as exc:
        problems.append(str(exc))
    if table.duplicated(subset=clean.KEY_COLUMNS).any():
        problems.append("duplicate signal rows found")
    return problems


def build_signals(state: str | None = None) -> tuple[list[SourceReport], list[str]]:
    """Build the cleaned, aligned signals for one state.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.

    Returns:
        ``(reports, problems)``: one ``SourceReport`` per source and the
        problems found by the final verification.
    """
    info = config.get_state_info(state)
    db.init_db()
    reports = [build_native_source(source, info) for source in NATIVE_MONTHLY_SOURCES]
    reports.append(build_indeed(info))
    return reports, verify(info)


def main() -> int:
    """Command-line entry point: ``python -m src.process.build_signals``.

    Returns:
        0 if at least one source was built and nothing is wrong, otherwise 1.
        Sources that simply have no data yet do not make the build fail.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    info = config.get_state_info()
    print(f"Building signals for {info.name} ({info.abbr})")
    reports, problems = build_signals()

    print(f"{'source':14s} {'status':8s} {'read':>6s} {'written':>8s}  notes")
    for r in reports:
        notes = r.message
        if r.rows_dropped:
            notes = (notes + " " if notes else "") + "dropped: " + ", ".join(
                f"{n} x {reason}" for reason, n in r.rows_dropped.items())
        print(f"{r.source:14s} {r.status:8s} {r.rows_read:6d} {r.rows_written:8d}  {notes}")
    for problem in problems:
        print(f"PROBLEM: {problem}")

    built = any(r.status == "built" for r in reports)
    errors = any(r.status == "error" for r in reports) or problems
    return 0 if built and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
