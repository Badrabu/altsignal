"""Ingest the Census Monthly State Retail Sales (MSRS) CSV files.

Source: https://www.census.gov/retail/state_retail_sales.html

IMPORTANT LIMITATION: MSRS is an EXPERIMENTAL Census product. The numbers are
MODELED (a blend of the Monthly Retail Trade Survey, administrative data and
third-party data), not a direct survey count by state. They are published as
not-seasonally-adjusted year-over-year percent changes, from January 2019.

Each CSV is "wide": one row per state and NAICS retail subsector, one column
per month (e.g. ``yy202601`` = January 2026). This module reshapes it into one
row per month and stores it in the ``signals`` table:

    signal_id      msrs_yoy_<NAICS>   (``msrs_yoy_TOTAL`` = all retail excl.
                                       nonstore retailers)
    value          year-over-year % change
    quality_score  Census "non-imputed coverage" class mapped to a number
                   (see COVERAGE_SCORE below)

Cells the Census marks "S" (suppressed for data quality) are skipped. They are
never filled in.

Run it:  python -m src.ingest.msrs
"""

from __future__ import annotations

import io
import logging
import sys

import pandas as pd

import config
from src import db
from src.ingest.common import IngestError, ScrapeResult, exit_code, make_session, run_logged, setup_logging
from src.normalize import add_normalized_values

logger = logging.getLogger("altsignal")

# Census coverage classes describe the share of the state/NAICS total that is
# NOT imputed (higher = better data). We store the middle of each class range.
COVERAGE_SCORE = {"A": 0.05, "B": 0.175, "C": 0.375, "D": 0.75}

NAICS_NAMES = {
    "TOTAL": "Total retail excl. nonstore retailers",
    "441": "Motor vehicle and parts dealers",
    "442": "Furniture and home furnishings stores",
    "443": "Electronics and appliance stores",
    "444": "Building materials and garden stores",
    "445": "Food and beverage stores",
    "446": "Health and personal care stores",
    "447": "Gasoline stations",
    "448": "Clothing and accessories stores",
    "451": "Sporting goods, hobby, book and music stores",
    "452": "General merchandise stores",
    "453": "Miscellaneous store retailers",
}


def download_csv(file_key: str, session=None) -> str:
    """Download one MSRS CSV file from census.gov and return its text.

    The raw file is also saved under ``data/raw/msrs/`` for reference.

    Args:
        file_key: Key of ``config.MSRS_FILES`` ("yy" or "coverage").
        session: Optional ``requests`` session (used by the tests).

    Returns:
        The CSV file contents as text.

    Raises:
        IngestError: If the download fails.
    """
    session = session or make_session()
    url = f"{config.MSRS_BASE_URL}/{config.MSRS_FILES[file_key]}"
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except Exception as exc:  # network error or HTTP error status
        raise IngestError(f"Could not download MSRS file {url}: {exc}") from exc
    if "fips" not in response.text[:200].lower():
        raise IngestError(f"MSRS file {url} does not look like a CSV (is the layout changing?)")

    raw_dir = config.RAW_DIR / "msrs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / config.MSRS_FILES[file_key]).write_text(response.text, encoding="utf-8")
    return response.text


def _melt(csv_text: str, prefix: str, state_abbr: str, value_name: str) -> pd.DataFrame:
    """Reshape one wide MSRS CSV into long format for a single state.

    Args:
        csv_text: Contents of the CSV file.
        prefix: Column prefix of the month columns ("yy" or "cov").
        state_abbr: Keep only this state's rows (e.g. "CA").
        value_name: Name of the value column in the result.

    Returns:
        DataFrame with columns ``naics``, ``period_start`` and ``value_name``.
        Suppressed cells ("S") are dropped.
    """
    wide = pd.read_csv(io.StringIO(csv_text), dtype=str)
    wide.columns = [c.strip() for c in wide.columns]
    wide = wide[wide["stateabbr"].str.strip() == state_abbr]
    month_cols = [c for c in wide.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    long = wide.melt(id_vars=["naics"], value_vars=month_cols,
                     var_name="column", value_name=value_name)
    long["period_start"] = pd.to_datetime(long["column"].str[len(prefix):], format="%Y%m")
    long[value_name] = long[value_name].str.strip()
    long = long[long[value_name] != "S"]          # suppressed: skip, never impute
    long = long[long[value_name].notna() & (long[value_name] != "")]
    return long[["naics", "period_start", value_name]]


def parse_msrs(yy_csv: str, coverage_csv: str, state_abbr: str) -> pd.DataFrame:
    """Turn the MSRS CSV files into ``signals`` rows for one state.

    Args:
        yy_csv: Text of ``state_retail_yy.csv`` (year-over-year % changes).
        coverage_csv: Text of ``state_retail_coverage.csv``.
        state_abbr: Two-letter state abbreviation, e.g. "CA".

    Returns:
        A DataFrame of signal rows (without ``fetched_at``). Empty if the state
        is not in the file.
    """
    values = _melt(yy_csv, "yy", state_abbr, "value")
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    values = values.dropna(subset=["value"])
    coverage = _melt(coverage_csv, "cov", state_abbr, "coverage")

    merged = values.merge(coverage, on=["naics", "period_start"], how="left")
    state_name = config.get_state_info(state_abbr).name

    rows = pd.DataFrame({
        "signal_id": "msrs_yoy_" + merged["naics"],
        "source": config.SOURCE_MSRS,
        "geo_level": "state",
        "geo_name": state_name,
        "period_start": merged["period_start"],
        "period_end": merged["period_start"] + pd.offsets.MonthEnd(0),
        "value": merged["value"],
        "unit": "percent_yoy",
        # Unknown coverage class -> empty quality score (not a guess).
        "quality_score": merged["coverage"].map(COVERAGE_SCORE),
        "grain": "monthly",
    })
    if rows.empty:
        return rows
    rows = add_normalized_values(rows)
    return rows.sort_values(["signal_id", "period_start"]).reset_index(drop=True)


def ingest(state: str | None = None, session=None) -> ScrapeResult:
    """Download MSRS, store the active state's rows, and log the scrape.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.
        session: Optional ``requests`` session (used by the tests).

    Returns:
        The ``ScrapeResult`` (also written to the ``scrapes`` table).
    """

    def job(result: ScrapeResult) -> None:
        info = config.get_state_info(state)
        yy_csv = download_csv("yy", session)
        coverage_csv = download_csv("coverage", session)
        rows = parse_msrs(yy_csv, coverage_csv, info.abbr)
        if rows.empty:
            raise IngestError(f"No MSRS rows found for {info.name} ({info.abbr}) in the Census files.")
        result.rows_returned = db.write_signals(rows)

    return run_logged(config.SOURCE_MSRS, job)


def main() -> int:
    """Command-line entry point: ``python -m src.ingest.msrs``.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging()
    return exit_code(ingest())


if __name__ == "__main__":
    sys.exit(main())
