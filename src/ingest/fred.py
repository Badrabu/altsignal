"""Ingest series from FRED (Federal Reserve Economic Data) with ``fredapi``.

Two separate sources, each with its own row in the ``scrapes`` table:

    fred_targets  The official numbers we will later try to nowcast:
                  state unemployment rate (BLS, e.g. ``CAUR``), national
                  unemployment rate (``UNRATE``) as a fallback, and national
                  retail sales (Census advance report, ``RSAFS``).
    fred_card     Card spending index. NATIONAL ONLY, used only as a secondary
                  cross-check, never as the state-level consumption input.
                  The series ID is configured in ``config.CARD_SPENDING_SERIES_ID``
                  (see blockers.md - no verified SF Fed series ID exists on
                  FRED, so none is guessed).

You need a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)
in the ``FRED_API_KEY`` environment variable. Never commit the key.

Run it:  python -m src.ingest.fred
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

import config
from src import db
from src.ingest.common import IngestError, ScrapeResult, exit_code, run_logged, setup_logging
from src.normalize import add_normalized_values

logger = logging.getLogger("altsignal")


def get_fred_client(api_key: str | None = None):
    """Create a ``fredapi.Fred`` client from the environment.

    Args:
        api_key: Optional key. Defaults to the ``FRED_API_KEY`` variable.

    Returns:
        A ``fredapi.Fred`` object.

    Raises:
        IngestError: If no API key is available or the library is missing.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise IngestError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and export FRED_API_KEY=<key>."
        )
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise IngestError("The fredapi package is not installed (pip install -r requirements.txt).") from exc
    return Fred(api_key=key)


def fetch_series(client, series_id: str) -> pd.Series:
    """Download one series from FRED, turning key problems into clear errors.

    Args:
        client: A ``fredapi.Fred`` client.
        series_id: FRED series ID, e.g. "UNRATE".

    Returns:
        The series (indexed by date).

    Raises:
        IngestError: If FRED rejects the API key (affects every series).
        ValueError: If the series does not exist (the caller decides what to do).
    """
    try:
        return client.get_series(series_id)
    except ValueError as exc:
        if "api_key" in str(exc).lower():
            raise IngestError(f"FRED rejected the API key: {exc}") from exc
        raise


def series_to_rows(
    series: pd.Series,
    signal_id: str,
    geo_level: str,
    geo_name: str,
    unit: str,
    source: str,
) -> pd.DataFrame:
    """Convert a monthly FRED series into ``signals`` rows.

    Args:
        series: FRED data indexed by date (NaN for missing observations).
        signal_id: Series name for the ``signals`` table.
        geo_level: "state" or "national".
        geo_name: Place name (state name or "United States").
        unit: Unit label, e.g. "percent".
        source: Source name for the ``signals`` table.

    Returns:
        A DataFrame of signal rows, with ``normalized_value`` filled in.

    Raises:
        IngestError: If the series is not monthly (aggregation is a later phase).
    """
    clean = series.dropna()
    if clean.empty:
        return pd.DataFrame()
    starts = pd.to_datetime(clean.index)
    if len(starts) > 1:
        median_gap = pd.Series(starts).diff().dropna().dt.days.median()
        if not 27 <= median_gap <= 32:
            raise IngestError(f"{signal_id} is not a monthly series (median gap {median_gap} days); "
                              "convert it to monthly before ingesting.")
    rows = pd.DataFrame({
        "signal_id": signal_id,
        "source": source,
        "geo_level": geo_level,
        "geo_name": geo_name,
        "period_start": starts,
        "period_end": starts + pd.offsets.MonthEnd(0),
        "value": clean.astype(float).to_numpy(),
        "unit": unit,
        "quality_score": None,
        "grain": "monthly",
    })
    return add_normalized_values(rows)


def ingest_targets(state: str | None = None, fred=None) -> ScrapeResult:
    """Fetch the official target series and log the scrape.

    The state unemployment series is looked up as ``<ABBR>UR`` (e.g. ``CAUR``).
    If FRED has no such series the national series remain as the fallback, and
    the run is recorded as "partial".

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.
        fred: Optional FRED client (used by tests).

    Returns:
        The ``ScrapeResult`` (also written to the ``scrapes`` table).
    """

    def job(result: ScrapeResult) -> None:
        info = config.get_state_info(state)
        client = fred or get_fred_client()
        frames, missing = [], []
        for signal_id, (template, geo_level, unit) in config.FRED_TARGET_SERIES.items():
            series_id = template.format(abbr=info.abbr)
            try:
                series = fetch_series(client, series_id)
            except ValueError as exc:   # fredapi raises ValueError for unknown series
                missing.append(f"{series_id} ({exc})")
                continue
            geo_name = info.name if geo_level == "state" else "United States"
            rows = series_to_rows(series, f"fred_{signal_id}", geo_level, geo_name,
                                  unit, config.SOURCE_FRED_TARGETS)
            if not rows.empty:
                frames.append(rows)
        if not frames:
            raise IngestError(f"No FRED target series could be fetched: {'; '.join(missing)}")
        result.rows_returned = db.write_signals(pd.concat(frames, ignore_index=True))
        if missing:
            result.status = db.STATUS_PARTIAL
            result.error_message = f"Series not found on FRED: {'; '.join(missing)}"[:1000]

    return run_logged(config.SOURCE_FRED_TARGETS, job)


def ingest_card_spending(fred=None, series_id: str | None = None) -> ScrapeResult:
    """Fetch the national card spending index and log the scrape.

    Args:
        fred: Optional FRED client (used by tests).
        series_id: FRED series ID. Defaults to ``config.CARD_SPENDING_SERIES_ID``.

    Returns:
        The ``ScrapeResult`` (also written to the ``scrapes`` table).
    """

    def job(result: ScrapeResult) -> None:
        sid = series_id or config.CARD_SPENDING_SERIES_ID
        if not sid:
            raise IngestError(
                "No card spending series configured. There is no verified SF Fed card "
                "spending series on FRED (see blockers.md). Set CARD_SPENDING_SERIES_ID "
                "(environment variable or config.py) to a confirmed FRED series ID."
            )
        client = fred or get_fred_client()
        try:
            series = fetch_series(client, sid)
        except ValueError as exc:
            raise IngestError(f"FRED series {sid!r} not found: {exc}") from exc
        rows = series_to_rows(series, "card_spending_index", "national", "United States",
                              config.CARD_SPENDING_UNIT, config.SOURCE_FRED_CARD)
        if rows.empty:
            raise IngestError(f"FRED series {sid!r} returned no observations.")
        result.rows_returned = db.write_signals(rows)

    return run_logged(config.SOURCE_FRED_CARD, job)


def main() -> int:
    """Command-line entry point: ``python -m src.ingest.fred``.

    Runs both FRED sources (targets, then card spending).

    Returns:
        Process exit code (0 only if both succeeded).
    """
    setup_logging()
    return exit_code(ingest_targets(), ingest_card_spending())


if __name__ == "__main__":
    sys.exit(main())
