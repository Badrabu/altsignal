"""Data-loading helpers for the Streamlit app.

Everything is read through ``src/db.py`` (the app never writes SQL and never
writes to the database). The functions take plain arguments and return pandas
tables, so they can be unit-tested without Streamlit.

Rule of the whole app: a source that has no data is shown as "no data yet".
Nothing here fills, interpolates or invents values.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
from src import db
from src.ingest.msrs import NAICS_NAMES

# What each source needs before it has data. Shown when a source has no rows.
# (Facts taken from README.md and blockers.md.)
SOURCE_CATALOGUE = [
    {"source": config.SOURCE_MSRS, "title": "Census Monthly State Retail Sales (MSRS)",
     "no_data": "No MSRS rows stored yet. Run `python -m src.ingest.msrs`, then `python -m src.process.build_signals`."},
    {"source": config.SOURCE_INDEED, "title": "Indeed job postings",
     "no_data": ("No Indeed data yet. It needs the locally run scraper, and a z-score needs 12 months "
                 "of history, which does not exist yet.")},
    {"source": config.SOURCE_VIIRS, "title": "VIIRS nighttime lights",
     "no_data": ("No VIIRS data yet. It needs a free Google Earth Engine project and credentials "
                 "(blocked so far, see blockers.md).")},
    {"source": config.SOURCE_FRED_CARD, "title": "Card spending index (national cross-check)",
     "no_data": ("No card spending data yet. No verified FRED series ID exists for the SF Fed card "
                 "spending index (see blockers.md).")},
    {"source": config.SOURCE_FRED_TARGETS, "title": "Official targets from FRED",
     "no_data": ("No FRED data yet. Run `python -m src.ingest.fred` with a free `FRED_API_KEY` "
                 "(not in this database, see the README).")},
]

TARGET_TITLES = {"retail_sales": "Retail sales (Census MSRS, year-over-year %)",
                 "unemployment": "Unemployment rate (FRED, %)"}
TARGET_NO_DATA = {
    "retail_sales": "No retail sales nowcasts yet. Run build_features and run_models (see README).",
    "unemployment": ("No unemployment nowcasts yet: the official series comes from FRED, which needs "
                     "`FRED_API_KEY` (not in this database); run the FRED ingest, build_features and run_models."),
}

UNIT_LABELS = {"percent_yoy": "% year-over-year", "percent": "%", "postings": "postings",
               "nW/cm2/sr": "nW/cm2/sr", "millions_usd": "million USD", "index": "index"}


@dataclass(frozen=True)
class DatabaseChoice:
    """Which database the app reads and why.

    Attributes:
        path: The SQLite file to read.
        is_demo: True when this is the committed demo snapshot.
        note: One sentence for the user explaining the choice.
    """

    path: Path
    is_demo: bool
    note: str


def database_has_signals(path: str | Path) -> bool:
    """Check, without creating or changing anything, whether a database has signal rows.

    Args:
        path: Database file.

    Returns:
        True if the file exists and its ``signals`` table has at least one row.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        return not db.read_signals(db_path=path).empty
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return False


def choose_database() -> DatabaseChoice:
    """Pick the database to display.

    The live database (``config.DB_PATH``, built by the pipelines) is used when
    it exists and holds signals. Otherwise the app falls back to the committed
    demo snapshot, which is how a deployed copy gets data without credentials.

    Returns:
        A ``DatabaseChoice``.
    """
    live = Path(config.DB_PATH)
    if database_has_signals(live):
        return DatabaseChoice(live, False, f"Reading the live database `{_display_path(live)}`.")
    demo = Path(config.DEMO_DB_PATH)
    if database_has_signals(demo):
        return DatabaseChoice(
            demo, True,
            "Showing the bundled **demo snapshot** (a small export of public data; no live database found). "
            "See the README for how it is refreshed.")
    return DatabaseChoice(demo, True, "No database with data was found (neither the live one nor the demo snapshot).")


def _display_path(path: Path) -> str:
    """Show a path relative to the project root when possible.

    Args:
        path: Any path.

    Returns:
        The relative path as text, or the absolute path if it is outside the project.
    """
    try:
        return str(path.relative_to(config.ROOT_DIR))
    except ValueError:
        return str(path)


def _safe(read, *args, **kwargs) -> pd.DataFrame:
    """Call a ``db`` read function, returning an empty table if the tables do not exist.

    Args:
        read: A function from ``src/db.py``.
        *args: Positional arguments for it.
        **kwargs: Keyword arguments for it.

    Returns:
        Its table, or an empty ``DataFrame``.
    """
    try:
        return read(*args, **kwargs)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return pd.DataFrame()


def load_signals(state_name: str, db_path: str | Path) -> pd.DataFrame:
    """Load the monthly state-level and national signals for the active state.

    Metro rows and non-monthly grains (weekly Indeed counts) are left out: the
    app compares signals on the monthly grain only.

    Args:
        state_name: Full state name, e.g. "California".
        db_path: Database file.

    Returns:
        Signal rows with ``period_start`` and ``period_end`` as datetimes and
        extra columns ``label`` and ``geo_scope``; empty if there are none.
    """
    state = _safe(db.read_signals, geo_level="state", geo_name=state_name, db_path=db_path)
    national = _safe(db.read_signals, geo_level="national", db_path=db_path)
    frames = [f for f in (state, national) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=db.SIGNAL_COLUMNS + ["label", "geo_scope"])
    signals = pd.concat(frames, ignore_index=True)
    signals = signals[signals["grain"] == "monthly"].copy()
    for col in ("period_start", "period_end"):
        signals[col] = pd.to_datetime(signals[col])
    signals["label"] = [signal_label(s) for s in signals["signal_id"]]
    signals["geo_scope"] = signals["geo_level"].map({"state": "State", "national": "National"})
    return signals.sort_values(["signal_id", "geo_level", "period_start"]).reset_index(drop=True)


def signal_label(signal_id: str) -> str:
    """Turn a ``signal_id`` into a readable name.

    Args:
        signal_id: For example "msrs_yoy_445" or "viirs_avg_radiance".

    Returns:
        For example "Retail: Food and beverage stores (445)"; unknown ids are
        returned unchanged rather than guessed at.
    """
    if signal_id.startswith("msrs_yoy_"):
        code = signal_id.removeprefix("msrs_yoy_")
        name = NAICS_NAMES.get(code)
        if name is None:
            return signal_id
        return "Retail: " + (name if code == "TOTAL" else f"{name} ({code})")
    fixed = {
        "viirs_avg_radiance": "Nighttime lights: average radiance",
        "indeed_postings_total": "Indeed postings: total",
        "card_spending_index": "Card spending index",
        "fred_unemployment_rate_state": "Unemployment rate (state)",
        "fred_unemployment_rate_national": "Unemployment rate (national)",
        "fred_retail_sales_national": "Retail sales (national)",
    }
    if signal_id in fixed:
        return fixed[signal_id]
    if signal_id.startswith("indeed_postings_"):
        return "Indeed postings: " + signal_id.removeprefix("indeed_postings_").replace("_", " ")
    return signal_id


def source_status(signals: pd.DataFrame, scrapes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Say, per expected source, whether it has data in the signals table.

    Args:
        signals: The table from ``load_signals``.
        scrapes: Optional scrape log (``db.read_scrapes``) to report the last run.

    Returns:
        One row per source in ``SOURCE_CATALOGUE`` (plus any other source found
        in the data) with columns ``source``, ``title``, ``has_data``, ``n_series``,
        ``rows``, ``latest_period``, ``last_scrape`` and ``no_data``.
    """
    rows = []
    known = {entry["source"] for entry in SOURCE_CATALOGUE}
    entries = list(SOURCE_CATALOGUE)
    if not signals.empty:
        entries += [{"source": s, "title": s, "no_data": ""} for s in sorted(set(signals["source"]) - known)]
    for entry in entries:
        part = signals[signals["source"] == entry["source"]] if not signals.empty else signals
        last_scrape = None
        if scrapes is not None and not scrapes.empty:
            runs = scrapes[scrapes["source"] == entry["source"]]
            if not runs.empty:
                latest = runs.iloc[0]          # read_scrapes returns newest first
                last_scrape = f"{latest['status']} ({str(latest['finished_at'] or latest['started_at'])[:10]})"
        rows.append({
            "source": entry["source"], "title": entry["title"], "has_data": not part.empty,
            "n_series": int(part["signal_id"].nunique()) if not part.empty else 0,
            "rows": int(len(part)),
            "latest_period": part["period_start"].max() if not part.empty else pd.NaT,
            "last_scrape": last_scrape, "no_data": entry["no_data"]})
    return pd.DataFrame(rows)


def snapshot_time(signals: pd.DataFrame) -> str | None:
    """Return when the newest signal row was fetched.

    Args:
        signals: The table from ``load_signals``.

    Returns:
        A date text such as "2026-11-10", or ``None`` if there are no rows.
    """
    if signals.empty:
        return None
    return str(signals["fetched_at"].max())[:10]


def wide_series(signals: pd.DataFrame, value_col: str = "normalized_value") -> pd.DataFrame:
    """Reshape signals to one column per series and one row per month.

    Months where a series has no value stay empty (``NaN``); nothing is filled.

    Args:
        signals: The table from ``load_signals``.
        value_col: ``"value"`` (raw) or ``"normalized_value"`` (z-score).

    Returns:
        A table indexed by month start, with ``signal_id`` columns.
    """
    if signals.empty:
        return pd.DataFrame()
    return signals.pivot_table(index="period_start", columns="signal_id", values=value_col, aggfunc="first", dropna=False).sort_index()


def load_models(state_abbr: str, target: str, db_path: str | Path) -> pd.DataFrame:
    """Load the metric rows (MAE, RMSE, CV_RMSE) of one target's models for a state.

    Args:
        state_abbr: Two-letter abbreviation (model ids start with it, e.g. "CA_").
        target: Target name, e.g. "retail_sales".
        db_path: Database file.

    Returns:
        Rows of the ``models`` table; empty if there are none.
    """
    models = _safe(db.read_models, db_path=db_path)
    if models.empty:
        return models
    mask = models["model_id"].str.startswith(f"{state_abbr}_") & (models["target"] == target)
    return models[mask].reset_index(drop=True)


def load_forecasts(model_ids: list[str], db_path: str | Path) -> pd.DataFrame:
    """Load every stored forecast of the given models (scored and still pending).

    Args:
        model_ids: Model ids to keep.
        db_path: Database file.

    Returns:
        Rows of the ``forecasts`` table with ``target_period`` as a datetime and
        a ``model`` column (the model id without the state/target prefix);
        empty if there are none.
    """
    forecasts = _safe(db.read_forecasts, db_path=db_path)
    if forecasts.empty:
        return forecasts
    forecasts = forecasts[forecasts["model_id"].isin(model_ids)].copy()
    forecasts["target_period"] = pd.to_datetime(forecasts["target_period"])
    return forecasts.sort_values(["model_id", "target_period"]).reset_index(drop=True)


def official_series(target: str, signals: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Find the official series behind a target, using the same choice as phase three.

    Args:
        target: Target name, e.g. "retail_sales".
        signals: The table from ``load_signals``.

    Returns:
        ``(rows, signal_id)`` for the first candidate in ``config.FEATURE_TARGETS``
        that has data, or an empty table and ``None``.
    """
    for signal_id, geo_level in config.FEATURE_TARGETS.get(target, []):
        part = signals[(signals["signal_id"] == signal_id) & (signals["geo_level"] == geo_level)]
        if not part.empty:
            return part.sort_values("period_start").reset_index(drop=True), signal_id
    return pd.DataFrame(columns=signals.columns), None


def model_short_name(model_id: str, state_abbr: str, target: str) -> str:
    """Strip the state and target prefix from a model id.

    Args:
        model_id: For example "CA_retail_sales_ridge".
        state_abbr: Two-letter abbreviation.
        target: Target name.

    Returns:
        For example "ridge".
    """
    return model_id.removeprefix(f"{state_abbr}_{target}_")
