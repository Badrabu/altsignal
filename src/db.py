"""All database access for AltSignal lives in this one module.

The rest of the code never writes SQL. It calls the read/write functions
below, so SQLite could later be swapped for Postgres by changing only this
file.

The database has four tables (see docs/altsignal-spec.md):
    signals    one row per signal value (raw value + normalized z-score)
    scrapes    one row per ingestion run (success or failure)
    models     one row per model metric (used from phase four)
    forecasts  one row per model and target period (used from phase four)

Timestamps are stored as ISO-8601 text in UTC (e.g. "2026-11-09T18:00:00+00:00")
and dates as "YYYY-MM-DD" text, which SQLite sorts correctly.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

import config

# SQL that creates the four tables. "IF NOT EXISTS" makes init_db() safe to
# call every time.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id        TEXT NOT NULL,
    source           TEXT NOT NULL,
    geo_level        TEXT NOT NULL,
    geo_name         TEXT NOT NULL,
    period_start     DATE NOT NULL,
    period_end       DATE NOT NULL,
    value            REAL,
    normalized_value REAL,
    unit             TEXT,
    quality_score    REAL,
    grain            TEXT NOT NULL,
    fetched_at       TIMESTAMP NOT NULL
);

-- One value per series, place, period and grain: re-running an ingest
-- updates rows instead of duplicating them.
CREATE UNIQUE INDEX IF NOT EXISTS ux_signals_series_period
    ON signals (signal_id, geo_level, geo_name, period_start, grain);

CREATE TABLE IF NOT EXISTS scrapes (
    scrape_id     INTEGER PRIMARY KEY,
    source        TEXT NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP,
    status        TEXT NOT NULL,
    rows_returned INTEGER,
    error_message TEXT,
    metro_coverage REAL
);

CREATE TABLE IF NOT EXISTS models (
    model_id     TEXT NOT NULL,
    model_type   TEXT NOT NULL,
    target       TEXT NOT NULL,
    train_start  DATE,
    train_end    DATE,
    test_start   DATE,
    test_end     DATE,
    metric_name  TEXT NOT NULL,
    metric_value REAL,
    created_at   TIMESTAMP NOT NULL,
    UNIQUE (model_id, metric_name)
);

-- model_id refers to models.model_id. The spec stores one row per metric in
-- ``models`` (so model_id repeats there), which means SQLite cannot enforce a
-- real FOREIGN KEY. write_forecasts() checks the reference in Python instead.
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT NOT NULL,
    target_period   DATE NOT NULL,
    predicted_value REAL,
    actual_value    REAL,
    error           REAL,
    UNIQUE (model_id, target_period)
);
"""

SIGNAL_COLUMNS = [
    "signal_id", "source", "geo_level", "geo_name", "period_start",
    "period_end", "value", "normalized_value", "unit", "quality_score",
    "grain", "fetched_at",
]

# Allowed values for scrapes.status.
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"   # some, but not all, of the data was collected
STATUS_FAILED = "failed"


def now_utc() -> str:
    """Return the current time as an ISO-8601 UTC string.

    Returns:
        For example "2026-11-09T18:00:00+00:00".
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a database connection that commits on success and always closes.

    Use it with ``with``:

        with get_connection() as conn:
            conn.execute(...)

    Args:
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Yields:
        An open ``sqlite3.Connection`` with rows returned as ``sqlite3.Row``.
    """
    path = Path(db_path) if db_path is not None else Path(config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create the four tables if they do not exist yet.

    Args:
        db_path: Database file. Defaults to ``config.DB_PATH``.
    """
    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA)


def _clean(value):
    """Convert pandas/numpy missing values (NaN, NaT) to None for SQLite.

    Args:
        value: Any value.

    Returns:
        ``None`` if the value is missing, otherwise the value unchanged
        (numpy scalars are converted to plain Python numbers).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):  # numpy scalar -> python scalar
        return value.item()
    return value


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def write_signals(rows: pd.DataFrame | list[dict], db_path: str | Path | None = None) -> int:
    """Insert signal rows, updating any row that already exists.

    A row is "the same" if signal_id, geo_level, geo_name, period_start and
    grain all match. ``fetched_at`` is filled in automatically when missing.

    Args:
        rows: A DataFrame or list of dicts with the ``signals`` columns.
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        The number of rows written.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    if "fetched_at" not in df.columns:
        df["fetched_at"] = now_utc()
    missing = [c for c in SIGNAL_COLUMNS if c not in df.columns and c not in
               ("normalized_value", "quality_score", "unit")]
    if missing:
        raise ValueError(f"write_signals: missing columns {missing}")
    for optional in ("normalized_value", "quality_score", "unit"):
        if optional not in df.columns:
            df[optional] = None

    for date_col in ("period_start", "period_end"):
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    records = [
        tuple(_clean(v) for v in row)
        for row in df[SIGNAL_COLUMNS].itertuples(index=False, name=None)
    ]
    sql = """
        INSERT INTO signals (signal_id, source, geo_level, geo_name,
                             period_start, period_end, value, normalized_value,
                             unit, quality_score, grain, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (signal_id, geo_level, geo_name, period_start, grain)
        DO UPDATE SET
            source = excluded.source,
            period_end = excluded.period_end,
            value = excluded.value,
            normalized_value = excluded.normalized_value,
            unit = excluded.unit,
            quality_score = excluded.quality_score,
            fetched_at = excluded.fetched_at
    """
    with get_connection(db_path) as conn:
        conn.executemany(sql, records)
    return len(records)


def read_signals(
    source: str | None = None,
    signal_id: str | None = None,
    geo_level: str | None = None,
    geo_name: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read signal rows, optionally filtered, ordered by period.

    Args:
        source: Keep only this source (e.g. "msrs").
        signal_id: Keep only this series (e.g. "msrs_yoy_TOTAL").
        geo_level: Keep only this level ("state", "metro", "national").
        geo_name: Keep only this place (e.g. "California").
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        A DataFrame with the ``signals`` columns (empty if nothing matches).
    """
    filters = {"source": source, "signal_id": signal_id,
               "geo_level": geo_level, "geo_name": geo_name}
    clauses = [f"{col} = ?" for col, val in filters.items() if val is not None]
    params = [val for val in filters.values() if val is not None]
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (f"SELECT {', '.join(SIGNAL_COLUMNS)} FROM signals {where} "
           "ORDER BY signal_id, geo_name, period_start")
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# scrapes
# ---------------------------------------------------------------------------
def start_scrape(source: str, db_path: str | Path | None = None) -> int:
    """Record the start of an ingestion run with status "running".

    Args:
        source: Source name, e.g. "msrs".
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        The new ``scrape_id``, to pass to ``finish_scrape``.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO scrapes (source, started_at, status) VALUES (?, ?, ?)",
            (source, now_utc(), STATUS_RUNNING),
        )
        return int(cursor.lastrowid)


def finish_scrape(
    scrape_id: int,
    status: str,
    rows_returned: int | None = None,
    error_message: str | None = None,
    metro_coverage: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Close an ingestion run started with ``start_scrape``.

    Args:
        scrape_id: The id returned by ``start_scrape``.
        status: "success", "partial" or "failed".
        rows_returned: How many rows the run collected.
        error_message: What went wrong (or a note for partial runs).
        metro_coverage: Share (0-1) of the state's metros that returned data.
        db_path: Database file. Defaults to ``config.DB_PATH``.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE scrapes SET finished_at = ?, status = ?, rows_returned = ?,
                      error_message = ?, metro_coverage = ?
               WHERE scrape_id = ?""",
            (now_utc(), status, rows_returned, error_message, metro_coverage, scrape_id),
        )


def read_scrapes(
    source: str | None = None,
    status: str | None = None,
    limit: int | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read the scrape log, newest first.

    Args:
        source: Keep only this source.
        status: Keep only this status.
        limit: Return at most this many rows.
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        A DataFrame with the ``scrapes`` columns.
    """
    clauses, params = [], []
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM scrapes {where} ORDER BY scrape_id DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def last_successful_scrape(source: str, db_path: str | Path | None = None) -> datetime | None:
    """Return when a source last finished a successful scrape.

    Runs with status "partial" do not count as successful.

    Args:
        source: Source name, e.g. "indeed".
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        A timezone-aware UTC ``datetime``, or ``None`` if there never was one.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(finished_at) AS last FROM scrapes WHERE source = ? AND status = ?",
            (source, STATUS_SUCCESS),
        ).fetchone()
    if row is None or row["last"] is None:
        return None
    return datetime.fromisoformat(row["last"])


# ---------------------------------------------------------------------------
# models and forecasts (filled in from phase four)
# ---------------------------------------------------------------------------
def write_model_metrics(rows: pd.DataFrame | list[dict], db_path: str | Path | None = None) -> int:
    """Insert (or update) model evaluation metrics.

    Each row is one metric (e.g. MAE) of one model, so a model with MAE and
    RMSE has two rows sharing the same ``model_id``.

    Args:
        rows: DataFrame or list of dicts with the ``models`` columns.
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        The number of rows written.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    if "created_at" not in df.columns:
        df["created_at"] = now_utc()
    columns = ["model_id", "model_type", "target", "train_start", "train_end",
               "test_start", "test_end", "metric_name", "metric_value", "created_at"]
    for col in ("train_start", "train_end", "test_start", "test_end"):
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    records = [tuple(_clean(v) for v in row)
               for row in df[columns].itertuples(index=False, name=None)]
    sql = """
        INSERT INTO models (model_id, model_type, target, train_start, train_end,
                            test_start, test_end, metric_name, metric_value, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (model_id, metric_name) DO UPDATE SET
            model_type = excluded.model_type, target = excluded.target,
            train_start = excluded.train_start, train_end = excluded.train_end,
            test_start = excluded.test_start, test_end = excluded.test_end,
            metric_value = excluded.metric_value, created_at = excluded.created_at
    """
    with get_connection(db_path) as conn:
        conn.executemany(sql, records)
    return len(records)


def read_models(model_id: str | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    """Read model metrics.

    Args:
        model_id: Keep only this model.
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        A DataFrame with the ``models`` columns.
    """
    sql = "SELECT * FROM models"
    params: list = []
    if model_id is not None:
        sql += " WHERE model_id = ?"
        params.append(model_id)
    sql += " ORDER BY created_at, model_id, metric_name"
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def write_forecasts(rows: pd.DataFrame | list[dict], db_path: str | Path | None = None) -> int:
    """Insert forecasts, updating the row for a model and period if it exists.

    Retraining a model therefore updates its old forecasts instead of
    duplicating them (UNIQUE model_id + target_period).

    Args:
        rows: DataFrame or list of dicts with ``model_id``, ``target_period``,
            ``predicted_value`` and optionally ``actual_value`` and ``error``.
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If a ``model_id`` is not in the ``models`` table.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    for col in ("actual_value", "error"):
        if col not in df.columns:
            df[col] = None
    df["target_period"] = pd.to_datetime(df["target_period"]).dt.strftime("%Y-%m-%d")

    with get_connection(db_path) as conn:
        known = {r["model_id"] for r in conn.execute("SELECT DISTINCT model_id FROM models")}
        unknown = set(df["model_id"]) - known
        if unknown:
            raise ValueError(f"write_forecasts: unknown model_id(s) {sorted(unknown)}")
        records = [
            tuple(_clean(v) for v in row)
            for row in df[["model_id", "target_period", "predicted_value",
                           "actual_value", "error"]].itertuples(index=False, name=None)
        ]
        conn.executemany(
            """INSERT INTO forecasts (model_id, target_period, predicted_value,
                                      actual_value, error)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (model_id, target_period) DO UPDATE SET
                   predicted_value = excluded.predicted_value,
                   actual_value = excluded.actual_value,
                   error = excluded.error""",
            records,
        )
    return len(records)


def read_forecasts(model_id: str | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    """Read forecasts, ordered by target period.

    Args:
        model_id: Keep only this model.
        db_path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        A DataFrame with the ``forecasts`` columns.
    """
    sql = "SELECT * FROM forecasts"
    params: list = []
    if model_id is not None:
        sql += " WHERE model_id = ?"
        params.append(model_id)
    sql += " ORDER BY model_id, target_period"
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)
