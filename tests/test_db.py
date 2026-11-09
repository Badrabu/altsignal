"""Tests for src/db.py."""

import sqlite3

import pandas as pd
import pytest

import config
from src import db


def make_signal(**overrides):
    """Build one valid signal row for tests."""
    row = {
        "signal_id": "test_signal", "source": "msrs", "geo_level": "state",
        "geo_name": "California", "period_start": "2026-01-01",
        "period_end": "2026-01-31", "value": 1.5, "normalized_value": 0.3,
        "unit": "percent", "quality_score": 0.5, "grain": "monthly",
    }
    row.update(overrides)
    return row


def test_init_db_creates_the_four_tables():
    """init_db creates exactly the four tables from the spec."""
    db.init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    assert tables == {"signals", "scrapes", "models", "forecasts"}


def test_init_db_is_safe_to_repeat():
    """Calling init_db twice does not fail or wipe data."""
    db.init_db()
    db.write_signals([make_signal()])
    db.init_db()
    assert len(db.read_signals()) == 1


def test_write_signals_updates_instead_of_duplicating():
    """Writing the same series/period twice keeps one row with the new value."""
    db.init_db()
    db.write_signals([make_signal(value=1.0)])
    db.write_signals([make_signal(value=2.0)])
    rows = db.read_signals()
    assert len(rows) == 1
    assert rows.loc[0, "value"] == 2.0


def test_write_signals_stores_missing_values_as_null():
    """NaN becomes NULL, and fetched_at is filled in automatically."""
    db.init_db()
    db.write_signals(pd.DataFrame([make_signal(normalized_value=float("nan"))]))
    row = db.read_signals().iloc[0]
    assert pd.isna(row["normalized_value"])
    assert row["fetched_at"]


def test_write_signals_requires_columns():
    """A row without required columns is rejected with a clear message."""
    db.init_db()
    with pytest.raises(ValueError, match="missing columns"):
        db.write_signals([{"signal_id": "x"}])


def test_read_signals_filters():
    """read_signals filters by source, series and geography."""
    db.init_db()
    db.write_signals([
        make_signal(),
        make_signal(source="viirs", signal_id="other", geo_name="Texas"),
    ])
    assert len(db.read_signals(source="viirs")) == 1
    assert len(db.read_signals(geo_name="California")) == 1
    assert len(db.read_signals(signal_id="nothing")) == 0


def test_scrape_lifecycle_and_last_success():
    """start/finish record a run; only 'success' counts as a successful scrape."""
    db.init_db()
    assert db.last_successful_scrape("msrs") is None

    failed = db.start_scrape("msrs")
    db.finish_scrape(failed, db.STATUS_FAILED, rows_returned=0, error_message="boom")
    assert db.last_successful_scrape("msrs") is None

    partial = db.start_scrape("msrs")
    db.finish_scrape(partial, db.STATUS_PARTIAL, rows_returned=3, metro_coverage=0.5)
    assert db.last_successful_scrape("msrs") is None

    ok = db.start_scrape("msrs")
    db.finish_scrape(ok, db.STATUS_SUCCESS, rows_returned=10, metro_coverage=1.0)
    assert db.last_successful_scrape("msrs") is not None

    log = db.read_scrapes(source="msrs")
    assert list(log["status"]) == ["success", "partial", "failed"]  # newest first
    assert log.loc[2, "error_message"] == "boom"
    assert len(db.read_scrapes(status="failed")) == 1


def test_models_and_forecasts_upsert():
    """Retraining updates forecasts (UNIQUE model_id + target_period)."""
    db.init_db()
    db.write_model_metrics([
        {"model_id": "ridge_v1", "model_type": "ridge", "target": "unrate",
         "metric_name": "MAE", "metric_value": 0.2},
        {"model_id": "ridge_v1", "model_type": "ridge", "target": "unrate",
         "metric_name": "RMSE", "metric_value": 0.3},
    ])
    assert len(db.read_models("ridge_v1")) == 2

    db.write_forecasts([{"model_id": "ridge_v1", "target_period": "2026-06-01", "predicted_value": 4.0}])
    db.write_forecasts([{"model_id": "ridge_v1", "target_period": "2026-06-01",
                         "predicted_value": 4.2, "actual_value": 4.1, "error": 0.1}])
    forecasts = db.read_forecasts("ridge_v1")
    assert len(forecasts) == 1
    assert forecasts.loc[0, "predicted_value"] == 4.2


def test_forecast_for_unknown_model_is_rejected():
    """Forecasts must reference an existing model."""
    db.init_db()
    with pytest.raises(ValueError, match="unknown model_id"):
        db.write_forecasts([{"model_id": "ghost", "target_period": "2026-06-01", "predicted_value": 1.0}])
