"""Tests for app/data_loading.py and src/export_demo.py (synthetic data, plus the committed demo database)."""

import sqlite3

import pandas as pd
import pytest

import config
from app import data_loading as dl
from src import db, export_demo
from tests.helpers import write_series


def add_models(abbr="CA", target="retail_sales"):
    """Store a naive and a ridge model with forecasts in the test database."""
    db.init_db()
    base = {"target": target, "test_start": "2026-01-01", "test_end": "2026-02-01"}
    db.write_model_metrics([
        {"model_id": f"{abbr}_{target}_naive_locf", "model_type": "naive", "metric_name": "MAE", "metric_value": 2.0, **base},
        {"model_id": f"{abbr}_{target}_ridge", "model_type": "ridge", "metric_name": "MAE", "metric_value": 1.0, **base},
        {"model_id": "TX_retail_sales_ridge", "model_type": "ridge", "metric_name": "MAE", "metric_value": 9.0,
         "target": "retail_sales", "test_start": "2026-01-01", "test_end": "2026-02-01"},
    ])
    db.write_forecasts([
        {"model_id": f"{abbr}_{target}_ridge", "target_period": "2026-01-01", "predicted_value": 1.0, "actual_value": 2.0, "error": 1.0},
        {"model_id": f"{abbr}_{target}_ridge", "target_period": "2026-03-01", "predicted_value": 1.5},
        {"model_id": "TX_retail_sales_ridge", "target_period": "2026-01-01", "predicted_value": 7.0},
    ])


def test_signal_label():
    assert dl.signal_label("msrs_yoy_TOTAL") == "Retail: Total retail excl. nonstore retailers"
    assert dl.signal_label("msrs_yoy_445") == "Retail: Food and beverage stores (445)"
    assert dl.signal_label("msrs_yoy_999") == "msrs_yoy_999"           # unknown ids are not guessed
    assert dl.signal_label("viirs_avg_radiance") == "Nighttime lights: average radiance"
    assert dl.signal_label("indeed_postings_retail") == "Indeed postings: retail"
    assert dl.signal_label("something_else") == "something_else"


def test_load_signals_keeps_state_and_national_monthly_only():
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 14)
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "Texas", 14)         # other state: excluded
    write_series("fred_unemployment_rate_national", "fred_targets", "percent", "national", "United States", 14)
    write_series("viirs_avg_radiance", "viirs", "nW/cm2/sr", "metro", "Fresno, CA", 14)   # metro: excluded
    weekly = db.read_signals(signal_id="msrs_yoy_TOTAL", geo_name="California").assign(
        grain="weekly", signal_id="indeed_postings_total", source="indeed")
    db.write_signals(weekly)                                                              # weekly grain: excluded
    signals = dl.load_signals("California", config.DB_PATH)
    assert set(signals["signal_id"]) == {"msrs_yoy_TOTAL", "fred_unemployment_rate_national"}
    assert set(signals["geo_scope"]) == {"State", "National"}
    assert pd.api.types.is_datetime64_any_dtype(signals["period_start"])
    assert (signals["label"] == signals["signal_id"].map(dl.signal_label)).all()


def test_load_signals_empty_database():
    db.init_db()
    signals = dl.load_signals("California", config.DB_PATH)
    assert signals.empty and "label" in signals.columns
    assert dl.snapshot_time(signals) is None
    assert dl.wide_series(signals).empty


def test_source_status_marks_missing_sources_as_no_data():
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 14)
    signals = dl.load_signals("California", config.DB_PATH)
    status = dl.source_status(signals).set_index("source")
    assert bool(status.loc[config.SOURCE_MSRS, "has_data"]) and status.loc[config.SOURCE_MSRS, "n_series"] == 1
    for source in (config.SOURCE_INDEED, config.SOURCE_VIIRS, config.SOURCE_FRED_TARGETS):
        assert not status.loc[source, "has_data"] and status.loc[source, "rows"] == 0
        assert status.loc[source, "no_data"]                      # a reason is always given
    assert pd.isna(status.loc[config.SOURCE_VIIRS, "latest_period"])


def test_source_status_reports_last_scrape_and_unknown_sources():
    write_series("custom_thing", "custom", "index", "state", "California", 14)
    scrape_id = db.start_scrape(config.SOURCE_VIIRS)
    db.finish_scrape(scrape_id, db.STATUS_FAILED, error_message="no credentials")
    signals = dl.load_signals("California", config.DB_PATH)
    status = dl.source_status(signals, db.read_scrapes()).set_index("source")
    assert status.loc[config.SOURCE_VIIRS, "last_scrape"].startswith("failed")
    assert not status.loc[config.SOURCE_VIIRS, "has_data"]        # a failed run is not data
    assert bool(status.loc["custom", "has_data"])


def test_wide_series_keeps_missing_months_empty():
    write_series("a", "msrs", "percent_yoy", "state", "California", 15)
    signals = dl.load_signals("California", config.DB_PATH)
    wide = dl.wide_series(signals, "normalized_value")
    assert wide.shape == (15, 1) and wide["a"].isna().sum() == 12          # no z-score for the first 12 months
    assert dl.wide_series(signals, "value")["a"].notna().all()


def test_load_models_and_forecasts_are_scoped_to_state_and_target():
    add_models()
    models = dl.load_models("CA", "retail_sales", config.DB_PATH)
    assert set(models["model_id"]) == {"CA_retail_sales_naive_locf", "CA_retail_sales_ridge"}
    assert dl.load_models("CA", "unemployment", config.DB_PATH).empty
    forecasts = dl.load_forecasts(list(models["model_id"]), config.DB_PATH)
    assert len(forecasts) == 2 and pd.api.types.is_datetime64_any_dtype(forecasts["target_period"])
    assert forecasts["actual_value"].isna().sum() == 1                    # the pending nowcast stays empty
    assert dl.model_short_name("CA_retail_sales_ridge", "CA", "retail_sales") == "ridge"


def test_official_series_follows_feature_targets_with_fallback():
    write_series("fred_unemployment_rate_national", "fred_targets", "percent", "national", "United States", 14)
    signals = dl.load_signals("California", config.DB_PATH)
    series, signal_id = dl.official_series("unemployment", signals)
    assert signal_id == "fred_unemployment_rate_national" and len(series) == 14     # national fallback
    none, none_id = dl.official_series("retail_sales", signals)
    assert none.empty and none_id is None


def test_choose_database_prefers_live_then_demo_then_reports_nothing(tmp_path, monkeypatch):
    demo = tmp_path / "demo.db"
    monkeypatch.setattr(config, "DEMO_DB_PATH", demo)
    assert not dl.choose_database().path.exists() and dl.choose_database().is_demo     # nothing anywhere
    assert "No database" in dl.choose_database().note
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 14, )
    live = dl.choose_database()
    assert live.path == config.DB_PATH and not live.is_demo
    export_demo.export_demo(dest=demo)
    config.DB_PATH.unlink()
    choice = dl.choose_database()
    assert choice.path == demo and choice.is_demo and "demo snapshot" in choice.note


def test_database_has_signals_never_creates_files(tmp_path):
    missing = tmp_path / "nope.db"
    assert not dl.database_has_signals(missing) and not missing.exists()
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()                                     # a file with no tables
    assert not dl.database_has_signals(empty)
    db.init_db(empty)
    assert not dl.database_has_signals(empty)                          # tables but no rows


def test_export_demo_copies_only_public_active_state_data(tmp_path):
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 14)
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "Texas", 14)
    write_series("indeed_postings_total", "indeed", "postings", "state", "California", 14)
    write_series("fred_retail_sales_national", "fred_targets", "millions_usd", "national", "United States", 14)
    write_series("viirs_avg_radiance", "viirs", "nW/cm2/sr", "metro", "Fresno, CA", 14)
    add_models()
    dest = tmp_path / "out" / "demo.db"
    counts = export_demo.export_demo(dest=dest)
    exported = db.read_signals(db_path=dest)
    assert set(exported["source"]) == {"msrs", "fred_targets"}                  # no Indeed or metro rows
    assert set(exported["geo_name"]) == {"California", "United States"}
    assert counts == {"signals": 28, "models": 2, "forecasts": 2}
    assert set(db.read_models(db_path=dest)["model_id"]) == {"CA_retail_sales_naive_locf", "CA_retail_sales_ridge"}
    assert db.read_scrapes(db_path=dest).empty
    # values are copied exactly, z-scores included
    original = db.read_signals(source="msrs", geo_name="California")
    copied = db.read_signals(source="msrs", db_path=dest)
    pd.testing.assert_frame_equal(original, copied)


def test_export_demo_leaves_existing_snapshot_alone_when_nothing_to_export(tmp_path):
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 14)
    dest = tmp_path / "demo.db"
    export_demo.export_demo(dest=dest)
    before = dest.read_bytes()
    empty_source = tmp_path / "empty.db"
    db.init_db(empty_source)
    assert export_demo.export_demo(source_db=empty_source, dest=dest)["signals"] == 0
    assert dest.read_bytes() == before and not dest.with_name("demo.db.tmp").exists()
    assert export_demo.main(["--source", str(empty_source), "--dest", str(dest)]) == 1


# ---------------------------------------------------------------------------
# The committed demo database
# ---------------------------------------------------------------------------
@pytest.fixture
def demo_path():
    """The committed demo snapshot (the autouse fixture only redirects ``DB_PATH``)."""
    assert config.DEMO_DB_PATH.exists(), "run `python -m src.export_demo` and commit data/demo/altsignal_demo.db"
    return config.DEMO_DB_PATH


def test_demo_database_holds_only_allowed_public_data(demo_path):
    signals = db.read_signals(db_path=demo_path)
    assert not signals.empty
    assert set(signals["source"]) <= set(config.DEMO_EXPORT_SOURCES)       # never Indeed
    assert set(signals["grain"]) == {"monthly"} and set(signals["geo_level"]) <= {"state", "national"}
    assert db.read_scrapes(db_path=demo_path).empty                        # no scrape log, no error messages
    assert (config.DEMO_DB_PATH.stat().st_size) < 2_000_000                # small enough to commit


def test_demo_database_is_consistent_with_its_models(demo_path):
    """Stored metrics equal the errors of the stored forecasts on the months every model of the target scored."""
    models, forecasts = db.read_models(db_path=demo_path), db.read_forecasts(db_path=demo_path)
    assert not models.empty and not forecasts.empty
    assert set(forecasts["model_id"]) <= set(models["model_id"])
    wide = models.pivot_table(index="model_id", columns="metric_name", values="metric_value")
    meta = models.groupby("model_id").first()

    def scored(model_id):
        rows = forecasts[(forecasts["model_id"] == model_id) & forecasts["actual_value"].notna() &
                         (forecasts["target_period"] >= meta.loc[model_id, "test_start"]) &
                         (forecasts["target_period"] <= meta.loc[model_id, "test_end"])]
        return rows.set_index("target_period")

    for target, ids in meta.groupby("target").groups.items():
        common = set.intersection(*[set(scored(m).index) for m in ids])
        assert len(common) >= 4, target                                    # every target keeps a usable test set
        for model_id in ids:
            errors = scored(model_id).loc[sorted(common), "error"]
            assert wide.loc[model_id, "MAE"] == pytest.approx(errors.abs().mean())
            assert wide.loc[model_id, "RMSE"] == pytest.approx((errors ** 2).mean() ** 0.5)
