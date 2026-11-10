"""End-to-end tests for ``python -m src.features.build_features``."""

import json
from datetime import date

import pandas as pd

import config
from src import db
from src.features import build_features as bf, split as sp
from tests.helpers import series_rows, write_series

TODAY = date(2026, 11, 10)


def populate(n_msrs=90):
    """Store synthetic MSRS and VIIRS series for California (test data)."""
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", n_msrs, seed=1, quality=0.2)
    write_series("viirs_avg_radiance", "viirs", "nW/cm2/sr", "state", "California", 90, seed=4, quality=0.93)


def test_run_writes_table_meta_and_frozen_split():
    """The three output files exist and agree with each other."""
    populate()
    build, description = bf.build_features("California", today=TODAY)
    table = pd.read_csv(bf.features_path("CA"))
    assert len(table) == len(build.table) and set(table["target_name"]) == {"retail_sales"}
    meta = json.loads(bf.meta_path("CA").read_text())
    assert meta["targets"]["unemployment"] == "no data"
    assert meta["split_boundaries"]["retail_sales"] == description["split_boundaries"]["retail_sales"]
    assert sp.load_frozen("CA")["retail_sales"].test_end == pd.Timestamp("2026-06-01")
    assert meta["release_lag_days_used"]["msrs_yoy_TOTAL.state"] == 115


def test_missing_values_are_empty_cells_not_filled():
    """Early rows have no 12-month z-scores; the CSV keeps them empty."""
    populate()
    bf.build_features("California", today=TODAY)
    table = pd.read_csv(bf.features_path("CA"))
    first = table.iloc[0]
    assert pd.isna(first["msrs_yoy_TOTAL.state__z"])
    assert table["msrs_yoy_TOTAL.state__z"].notna().sum() > 0
    assert "viirs_avg_radiance.state__coverage_pct" in table.columns


def test_rerun_is_idempotent_byte_for_byte():
    """Running twice gives identical files."""
    populate()
    bf.build_features("California", today=TODAY)
    first = (bf.features_path("CA").read_bytes(), bf.meta_path("CA").read_bytes(), sp.split_path("CA").read_bytes())
    bf.build_features("California", today=TODAY)
    second = (bf.features_path("CA").read_bytes(), bf.meta_path("CA").read_bytes(), sp.split_path("CA").read_bytes())
    assert first == second


def test_test_window_stays_frozen_when_new_data_arrives():
    """After new months are ingested, the old test window is kept and new months are 'later'."""
    populate(n_msrs=80)                                   # 2019-01 .. 2025-08
    bf.build_features("California", today=TODAY)
    before = sp.load_frozen("CA")["retail_sales"]
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 90, seed=1, quality=0.2)
    build, _ = bf.build_features("California", today=TODAY)
    after = sp.load_frozen("CA")["retail_sales"]
    assert after == before
    table = build.table
    later = table[table["split"] == "later"]
    assert len(later) > 0 and (later["target_month"] > before.test_end).all()
    assert (table[table["split"] == "test"]["target_month"] <= before.test_end).all()

    bf.build_features("California", today=TODAY, refreeze=True)
    assert sp.load_frozen("CA")["retail_sales"].test_end == pd.Timestamp("2026-06-01")


def test_train_set_from_the_written_table_excludes_the_test_window():
    """Reading the CSV back and making a TrainSet never includes test months."""
    populate()
    build, _ = bf.build_features("California", today=TODAY)
    table = pd.read_csv(bf.features_path("CA"), parse_dates=["target_month", "as_of", "target_available"])
    boundary = sp.load_frozen("CA")["retail_sales"]
    train = sp.make_train_set(table, "retail_sales", boundary)
    assert train.frame["target_month"].max() < boundary.test_start
    assert (table[table["split"] == "test"]["target_month"] >= boundary.test_start).all()


def test_main_exit_codes(monkeypatch, capsys):
    """Exit 1 with nothing to build, 0 once there is data."""
    assert bf.main([]) == 1
    populate()
    assert bf.main([]) == 0
    assert "test window 2026-01 to 2026-06" in capsys.readouterr().out


def test_other_states_data_is_ignored():
    """Texas rows in the same database do not enter California's features."""
    populate()
    write_series("msrs_yoy_441", "msrs", "percent_yoy", "state", "Texas", 90, seed=3, quality=0.2)
    build, _ = bf.build_features("California", today=TODAY)
    assert not any("441" in c for c in build.table.columns)


def test_switching_state_uses_its_own_files(monkeypatch):
    """Changing config.STATE alone builds the other state's table and split."""
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "Texas", 90, seed=1, quality=0.2)
    monkeypatch.setattr(config, "STATE", "Texas")
    assert bf.main([]) == 0
    assert bf.features_path("TX").exists() and not bf.features_path("CA").exists()
