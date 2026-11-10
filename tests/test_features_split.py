"""Tests for src/features/split.py: the frozen boundary and the train-only guard."""

import numpy as np
import pandas as pd
import pytest

import config
from src.features import split as sp


@pytest.fixture(autouse=True)
def full_history_training(monkeypatch):
    """These tests exercise the split mechanics on short synthetic tables, so training may start anywhere."""
    monkeypatch.setattr(config, "TRAIN_START", None)


def months(n, start="2019-01-01"):
    """Return ``n`` consecutive month starts."""
    return pd.date_range(start, periods=n, freq="MS")


def make_table(n=40, target="retail_sales", series="msrs_yoy_TOTAL"):
    """Build a small labelled feature table (test data) and its boundary."""
    index = months(n)
    boundary = sp.define_boundary(target, series, index)
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"target_name": target, "target_series": series, "target_month": index,
                          "as_of": index, "target_available": index, "pandemic_regime": 0, "target_value": rng.normal(size=n),
                          "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    table["split"] = [sp.assign_split(m, True, boundary) for m in index]
    return table, boundary


def test_boundary_is_the_last_six_observed_months():
    """With 40 months, the last 6 are test and the 34 before are train."""
    table, boundary = make_table(40)
    assert boundary.test_end == months(40)[-1]
    assert boundary.test_start == months(40)[-6]
    assert (table["split"] == "test").sum() == 6 and (table["split"] == "train").sum() == 34


def test_split_labels():
    """Train, test, later and unlabeled are assigned from the frozen boundary."""
    _, b = make_table(40)
    assert sp.assign_split(b.test_start - pd.DateOffset(months=1), True, b) == "train"
    assert sp.assign_split(b.test_start, True, b) == "test"
    assert sp.assign_split(b.test_end, True, b) == "test"
    assert sp.assign_split(b.test_end + pd.DateOffset(months=1), True, b) == "later"
    assert sp.assign_split(b.test_start, False, b) == "unlabeled"


def test_too_little_history_refuses_to_define_a_split():
    """Fewer than MIN_TRAIN_MONTHS labelled months before the test window is an error."""
    with pytest.raises(sp.SplitError, match="need at least"):
        sp.define_boundary("t", "s", months(config.MIN_TRAIN_MONTHS + config.TEST_MONTHS - 1))
    sp.define_boundary("t", "s", months(config.MIN_TRAIN_MONTHS + config.TEST_MONTHS))   # exactly enough


def test_boundary_is_frozen_and_does_not_move_when_new_months_arrive():
    """Once saved, new data does not shift the test window; refreeze does."""
    frozen = {"retail_sales": sp.define_boundary("retail_sales", "s", months(40))}
    sp.save_frozen("CA", frozen)
    reloaded = sp.load_frozen("CA")
    assert reloaded == frozen
    same = sp.resolve_boundary("retail_sales", "s", months(45), reloaded)
    assert same.test_end == months(40)[-1]
    moved = sp.resolve_boundary("retail_sales", "s", months(45), reloaded, refreeze=True)
    assert moved.test_end == months(45)[-1]


def test_frozen_boundary_for_another_series_is_refused():
    """Switching target series (state -> national unemployment) needs an explicit refreeze."""
    frozen = {"unemployment": sp.define_boundary("unemployment", "state_ur", months(40))}
    with pytest.raises(sp.SplitError, match="refreeze-split"):
        sp.resolve_boundary("unemployment", "national_ur", months(40), frozen)


def test_load_frozen_without_file_is_empty():
    """No file means nothing is frozen yet."""
    assert sp.load_frozen("ZZ") == {}


def test_train_set_accepts_only_clean_training_rows():
    """make_train_set returns the train rows; features exclude the meta columns."""
    table, boundary = make_table()
    train = sp.make_train_set(table, "retail_sales", boundary)
    assert len(train.frame) == 34
    assert list(train.features.columns) == ["f1", "f2"]
    assert train.frame["target_month"].max() < boundary.test_start


def test_train_set_rejects_tables_containing_test_period_months():
    """Constructing a TrainSet from test rows (or a mix) raises."""
    table, boundary = make_table()
    with pytest.raises(sp.SplitError, match="test-period"):
        sp.TrainSet(table, boundary)
    with pytest.raises(sp.SplitError, match="test-period"):
        sp.TrainSet(table[table["split"] == "test"], boundary)


def test_train_set_rejects_mislabelled_unlabeled_or_foreign_rows():
    """Rows must be labelled train, have a target, and belong to the target."""
    table, boundary = make_table()
    train_rows = table[table["split"] == "train"].copy()
    bad_label = train_rows.copy()
    bad_label.iloc[0, bad_label.columns.get_loc("split")] = "unlabeled"
    with pytest.raises(sp.SplitError, match="labelled 'train'"):
        sp.TrainSet(bad_label, boundary)
    no_target = train_rows.copy()
    no_target.iloc[0, no_target.columns.get_loc("target_value")] = np.nan
    with pytest.raises(sp.SplitError, match="observed target"):
        sp.TrainSet(no_target, boundary)
    other = train_rows.copy()
    other["target_name"] = "unemployment"
    with pytest.raises(sp.SplitError, match="different target"):
        sp.TrainSet(other, boundary)


def test_fitted_steps_reject_anything_but_a_train_set():
    """Passing a plain table (which could hold test rows) to a fitted step is a TypeError."""
    table, boundary = make_table()
    with pytest.raises(TypeError):
        sp.fit_standardizer(table)
    with pytest.raises(TypeError):
        sp.fit_standardizer(sp.open_test_set(table, "retail_sales"))
    with pytest.raises(TypeError):
        sp.select_columns(table)


def test_standardizer_is_learned_from_training_rows_only():
    """Changing the test rows cannot change the fitted mean/std."""
    table, boundary = make_table()
    first = sp.fit_standardizer(sp.make_train_set(table, "retail_sales", boundary))
    altered = table.copy()
    altered.loc[altered["split"] == "test", ["f1", "f2"]] = 1e6
    second = sp.fit_standardizer(sp.make_train_set(altered, "retail_sales", boundary))
    pd.testing.assert_series_equal(first.mean, second.mean)
    pd.testing.assert_series_equal(first.std, second.std)
    train_f1 = table.loc[table["split"] == "train", "f1"]
    assert first.mean["f1"] == pytest.approx(train_f1.mean())


def test_standardizer_transform_keeps_gaps():
    """Applying the training scaler to other rows leaves NaN as NaN."""
    table, boundary = make_table()
    scaler = sp.fit_standardizer(sp.make_train_set(table, "retail_sales", boundary))
    test = sp.open_test_set(table, "retail_sales")
    test.iloc[0, test.columns.get_loc("f1")] = np.nan
    out = scaler.transform(test)
    assert out["f1"].isna().iloc[0] and out["f2"].notna().all()


def test_select_columns_uses_training_rows_only():
    """A column that is mostly missing in training is dropped; test-period values are irrelevant."""
    table, boundary = make_table()
    train_mask = table["split"] == "train"
    table.loc[train_mask & (np.arange(len(table)) % 4 != 0), "f2"] = np.nan     # 75% missing in train
    table.loc[table["split"] == "test", "f2"] = 1.0                              # complete in test
    train = sp.make_train_set(table, "retail_sales", boundary)
    assert sp.select_columns(train, max_missing_share=0.5) == ["f1"]
    assert sp.select_columns(train, max_missing_share=0.9) == ["f1", "f2"]


def test_open_test_set_returns_only_test_rows():
    """The held-out rows are available for final evaluation, as a plain table."""
    table, _ = make_table()
    test = sp.open_test_set(table, "retail_sales")
    assert len(test) == 6 and set(test["split"]) == {"test"}
    assert not isinstance(test, sp.TrainSet)


# ---- configurable training-window start (the pandemic regime) -----------------

def test_train_start_labels_early_months_pre_window_and_leaves_the_test_window_alone(monkeypatch):
    """Months before TRAIN_START are kept but not train; test months are unaffected."""
    index = months(60)                                       # 2019-01 .. 2023-12
    boundary = sp.define_boundary("t", "s", index, train_start="2021-01-01")
    assert boundary.test_start == pd.Timestamp("2023-07-01")  # same as without a window
    labels = {m: sp.assign_split(m, True, boundary, "2021-01-01") for m in index}
    assert labels[pd.Timestamp("2020-12-01")] == "pre_window"
    assert labels[pd.Timestamp("2021-01-01")] == "train"
    assert sum(v == "pre_window" for v in labels.values()) == 24
    assert sum(v == "test" for v in labels.values()) == 6
    unrestricted = sp.define_boundary("t", "s", index, train_start=None)
    assert unrestricted == boundary


def test_train_start_cannot_move_a_frozen_test_window():
    """A frozen boundary is reused as is, whatever TRAIN_START says."""
    frozen = {"t": sp.define_boundary("t", "s", months(60), train_start=None)}
    again = sp.resolve_boundary("t", "s", months(60), frozen, train_start="2022-06-01")
    assert again == frozen["t"]


def test_train_start_counts_toward_the_minimum_training_months():
    """Too few months inside the window refuses the split with a clear message."""
    with pytest.raises(sp.SplitError, match="since 2022-01"):
        sp.define_boundary("t", "s", months(60), train_start="2022-01-01")     # only 18 in-window months
    sp.define_boundary("t", "s", months(60), train_start="2021-06-01")        # 24 in-window months


def test_train_set_rejects_months_before_train_start():
    """Even rows mislabelled 'train' cannot enter a TrainSet before the window start."""
    table, boundary = make_table(60)
    train_rows = table[table["split"] == "train"]
    with pytest.raises(sp.SplitError, match="training start 2020-01"):
        sp.TrainSet(train_rows, boundary, train_start="2020-01-01")
    ok = sp.TrainSet(train_rows[train_rows["target_month"] >= "2020-01-01"], boundary, train_start="2020-01-01")
    assert ok.frame["target_month"].min() == pd.Timestamp("2020-01-01")


def test_make_train_set_applies_the_configured_start(monkeypatch):
    """make_train_set reads config.TRAIN_START by default and drops pre-window rows."""
    monkeypatch.setattr(config, "TRAIN_START", "2020-06-01")
    table, boundary = make_table(60)
    train = sp.make_train_set(table, "retail_sales", boundary)
    assert train.frame["target_month"].min() == pd.Timestamp("2020-06-01")
    assert train.frame["target_month"].max() < boundary.test_start


def test_pre_window_rows_are_not_available_to_fitted_steps(monkeypatch):
    """Changing pre-window rows cannot change the fitted scaler."""
    monkeypatch.setattr(config, "TRAIN_START", "2021-01-01")
    index = months(60)
    boundary = sp.define_boundary("retail_sales", "msrs_yoy_TOTAL", index)
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"target_name": "retail_sales", "target_series": "msrs_yoy_TOTAL",
                          "target_month": index, "as_of": index, "target_available": index,
                          "pandemic_regime": 0, "target_value": rng.normal(size=60), "f1": rng.normal(size=60)})
    table["split"] = [sp.assign_split(m, True, boundary, config.TRAIN_START) for m in index]
    first = sp.fit_standardizer(sp.make_train_set(table, "retail_sales", boundary))
    wild = table.copy()
    wild.loc[wild["split"] == "pre_window", "f1"] = 1e6
    second = sp.fit_standardizer(sp.make_train_set(wild, "retail_sales", boundary))
    pd.testing.assert_series_equal(first.mean, second.mean)
