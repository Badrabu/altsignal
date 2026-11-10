"""Tests for src/models/fit.py and run_models.py (synthetic feature tables, no network)."""

import json

import numpy as np
import pandas as pd
import pytest

import config
from src import db
from src.features import build_features as bf
from src.features import split as sp
from src.models import fit, run_models as rm


def make_table(n=60, signal=True, seed=0, lag_days=40, unlabeled=2):
    """Build a synthetic feature table for one target ("retail_sales") with 4 features.

    The target depends on feature ``f1`` when ``signal`` is true, is white noise when false,
    and is an unrelated random walk when ``signal`` is "walk".
    The first 3 rows have missing features (like the z-score warm-up in phase three).
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2019-01-01", periods=n + unlabeled, freq="MS")
    feats = pd.DataFrame(rng.normal(size=(len(months), 4)), columns=["f1", "f2", "f3", "f4"])
    noise = rng.normal(scale=0.3, size=len(months))
    if signal == "walk":                  # persistent series unrelated to the features: naive is hard to beat
        target = pd.Series(np.cumsum(noise))
    else:
        target = (2.0 * feats["f1"] + noise) if signal else pd.Series(noise)
    target = target.to_numpy().copy()
    target[n:] = np.nan
    feats.iloc[:3] = np.nan
    month_end = months + pd.offsets.MonthEnd(0)
    table = pd.DataFrame({"target_name": "retail_sales", "target_series": "msrs_yoy_TOTAL",
                          "target_month": months, "as_of": month_end + pd.Timedelta(days=1),
                          "target_available": month_end + pd.Timedelta(days=lag_days),
                          "target_value": target})
    boundary = sp.define_boundary("retail_sales", "msrs_yoy_TOTAL", months[:n])
    table["split"] = [sp.assign_split(m, pd.notna(v), boundary) for m, v in zip(months, target)]
    return pd.concat([table, feats], axis=1), boundary


def train_for(table, boundary):
    """Prepare complete training rows exactly as run_models does."""
    train_all = sp.make_train_set(table, "retail_sales", boundary)
    return fit.prepare_training_rows(train_all)


def test_prepare_drops_incomplete_rows_and_never_imputes():
    """Rows with a missing feature are dropped; the rest are untouched."""
    table, b = make_table()
    train, dropped = train_for(table, b)
    assert len(train.frame) == (table["split"] == "train").sum() - 3
    assert not train.features.isna().any().any()
    assert dropped == 0


def test_too_few_complete_rows_raises():
    """A short training window gives a clear error rather than a poor fit."""
    table, b = make_table(n=45)
    with pytest.raises(fit.ModelError, match="complete training rows"):
        train_for(table, b)


def test_fit_model_only_accepts_a_trainset():
    """A plain DataFrame (for example the test rows) is rejected."""
    table, b = make_table()
    with pytest.raises(TypeError):
        fit.fit_model("ridge", sp.open_test_set(table, "retail_sales"))


def test_trainset_cannot_contain_test_rows():
    """The guard from phase three still protects the models."""
    table, b = make_table()
    rows = table[table["split"].isin(["train", "test"])]
    with pytest.raises(sp.SplitError):
        sp.TrainSet(rows, b)


def test_ridge_learns_a_real_signal_and_beats_noise():
    """With a true linear signal, ridge predicts the test months far better than the training mean."""
    table, b = make_table(signal=True)
    train, dropped = train_for(table, b)
    model = fit.fit_model("ridge", train, dropped)
    test = sp.open_test_set(table, "retail_sales")
    pred = model.predict(test)
    err = fit.rmse(test["target_value"].to_numpy(), pred.to_numpy())
    baseline = fit.rmse(test["target_value"].to_numpy(), np.full(len(test), train.target.mean()))
    assert err < 0.5 * baseline
    assert model.cv_rmse < 1.0 and model.params["alpha"] in config.RIDGE_ALPHAS


def test_models_do_not_depend_on_test_targets():
    """Changing the test-period targets must not change the fitted model or its predictions."""
    table, b = make_table()
    other = table.copy()
    other.loc[other["split"] == "test", "target_value"] += 1000.0
    a = fit.fit_model("ridge", train_for(table, b)[0])
    c = fit.fit_model("ridge", train_for(other, b)[0])
    assert a.params == c.params and a.cv_rmse == c.cv_rmse
    pd.testing.assert_series_equal(a.predict(table), c.predict(other))


def test_cross_validation_is_time_ordered_with_a_gap():
    """Every validation month is later than all training months of its fold, by at least the gap."""
    table, b = make_table(lag_days=100)
    train, _ = train_for(table, b)
    gap = fit.label_gap(train)
    assert gap == 3                       # labels public 100 days after month end: 3 months not yet public
    from sklearn.model_selection import TimeSeriesSplit
    for fit_rows, val_rows in TimeSeriesSplit(config.MODEL_CV_FOLDS, test_size=config.MODEL_CV_VALIDATION_MONTHS,
                                              gap=gap).split(np.arange(len(train.frame))):
        assert fit_rows.max() + gap < val_rows.min()


def test_prediction_is_nan_when_a_feature_is_missing():
    """A row with a gap gets no prediction instead of a filled-in one."""
    table, b = make_table()
    model = fit.fit_model("ridge", train_for(table, b)[0])
    pred = model.predict(table)
    assert pred.iloc[:3].isna().all() and pred.iloc[3:].notna().all()


def test_tree_is_shallow():
    """The tree's depth stays within the configured grid."""
    table, b = make_table()
    model = fit.fit_model("tree", train_for(table, b)[0])
    assert model.estimator.get_depth() <= max(config.TREE_GRID["max_depth"])


def test_constant_columns_are_not_used():
    """A feature with no variation in training is dropped, not turned into NaN."""
    table, b = make_table()
    table["f4"] = 1.0
    train, dropped = train_for(table, b)
    assert "f4" not in train.features.columns and dropped == 1
    assert fit.fit_model("ridge", train, dropped).predict(table).iloc[3:].notna().all()


def run_with(table, b, state="CA"):
    """Run the per-target pipeline on a synthetic table."""
    return rm.run_target(table, "retail_sales", b, state)


def test_run_target_scores_every_model_on_the_same_months():
    """All models share one set of scored test months and report MAE and RMSE."""
    table, b = make_table(signal=True)
    report = run_with(table, b)
    assert report["test_months_scored"] == 6
    assert set(report["models"]) >= {"naive_locf", "naive_seasonal", "ridge", "tree"}
    for entry in report["models"].values():
        assert entry["MAE"] >= 0 and entry["RMSE"] >= entry["MAE"] - 1e-9
    assert report["verdict"]["any_model_meaningfully_beats_naive"] is True     # a real signal is found


def test_gradient_boosting_runs_only_if_ridge_beats_naive_in_cv():
    """With a real signal ridge works and GBM runs; with an unrelated random walk it does not."""
    with_signal = run_with(*make_table(signal=True))
    assert with_signal["ridge_gate"]["ridge_works"] and "gbm" in with_signal["models"]
    noise = run_with(*make_table(signal="walk", seed=5))
    assert not noise["ridge_gate"]["ridge_works"]
    assert "gbm" not in noise["models"] and any("gradient boosting not run" in n for n in noise["notes"])


def test_unrelated_persistent_target_is_not_meaningfully_beaten():
    """When the features carry no information about a persistent target, naive wins."""
    report = run_with(*make_table(signal="walk", seed=5))
    assert report["verdict"]["any_model_meaningfully_beats_naive"] is False


def test_store_results_is_idempotent_and_writes_only_out_of_sample_rows():
    """Storing twice updates rows (no duplicates); only test/unlabeled months become forecasts."""
    table, b = make_table(unlabeled=2)
    report = run_with(table, b)
    first = rm.store_results(report, "CA")
    metrics_1, forecasts_1 = db.read_models(), db.read_forecasts()
    rm.store_results(report, "CA")
    metrics_2, forecasts_2 = db.read_models(), db.read_forecasts()
    assert len(metrics_1) == len(metrics_2) and len(forecasts_1) == len(forecasts_2) == first[1]
    ridge = forecasts_2[forecasts_2["model_id"] == "CA_retail_sales_ridge"]
    assert len(ridge) == 8                                    # 6 test months + 2 unlabeled
    assert ridge["actual_value"].notna().sum() == 6
    scored = ridge.dropna(subset=["actual_value"])
    np.testing.assert_allclose(scored["error"], scored["actual_value"] - scored["predicted_value"])
    assert set(metrics_2.loc[metrics_2["model_id"] == "CA_retail_sales_ridge", "metric_name"]) == {"MAE", "RMSE", "CV_RMSE"}
    assert set(metrics_2["model_id"]) >= {"CA_retail_sales_naive_locf", "CA_retail_sales_naive_seasonal"}


def test_retraining_updates_forecasts_instead_of_duplicating():
    """A second run with different data replaces the old forecast values."""
    table, b = make_table(seed=1)
    rm.store_results(run_with(table, b), "CA")
    before = db.read_forecasts("CA_retail_sales_ridge")
    table2, b2 = make_table(seed=2)
    rm.store_results(run_with(table2, b2), "CA")
    after = db.read_forecasts("CA_retail_sales_ridge")
    assert len(after) == len(before)
    assert not np.allclose(before["predicted_value"], after["predicted_value"])


def write_features(table):
    """Save a synthetic feature table and frozen split where phase three would put them."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(bf.features_path("CA"), index=False, date_format="%Y-%m-%d")
    months = pd.to_datetime(table.loc[table["target_value"].notna(), "target_month"])
    sp.save_frozen("CA", {"retail_sales": sp.define_boundary("retail_sales", "msrs_yoy_TOTAL", months)})


def test_main_is_idempotent_and_reports_a_missing_target(capsys):
    """The command runs end to end, rewrites an identical report, and says unemployment has no data."""
    table, _ = make_table()
    write_features(table)
    assert rm.main([]) == 0
    first = (config.PROCESSED_DIR / "models_CA.json").read_text()
    n_metrics, n_forecasts = len(db.read_models()), len(db.read_forecasts())
    assert rm.main([]) == 0
    assert (config.PROCESSED_DIR / "models_CA.json").read_text() == first
    assert (len(db.read_models()), len(db.read_forecasts())) == (n_metrics, n_forecasts)
    out = capsys.readouterr().out
    assert "target unemployment: no data in the feature table" in out and "ridge" in out
    assert "retail_sales" in json.loads(first)


def test_main_without_a_feature_table_fails_cleanly(capsys):
    """A missing feature table gives a clear message and exit code 1."""
    assert rm.main([]) == 1
    assert "build_features" in capsys.readouterr().out
