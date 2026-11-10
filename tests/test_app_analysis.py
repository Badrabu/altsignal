"""Tests for app/analysis.py: trend rule, tab 1 summary table, scoreboard and caveat notes."""

import numpy as np
import pandas as pd

import config
from app import analysis as an
from app import data_loading as dl
from tests.helpers import series_rows


def monthly(values, start="2025-01-01"):
    """A monthly series from a list (``np.nan`` = missing month)."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def test_trend_rising_falling_flat():
    base = [0, 1, 0, 1, 0, 1, 0, 1]                      # noisy history gives the series a spread
    assert an.classify_trend(monthly(base + [3, 4, 5]))["direction"] == an.RISING
    assert an.classify_trend(monthly(base + [-3, -4, -5]))["direction"] == an.FALLING
    assert an.classify_trend(monthly(base + [0.5] * 6))["direction"] == an.FLAT


def test_trend_numbers():
    result = an.classify_trend(monthly([1, 1, 1, 2, 2, 2]), window=3, min_ratio=0.25)
    assert result["prior_mean"] == 1 and result["recent_mean"] == 2 and result["delta"] == 1
    assert result["ratio"] == 1 / pd.Series([1, 1, 1, 2, 2, 2]).std(ddof=1)
    assert result["direction"] == an.RISING


def test_trend_needs_complete_recent_months_and_enough_history():
    assert an.classify_trend(monthly([1, 2, 3, 4, 5]))["direction"] == an.UNKNOWN          # 5 < 6 months
    gap = monthly([1, 2, 3, np.nan, 5, 6, 7])
    assert an.classify_trend(gap)["direction"] == an.UNKNOWN                               # gap in last 6 months
    assert an.classify_trend(monthly([np.nan, np.nan]))["direction"] == an.UNKNOWN
    assert an.classify_trend(pd.Series(dtype=float))["direction"] == an.UNKNOWN


def test_trend_constant_series_is_flat():
    assert an.classify_trend(monthly([2.0] * 8))["direction"] == an.FLAT


def make_signals(**kwargs):
    """Synthetic state-level rows with the columns the loaders add."""
    rows = series_rows("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 24, seed=1)
    rows["label"] = dl.signal_label("msrs_yoy_TOTAL")
    rows["geo_scope"] = "State"
    rows["period_start"] = pd.to_datetime(rows["period_start"])
    return rows


def test_summarize_signals_raw_and_normalized():
    signals = make_signals()
    raw = an.summarize_signals(signals, "value")
    norm = an.summarize_signals(signals, "normalized_value")
    assert len(raw) == len(norm) == 1
    assert raw.loc[0, "latest"] == signals["value"].iloc[-1]
    assert raw.loc[0, "change"] == signals["value"].iloc[-1] - signals["value"].iloc[-2]
    assert raw.loc[0, "unit"] == "percent_yoy" and norm.loc[0, "unit"] == "z-score"
    assert raw.loc[0, "n_points"] == 24 and norm.loc[0, "n_points"] == 12     # z-score starts after 12 months
    assert norm.loc[0, "latest"] == signals["normalized_value"].iloc[-1]
    assert raw.loc[0, "latest_period"] == pd.Timestamp("2020-12-01")


def test_summarize_signals_series_without_values_has_no_reading():
    signals = make_signals().iloc[:8].copy()            # fewer than 12 months: no z-scores at all
    table = an.summarize_signals(signals, "normalized_value")
    assert pd.isna(table.loc[0, "latest"]) and pd.isna(table.loc[0, "latest_period"])
    assert table.loc[0, "trend"] == an.UNKNOWN and table.loc[0, "n_points"] == 0


def test_summarize_signals_empty_input():
    empty = pd.DataFrame(columns=["signal_id", "geo_level", "source", "label", "geo_scope", "unit",
                                  "period_start", "value", "normalized_value", "quality_score"])
    assert an.summarize_signals(empty, "value").empty


def metric_rows(model_id, kind, mae, rmse, cv=None):
    """``models`` table rows for one model."""
    common = {"model_id": model_id, "model_type": kind, "target": "retail_sales",
              "train_start": None if kind == "naive" else "2022-01-01", "train_end": None if kind == "naive" else "2025-12-01",
              "test_start": "2026-01-01", "test_end": "2026-06-01"}
    rows = [{**common, "metric_name": "MAE", "metric_value": mae}, {**common, "metric_name": "RMSE", "metric_value": rmse}]
    if cv is not None:
        rows.append({**common, "metric_name": "CV_RMSE", "metric_value": cv})
    return rows


def forecast_rows(model_id, predicted, actual, start="2026-01-01"):
    """``forecasts`` rows for consecutive months."""
    months = pd.date_range(start, periods=len(predicted), freq="MS")
    return [{"model_id": model_id, "target_period": m, "predicted_value": p,
             "actual_value": a, "error": None if a is None else a - p}
            for m, p, a in zip(months, predicted, actual)]


def test_scoreboard_matches_phase_four_rule():
    models = pd.DataFrame(metric_rows("X_naive_locf", "naive", 2.0, 3.0) + metric_rows("X_naive_seasonal", "naive", 3.0, 4.0)
                          + metric_rows("X_good", "tree", 1.0, 1.5, cv=4.0)           # 50% / 50% better
                          + metric_rows("X_mixed", "ridge", 1.95, 2.0, cv=3.0))       # RMSE 33% better, MAE 2.5%
    actual = [1.0, 2.0, 3.0]
    forecasts = pd.DataFrame(forecast_rows("X_good", [1, 2, 3], actual) + forecast_rows("X_naive_locf", [0, 0, 0], actual)
                             + forecast_rows("X_good", [3.5], [None], start="2026-08-01"))
    forecasts["target_period"] = pd.to_datetime(forecasts["target_period"])
    board = an.build_scoreboard(models, forecasts, margin=0.10).set_index("model_id")
    assert list(board.index[:2]) == ["X_naive_locf", "X_naive_seasonal"]              # naive first
    assert an.best_naive_id(board.reset_index()) == "X_naive_locf"
    assert board.loc["X_good", "rmse_vs_naive"] == 1 - 1.5 / 3.0
    assert bool(board.loc["X_good", "clears_margin"]) is True
    assert bool(board.loc["X_mixed", "clears_margin"]) is False
    assert pd.isna(board.loc["X_naive_locf", "clears_margin"])
    assert board.loc["X_good", "test_months"] == 3        # the pending August nowcast is not scored
    assert board.loc["X_good", "CV_RMSE"] == 4.0 and pd.isna(board.loc["X_naive_locf", "CV_RMSE"])


def test_scoreboard_margin_uses_config_by_default():
    models = pd.DataFrame(metric_rows("X_naive_locf", "naive", 2.0, 2.0) + metric_rows("X_m", "ridge", 1.85, 1.85))
    assert bool(an.build_scoreboard(models, pd.DataFrame()).set_index("model_id").loc["X_m", "clears_margin"]) is False
    assert config.MEANINGFUL_MARGIN == 0.10


def test_scoreboard_without_metrics_or_benchmarks():
    assert an.build_scoreboard(pd.DataFrame(), pd.DataFrame()).empty
    only_model = pd.DataFrame(metric_rows("X_m", "ridge", 1.0, 1.0))
    board = an.build_scoreboard(only_model, pd.DataFrame())
    assert pd.isna(board.loc[0, "rmse_vs_naive"]) and an.best_naive_id(board) is None
    assert an.honesty_notes(board, pd.DataFrame(), {}) == [] or "test window" not in an.honesty_notes(board, pd.DataFrame(), {})[0]


def test_honesty_notes_flag_constant_biased_and_unpredicted_models():
    models = pd.DataFrame(metric_rows("X_naive_locf", "naive", 2.0, 3.0)
                          + metric_rows("X_tree", "tree", 1.0, 1.2, cv=4.0))
    actual = [2.0, 3.0, 4.0, 5.0]
    forecasts = pd.DataFrame(forecast_rows("X_tree", [1.0, 1.0, 1.0, 1.0], actual))   # constant, always under-predicts
    board = an.build_scoreboard(models, forecasts)
    notes = " ".join(an.honesty_notes(board, forecasts, {"X_tree": "tree", "X_naive_locf": "naive_locf"}))
    assert "only 4 scored months" in notes
    assert "tree: RMSE +60%" in notes and "clears the pre-set 10% margin" in notes
    assert "much higher than its test-set RMSE" in notes
    assert "under-predicts every month" in notes
    assert "only 1 distinct value(s)" in notes


def test_honesty_notes_do_not_cry_wolf():
    models = pd.DataFrame(metric_rows("X_naive_locf", "naive", 2.0, 3.0) + metric_rows("X_ridge", "ridge", 2.5, 3.5, cv=3.0))
    actual = [1.0, 3.0, 2.0, 4.0]
    forecasts = pd.DataFrame(forecast_rows("X_ridge", [2.0, 2.5, 3.0, 3.1], actual))   # mixed-sign errors, 4 distinct values
    board = an.build_scoreboard(models, forecasts)
    notes = " ".join(an.honesty_notes(board, forecasts, {"X_ridge": "ridge", "X_naive_locf": "naive_locf"}))
    assert "does NOT clear" in notes
    assert "same sign" not in notes and "distinct value" not in notes and "cross-validation" not in notes


def test_scoreboard_counts_only_months_every_model_forecast():
    """A model that could not forecast some test months shrinks the common scored set for all, with a note."""
    models = pd.DataFrame(metric_rows("X_naive_locf", "naive", 2.0, 3.0) + metric_rows("X_tree", "tree", 1.0, 1.2, cv=4.0))
    actual = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    forecasts = pd.DataFrame(forecast_rows("X_naive_locf", [0.0] * 6, actual)
                             + forecast_rows("X_tree", [1.0] * 4, actual[2:], start="2026-03-01"))
    board = an.build_scoreboard(models, forecasts)
    assert (board["test_months"] == 4).all()
    notes = " ".join(an.honesty_notes(board, forecasts, {"X_tree": "tree", "X_naive_locf": "naive_locf"}))
    assert "same 4 months" in notes and "up to 6" in notes
