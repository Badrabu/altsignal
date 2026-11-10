"""Tests for src/normalize.py (rolling z-scores)."""

import numpy as np
import pandas as pd
import pytest

from src.normalize import add_normalized_values, rolling_zscore


def monthly(values, start="2020-01-01"):
    """Make a monthly series from a list of values."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def test_first_twelve_periods_have_no_zscore():
    """A z-score needs a full 12-month baseline; earlier months stay empty."""
    z = rolling_zscore(monthly(range(14)), "monthly")
    assert z.iloc[:12].isna().all()
    assert z.iloc[12:].notna().all()


def test_zscore_matches_hand_calculation():
    """z = (x - mean of previous 12) / std (sample) of previous 12."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20]
    z = rolling_zscore(monthly(values), "monthly")
    baseline = np.array(values[:12], dtype=float)
    expected = (20 - baseline.mean()) / baseline.std(ddof=1)
    assert z.iloc[12] == pytest.approx(expected)


def test_current_value_is_not_in_its_own_baseline():
    """Changing the latest value must not change earlier z-scores."""
    base = list(range(1, 14))
    a = rolling_zscore(monthly(base + [5]), "monthly")
    b = rolling_zscore(monthly(base + [500]), "monthly")
    assert a.iloc[12] == pytest.approx(b.iloc[12])


def test_one_gap_in_baseline_is_tolerated_without_imputation():
    """One missing month: the z-score uses the 11 observed months only."""
    values = np.arange(20.0) ** 2
    series = monthly(values).drop(pd.Timestamp("2020-06-01"))
    z = rolling_zscore(series, "monthly")
    assert len(z) == len(series)                      # no month was added
    when = pd.Timestamp("2021-01-01")                 # its window (2020-01..2020-12) holds the gap
    baseline = series.loc["2020-01-01":"2020-12-01"]
    assert len(baseline) == 11
    expected = (series.loc[when] - baseline.mean()) / baseline.std(ddof=1)
    assert z.loc[when] == pytest.approx(expected)
    later = series.loc["2020-06-01":"2021-05-01"]     # window of 2021-06-01: gap at its first month
    assert z.loc["2021-06-01"] == pytest.approx((series.loc["2021-06-01"] - later.mean()) / later.std(ddof=1))


def test_two_gaps_in_baseline_leave_the_zscore_empty():
    """Two missing months in the window: no z-score, and nothing is filled."""
    series = monthly(np.arange(20.0) ** 2).drop([pd.Timestamp("2020-06-01"), pd.Timestamp("2020-08-01")])
    z = rolling_zscore(series, "monthly")
    assert len(z) == len(series)
    assert pd.isna(z.loc["2021-01-01"])               # window has 10 of 12 months
    assert pd.isna(z.loc["2021-06-01"])               # window 2020-06..2021-05 misses both
    assert pd.notna(z.loc["2021-07-01"])              # window 2020-07..2021-06 misses only 2020-08


def test_a_short_history_is_not_a_tolerated_gap():
    """Exactly 11 prior months exist at the 12th month: still no z-score."""
    z = rolling_zscore(monthly(np.arange(12.0)), "monthly")
    assert z.isna().all()


def test_the_october_2025_style_gap_recovers_after_one_month():
    """A single hole (like BLS October 2025) does not blank the following 12 months."""
    series = monthly(np.arange(40.0) % 7 + np.arange(40.0) * 0.1, start="2022-01-01")
    series = series.drop(pd.Timestamp("2025-02-01"))
    z = rolling_zscore(series, "monthly")
    after = z.loc["2025-03-01":]
    assert after.notna().all()


def test_flat_baseline_gives_no_zscore():
    """Zero variation would divide by zero, so the result is empty."""
    z = rolling_zscore(monthly([5] * 13), "monthly")
    assert pd.isna(z.iloc[12])


def test_weekly_window_is_52():
    """Weekly series need 52 weeks of history."""
    index = pd.date_range("2025-01-06", periods=53, freq="W-MON")
    z = rolling_zscore(pd.Series(np.arange(53.0), index=index), "weekly")
    assert z.iloc[:52].isna().all()
    assert pd.notna(z.iloc[52])


def test_weekly_baseline_tolerates_one_missing_week_but_not_two():
    """The same rule for weekly series: 51 of 52 weeks."""
    index = pd.date_range("2025-01-06", periods=60, freq="W-MON")
    weekly = pd.Series(np.arange(60.0) ** 1.5, index=index)
    one = rolling_zscore(weekly.drop(index[10]), "weekly")
    two = rolling_zscore(weekly.drop([index[10], index[11]]), "weekly")
    assert pd.notna(one.loc[index[55]])
    assert pd.isna(two.loc[index[55]])


def test_unknown_grain_raises():
    """Only monthly and weekly are supported."""
    with pytest.raises(ValueError):
        rolling_zscore(monthly([1, 2, 3]), "daily")


def test_add_normalized_values_keeps_series_separate():
    """Two series with different units are each normalized against themselves."""
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    frames = []
    for name, scale in [("a", 1.0), ("b", 1000.0)]:
        frames.append(pd.DataFrame({
            "signal_id": name, "geo_level": "state", "geo_name": "X", "grain": "monthly",
            "period_start": dates, "value": np.arange(13) * scale,
        }))
    out = add_normalized_values(pd.concat(frames, ignore_index=True))
    last = out[out["period_start"] == dates[-1]].set_index("signal_id")["normalized_value"]
    assert last["a"] == pytest.approx(last["b"])      # scale does not matter
