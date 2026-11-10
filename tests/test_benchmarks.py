"""Tests for src/models/benchmarks.py (synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.models import benchmarks as bm


def make_rows(n=30, lag_days=45, last_unlabeled=2):
    """Build one target's rows: value = 10 + month index, published ``lag_days`` after month end."""
    months = pd.date_range("2020-01-01", periods=n, freq="MS")
    month_end = months + pd.offsets.MonthEnd(0)
    values = 10.0 + np.arange(n)
    if last_unlabeled:
        values[-last_unlabeled:] = np.nan
    return pd.DataFrame({
        "target_name": "t", "target_month": months,
        "as_of": month_end + pd.Timedelta(days=1),
        "target_available": month_end + pd.Timedelta(days=lag_days),
        "target_value": values})


def test_locf_uses_the_latest_published_value_not_the_calendar_month():
    """With a 45-day lag, month T's nowcast carries forward month T-2 (T-1 is not public yet)."""
    rows = make_rows()
    pred = bm.last_observation_carried_forward(rows)
    # Month index 10 (value 20): month 9 ends ~28 days before as_of -> not yet public (45d lag);
    # month 8 ended ~58 days before -> public. So the prediction is month 8's value.
    assert pred.iloc[10] == pytest.approx(10 + 8)


def test_locf_never_uses_the_target_month_itself():
    """With any realistic release lag the prediction for T is never T's own value."""
    rows = make_rows(lag_days=20, last_unlabeled=0)
    pred = bm.last_observation_carried_forward(rows)
    values = rows["target_value"].to_numpy()
    for i in range(3, 30):
        assert pred.iloc[i] != values[i]
        assert pred.iloc[i] == values[i - 1]   # a 20-day lag: the previous month is public


def test_locf_ignores_values_published_after_the_prediction_time():
    """A value whose publication date is after as_of must not be used."""
    rows = make_rows(lag_days=45)
    rows.loc[8, "target_value"] = 999.0
    rows.loc[8, "target_available"] = rows.loc[10, "as_of"] + pd.Timedelta(days=1)   # public just after month 10's as_of
    pred = bm.last_observation_carried_forward(rows)
    assert pred.iloc[10] == pytest.approx(10 + 7)    # month 8 is hidden, so month 7 is the latest public value
    assert pred.iloc[11] == pytest.approx(10 + 9)    # by month 11 it is public, but month 9 is newer


def test_locf_is_nan_before_anything_is_published():
    """The first months have no published history."""
    pred = bm.last_observation_carried_forward(make_rows())
    assert np.isnan(pred.iloc[0]) and np.isnan(pred.iloc[1])


def test_locf_nowcasts_unlabeled_months_too():
    """Months whose own value is not published yet still get a benchmark prediction."""
    rows = make_rows(last_unlabeled=3)
    pred = bm.last_observation_carried_forward(rows)
    assert pred.iloc[-1] == pytest.approx(rows["target_value"].dropna().iloc[-1])


def test_seasonal_naive_is_the_value_twelve_months_earlier():
    """Prediction for month T equals the value of T-12."""
    rows = make_rows()
    pred = bm.seasonal_naive(rows)
    assert pred.iloc[15] == pytest.approx(rows["target_value"].iloc[3])
    assert pred.iloc[27] == pytest.approx(rows["target_value"].iloc[15])
    assert pred.iloc[:12].isna().all()


def test_seasonal_naive_requires_the_earlier_value_to_be_public():
    """If T-12 was not yet published at as_of(T) (very long lag), the prediction is NaN."""
    rows = make_rows(n=30, lag_days=400)
    assert bm.seasonal_naive(rows).isna().all()


def test_seasonal_naive_is_nan_when_the_earlier_value_is_missing():
    """A missing value a year ago gives no prediction (never filled in)."""
    rows = make_rows()
    rows.loc[3, "target_value"] = np.nan
    assert np.isnan(bm.seasonal_naive(rows).iloc[15])


def test_all_benchmarks_returns_both_columns():
    """The combined table has one column per benchmark, indexed by target month."""
    out = bm.all_benchmarks(make_rows())
    assert list(out.columns) == [bm.LOCF, bm.SEASONAL]
    assert out.index.name == "target_month" and len(out) == 30


def test_mixed_targets_and_missing_columns_are_rejected():
    """The input must be one target with the required columns."""
    rows = make_rows()
    rows.loc[0, "target_name"] = "other"
    with pytest.raises(ValueError, match="exactly one target"):
        bm.seasonal_naive(rows)
    with pytest.raises(ValueError, match="missing columns"):
        bm.last_observation_carried_forward(rows.drop(columns=["as_of"]))
