"""Tests for src/features/engineer.py: release lags, no look-ahead, no imputation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import config
from src.features import engineer, split as splitting
from tests.helpers import series_rows

TODAY = date(2026, 11, 10)


def ts(text):
    """Shorthand for a Timestamp."""
    return pd.Timestamp(text)


def test_as_of_is_day_after_month_end():
    """Prediction time = last day of the target month + 1 day."""
    assert engineer.as_of(ts("2026-02-01")) == ts("2026-03-01")
    assert engineer.as_of(ts("2026-12-01")) == ts("2027-01-01")


@pytest.mark.parametrize("lag,expected", [
    (1, "2026-06-01"),       # Indeed: the target month itself is complete at prediction time
    (14, "2026-05-01"),      # national unemployment: previous month
    (30, "2026-05-01"),      # state unemployment: previous month
    (90, "2026-03-01"),      # VIIRS (assumed)
    (115, "2026-02-01"),     # MSRS: about 4 months back
])
def test_latest_published_period(lag, expected):
    """P* is the latest period whose release date is not after the prediction time."""
    assert engineer.latest_published_period(ts("2026-06-01"), lag) == ts(expected)


def test_release_lag_lookup_prefers_series_then_source_and_never_guesses():
    """Series override source; unknown series have no lag at all."""
    assert engineer.release_lag("fred_unemployment_rate_state", "fred_targets") == 30
    assert engineer.release_lag("msrs_yoy_441", "msrs") == 115
    assert engineer.release_lag("mystery", "mystery_source") is None


def test_derive_series_features_values():
    """Lags, changes and rolling stats are computed on the normalized series."""
    z = pd.Series(np.arange(10.0) ** 2, index=pd.date_range("2025-01-01", periods=10, freq="MS"))
    f = engineer.derive_series_features(z)
    t = ts("2025-10-01")
    assert f.loc[t, "z"] == 81
    assert f.loc[t, "lag1"] == 64 and f.loc[t, "lag3"] == 36
    assert f.loc[t, "chg1"] == 17 and f.loc[t, "chg3"] == 45
    assert f.loc[t, "roll3_mean"] == pytest.approx((49 + 64 + 81) / 3)
    assert f.loc[t, "roll6_std"] == pytest.approx(np.std([16, 25, 36, 49, 64, 81], ddof=1))
    assert pd.isna(f.loc[ts("2025-02-01"), "lag3"])          # not enough history -> gap


def test_gaps_stay_gaps_no_forward_fill():
    """A missing month makes every feature that needs it NaN; nothing is filled."""
    z = pd.Series(np.arange(10.0), index=pd.date_range("2025-01-01", periods=10, freq="MS"))
    z = z.drop(ts("2025-05-01"))
    f = engineer.derive_series_features(z)
    assert pd.isna(f.loc[ts("2025-05-01"), "z"])
    assert pd.isna(f.loc[ts("2025-06-01"), "lag1"]) and pd.isna(f.loc[ts("2025-06-01"), "chg1"])
    assert pd.isna(f.loc[ts("2025-07-01"), "roll3_mean"])     # window contains the gap
    assert pd.notna(f.loc[ts("2025-06-01"), "chg3"])          # Jun - Mar: both exist
    assert pd.isna(f.loc[ts("2025-08-01"), "chg3"])           # Aug - May: May is missing
    assert pd.notna(f.loc[ts("2025-09-01"), "roll3_mean"])    # window after the gap is complete


def make_signals():
    """Synthetic signals: MSRS target (state), state + national unemployment, VIIRS, Indeed."""
    parts = [
        series_rows("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 90, seed=1, quality=0.2),
        series_rows("fred_unemployment_rate_state", "fred_targets", "percent", "state", "California", 90, seed=2),
        series_rows("fred_unemployment_rate_national", "fred_targets", "percent", "national", "United States", 90, seed=3),
        series_rows("viirs_avg_radiance", "viirs", "nW/cm2/sr", "state", "California", 90, seed=4, quality=0.93),
        series_rows("indeed_postings_total", "indeed", "postings", "state", "California", 40, start="2022-06-01", seed=5),
    ]
    return pd.concat(parts, ignore_index=True)


def build(signals, **kwargs):
    """Build the feature table with a fixed 'today' and no frozen split."""
    return engineer.build_feature_table(signals, TODAY, **kwargs)


def row_features(build_result, target_name, month):
    """Return the feature values of one row as a Series."""
    table = build_result.table
    row = table[(table["target_name"] == target_name) & (table["target_month"] == ts(month))].iloc[0]
    return row[splitting.feature_columns(table)]


def test_both_targets_built_with_meta_columns():
    """Targets, release dates and split labels are filled in."""
    result = build(make_signals())
    assert result.targets == {"retail_sales": "built: msrs_yoy_TOTAL",
                              "unemployment": "built: fred_unemployment_rate_state"}
    table = result.table
    first = table[table["target_name"] == "retail_sales"].iloc[0]
    assert first["as_of"] == engineer.as_of(first["target_month"])
    assert first["target_available"] == ts("2019-01-31") + pd.Timedelta(days=115)
    assert set(table["split"]) <= {"train", "pre_window", "test", "later", "unlabeled"}
    assert result.series_used["msrs_yoy_TOTAL.state"] == 115


def test_no_look_ahead_unpublished_data_cannot_change_a_row():
    """Scrambling every observation not yet published at as_of(T) leaves row T unchanged."""
    signals = make_signals()
    base = build(signals)
    target_month = ts("2023-08-01")
    deadline = engineer.as_of(target_month)

    scrambled = signals.copy()
    rng = np.random.default_rng(99)
    lags = [engineer.release_lag(s, src) for s, src in zip(scrambled["signal_id"], scrambled["source"])]
    available = pd.to_datetime(scrambled["period_end"]) + pd.to_timedelta(lags, unit="D")
    future = (available > deadline).to_numpy()
    assert future.any()
    for column in ("value", "normalized_value", "quality_score"):
        scrambled.loc[future, column] = rng.normal(100, 50, future.sum())
    changed = build(scrambled)

    for target in ("retail_sales", "unemployment"):
        pd.testing.assert_series_equal(row_features(base, target, "2023-08-01"),
                                       row_features(changed, target, "2023-08-01"))


def test_target_value_of_the_month_is_never_a_feature():
    """The label for T cannot appear in row T: changing it changes no feature."""
    signals = make_signals()
    base = build(signals)
    edited = signals.copy()
    hit = (edited["signal_id"] == "fred_unemployment_rate_state") & (edited["period_start"] == ts("2023-08-01"))
    edited.loc[hit, ["value", "normalized_value"]] = 1234.0
    changed = build(edited)
    pd.testing.assert_series_equal(row_features(base, "unemployment", "2023-08-01"),
                                   row_features(changed, "unemployment", "2023-08-01"))
    labels = changed.table.set_index(["target_name", "target_month"])["target_value"]
    assert labels[("unemployment", ts("2023-08-01"))] == 1234.0


def test_training_rows_do_not_depend_on_test_period_data():
    """Changing every signal from the test window on leaves all training-row features identical."""
    signals = make_signals()
    base = build(signals)
    boundary = base.boundaries["retail_sales"]
    perturbed = signals.copy()
    late = pd.to_datetime(perturbed["period_start"]) >= boundary.test_start
    perturbed.loc[late, ["value", "normalized_value"]] = 7777.0
    changed = build(perturbed, frozen=base.boundaries)

    def train_rows(result):
        t = result.table
        t = t[(t["target_name"] == "retail_sales") & (t["target_month"] < boundary.test_start)]
        return t[splitting.feature_columns(t)].reset_index(drop=True)

    pd.testing.assert_frame_equal(train_rows(base), train_rows(changed))


def test_features_use_the_published_period_not_the_calendar_month():
    """MSRS in row T=2023-08 is the z of 2023-04 (115-day lag), state unemployment that of 2023-07."""
    signals = make_signals()
    result = build(signals)
    feats = row_features(result, "retail_sales", "2023-08-01")
    z = signals.set_index(["signal_id", "period_start"])["normalized_value"]
    assert feats["msrs_yoy_TOTAL.state__z"] == z[("msrs_yoy_TOTAL", ts("2023-04-01"))]
    assert feats["fred_unemployment_rate_state.state__z"] == z[("fred_unemployment_rate_state", ts("2023-07-01"))]
    assert feats["indeed_postings_total.state__z"] == z[("indeed_postings_total", ts("2023-08-01"))]


def test_viirs_coverage_percentage_is_a_feature_and_only_for_viirs():
    """VIIRS gets a coverage_pct column; other sources do not."""
    table = build(make_signals()).table
    coverage = [c for c in table.columns if c.endswith("coverage_pct")]
    assert coverage == ["viirs_avg_radiance.state__coverage_pct"]
    assert table[coverage[0]].dropna().iloc[0] == pytest.approx(93.0)


def test_missing_viirs_month_is_an_explicit_gap_in_coverage_and_z():
    """A dropped (low coverage) VIIRS month yields NaN features, no filled value."""
    signals = make_signals()
    dropped = (signals["signal_id"] == "viirs_avg_radiance") & (signals["period_start"] == ts("2023-03-01"))
    result = build(signals[~dropped])
    feats = row_features(result, "retail_sales", "2023-06-01")    # VIIRS lag 90 -> P* = 2023-03
    assert pd.isna(feats["viirs_avg_radiance.state__z"])
    assert pd.isna(feats["viirs_avg_radiance.state__coverage_pct"])


def test_unemployment_falls_back_to_national_when_state_is_missing():
    """With no state series the national rate becomes the target."""
    signals = make_signals()
    signals = signals[signals["signal_id"] != "fred_unemployment_rate_state"]
    result = build(signals)
    assert result.targets["unemployment"] == "built: fred_unemployment_rate_national"
    assert set(result.table[result.table["target_name"] == "unemployment"]["target_series"]) == \
        {"fred_unemployment_rate_national"}


def test_target_without_data_is_reported_and_has_no_rows():
    """With only MSRS present, unemployment is 'no data'."""
    signals = make_signals()
    signals = signals[signals["source"] == "msrs"]
    result = build(signals)
    assert result.targets["unemployment"] == "no data"
    assert set(result.table["target_name"]) == {"retail_sales"}


def test_no_data_at_all_gives_empty_table():
    """An empty signals table builds nothing."""
    result = build(make_signals().iloc[0:0])
    assert result.table.empty and set(result.targets.values()) == {"no data"}


def test_series_without_a_release_lag_are_skipped_not_guessed():
    """An unknown source is left out and reported."""
    signals = pd.concat([make_signals(), series_rows("mystery", "mystery_src", "index", "national",
                                                     "United States", 60, seed=7)], ignore_index=True)
    result = build(signals)
    assert result.series_skipped == {"mystery.national": "no release lag configured"}
    assert not any(c.startswith("mystery") for c in result.table.columns)


def test_card_spending_is_optional_and_used_when_present():
    """No card column without data; with data it appears with its assumed lag."""
    without = build(make_signals())
    assert not any(c.startswith("card_spending") for c in without.table.columns)
    card = series_rows("card_spending_index", "fred_card", "index", "national", "United States", 90, seed=8)
    with_card = build(pd.concat([make_signals(), card], ignore_index=True))
    assert "card_spending_index.national__z" in with_card.table.columns
    assert with_card.series_used["card_spending_index.national"] == config.RELEASE_LAG_DAYS["fred_card"]


def test_metro_series_are_not_features_by_default():
    """Only state and national series enter the table."""
    metro = series_rows("viirs_avg_radiance", "viirs", "nW/cm2/sr", "metro", "Los Angeles, CA", 90, seed=9, quality=0.9)
    table = build(pd.concat([make_signals(), metro], ignore_index=True)).table
    assert not any("metro" in c for c in table.columns)


def test_rows_stop_at_the_last_month_whose_prediction_time_has_passed():
    """On 2026-11-10 the last row is October 2026 (as_of Nov 1); November is not yet predictable."""
    months = engineer.target_months_range(ts("2026-08-01"), date(2026, 11, 10))
    assert months[-1] == ts("2026-10-01")
    assert engineer.target_months_range(ts("2026-08-01"), date(2026, 11, 1))[-1] == ts("2026-10-01")
    assert engineer.target_months_range(ts("2026-08-01"), date(2026, 10, 31))[-1] == ts("2026-09-01")


# ---- the target's own published history -----------------------------------------

def hist(result, target, month):
    """Return the ``target_hist`` features of one row as a dict keyed by suffix."""
    feats = row_features(result, target, month)
    prefix = f"{config.TARGET_HISTORY_PREFIX}__"
    return {k[len(prefix):]: v for k, v in feats.items() if k.startswith(prefix)}


def raw_target(signals, signal_id, period):
    """Raw value of a signal in a month."""
    rows = signals[(signals["signal_id"] == signal_id) & (signals["period_start"] == ts(period))]
    return rows["value"].iloc[0]


def test_derive_target_history_values_and_gaps():
    """Last value, lags, changes, lagged change and rolling mean, with gaps kept."""
    values = pd.Series([1.0, 3.0, 6.0, 10.0, 15.0, 21.0], index=pd.date_range("2025-01-01", periods=6, freq="MS"))
    f = engineer.derive_target_history(values)
    t = ts("2025-06-01")
    assert f.loc[t, "last"] == 21 and f.loc[t, "lag1"] == 15 and f.loc[t, "lag3"] == 6
    assert f.loc[t, "chg1"] == 6 and f.loc[t, "chg3"] == 15
    assert f.loc[t, "chg1_lag1"] == 5                         # May - Apr
    assert f.loc[t, "roll3_mean"] == pytest.approx((10 + 15 + 21) / 3)
    gap = engineer.derive_target_history(values.drop(ts("2025-04-01")))
    assert pd.isna(gap.loc[ts("2025-04-01"), "last"])
    assert pd.isna(gap.loc[ts("2025-05-01"), "lag1"]) and pd.isna(gap.loc[ts("2025-05-01"), "chg1"])
    assert pd.isna(gap.loc[ts("2025-06-01"), "roll3_mean"])   # nothing is filled


def test_target_history_is_the_last_published_value_never_the_months_own():
    """MSRS (115-day lag) row T=2023-08 holds raw values of 2023-04 and earlier, not of 2023-08."""
    signals = make_signals()
    result = build(signals)
    h = hist(result, "retail_sales", "2023-08-01")
    assert h["last"] == raw_target(signals, "msrs_yoy_TOTAL", "2023-04-01")
    assert h["lag1"] == raw_target(signals, "msrs_yoy_TOTAL", "2023-03-01")
    assert h["lag3"] == raw_target(signals, "msrs_yoy_TOTAL", "2023-01-01")
    assert h["chg1"] == pytest.approx(h["last"] - h["lag1"])
    assert h["chg1_lag1"] == pytest.approx(h["lag1"] - raw_target(signals, "msrs_yoy_TOTAL", "2023-02-01"))
    # Unemployment (30-day lag) uses the previous month.
    u = hist(result, "unemployment", "2023-08-01")
    assert u["last"] == raw_target(signals, "fred_unemployment_rate_state", "2023-07-01")


def test_target_history_ignores_everything_unpublished_at_prediction_time():
    """Changing target values after P* (including the month itself) leaves the history features alone."""
    signals = make_signals()
    base = build(signals)
    edited = signals.copy()
    hit = (edited["signal_id"] == "msrs_yoy_TOTAL") & (edited["period_start"] > ts("2023-04-01"))
    edited.loc[hit, "value"] = 987.0
    changed = build(edited)
    assert hist(base, "retail_sales", "2023-08-01") == hist(changed, "retail_sales", "2023-08-01")
    assert hist(base, "retail_sales", "2023-08-01")["last"] != 987.0
    assert changed.table.set_index(["target_name", "target_month"])["target_value"][
        ("retail_sales", ts("2023-08-01"))] == 987.0     # still the label


def test_target_history_uses_the_target_release_lag(monkeypatch):
    """Shortening the configured lag moves P*, which proves the lag logic drives these features."""
    signals = make_signals()
    short = dict(config.RELEASE_LAG_DAYS, msrs=30)
    monkeypatch.setattr(config, "RELEASE_LAG_DAYS", short)
    h = hist(build(signals), "retail_sales", "2023-08-01")
    assert h["last"] == raw_target(signals, "msrs_yoy_TOTAL", "2023-07-01")


def test_target_history_gap_is_explicit():
    """A missing published month leaves the history NaN instead of using an older value."""
    signals = make_signals()
    drop = (signals["signal_id"] == "msrs_yoy_TOTAL") & (signals["period_start"] == ts("2023-04-01"))
    result = build(signals[~drop])
    h = hist(result, "retail_sales", "2023-08-01")
    assert pd.isna(h["last"]) and pd.isna(h["chg1"]) and pd.isna(h["roll3_mean"])
    assert h["lag1"] == raw_target(signals, "msrs_yoy_TOTAL", "2023-03-01")


def test_pandemic_regime_flag_marks_the_configured_months():
    """The flag is 1 only inside config.PANDEMIC_REGIME and is a meta column, not a feature."""
    table = build(make_signals()).table
    rows = table[table["target_name"] == "retail_sales"].set_index("target_month")["pandemic_regime"]
    assert rows[ts("2020-02-01")] == 0 and rows[ts("2020-03-01")] == 1
    assert rows[ts("2021-12-01")] == 1 and rows[ts("2022-01-01")] == 0
    assert "pandemic_regime" not in splitting.feature_columns(table)


def test_default_training_window_excludes_2020_21_without_dropping_rows():
    """2019-2021 labelled months are pre_window (kept in the table); training starts 2022-01."""
    result = build(make_signals())
    table = result.table[result.table["target_name"] == "retail_sales"].set_index("target_month")
    assert config.TRAIN_START == "2022-01-01"
    assert (table.loc[:"2021-12-01", "split"] == "pre_window").all()
    assert table.loc[ts("2022-01-01"), "split"] == "train"
    assert table.loc[ts("2020-04-01"), "target_value"] == table.loc[ts("2020-04-01"), "target_value"]   # not NaN
    boundary = result.boundaries["retail_sales"]
    assert (table.loc[boundary.test_start:boundary.test_end, "split"] == "test").all()


def test_train_start_none_restores_full_history_and_same_test_window(monkeypatch):
    """TRAIN_START=None trains on everything before the test window; the test window is identical."""
    default = build(make_signals())
    monkeypatch.setattr(config, "TRAIN_START", None)
    wide = build(make_signals())
    assert wide.boundaries == default.boundaries
    assert "pre_window" not in set(wide.table["split"])
    assert (wide.table["split"] == "train").sum() > (default.table["split"] == "train").sum()
