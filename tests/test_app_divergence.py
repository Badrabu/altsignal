"""Tests for app/divergence.py: the divergence rules and the plain-language summary."""

import numpy as np
import pandas as pd

import config
from app import divergence as dv


def z(values, start="2025-01-01"):
    """Build a monthly z-score series from a list (use ``np.nan`` for a missing month)."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def test_opposite_levels_diverge_and_gap_is_strong():
    result = dv.compare_pair(z([1.0, 1.5, 1.2, 1.4]), z([-0.2, -0.8, -1.0, -1.1]))
    assert result["status"] == dv.DIVERGE
    assert result["level_diverges"] and result["month"] == pd.Timestamp("2025-04-01")
    assert result["gap"] == 1.4 - (-1.1)
    assert result["severity"] in ("moderate", "strong")
    assert result["streak"] == 3   # months 2, 3 and 4 are opposite and outside the band


def test_severity_threshold_from_config():
    strong = dv.compare_pair(z([2.0]), z([-0.5]))               # gap 2.5
    moderate = dv.compare_pair(z([1.0]), z([-0.6]))             # gap 1.6
    assert strong["severity"] == "strong" and moderate["severity"] == "moderate"
    assert config.DIVERGENCE_STRONG_GAP == 2.0


def test_same_side_agrees_and_small_values_are_neutral():
    assert dv.compare_pair(z([1.0, 1.2]), z([0.8, 0.9]))["status"] == dv.AGREE
    near_zero = dv.compare_pair(z([1.0, 1.2]), z([-0.2, -0.3]))   # B is inside the noise band
    assert near_zero["status"] == dv.NEUTRAL and near_zero["severity"] is None


def test_noise_band_edge_counts_as_outside():
    assert dv.compare_pair(z([0.5]), z([-0.5]))["status"] == dv.DIVERGE
    assert dv.compare_pair(z([0.49]), z([-0.5]))["status"] == dv.NEUTRAL


def test_trend_divergence_without_level_divergence():
    a = z([0.0, 0.2, 0.4, 1.5])        # +1.5 over 3 months, level positive
    b = z([2.0, 1.9, 1.5, 1.3])        # -0.7 over 3 months, level positive: levels agree
    result = dv.compare_pair(a, b)
    assert result["status"] == dv.DIVERGE
    assert result["trend_diverges"] and not result["level_diverges"]
    assert result["streak"] == 0


def test_no_overlap_is_not_judged_and_nothing_is_filled_in():
    a = z([1.0, 1.0, np.nan, np.nan])
    b = z([np.nan, np.nan, -1.0, -1.0])
    result = dv.compare_pair(a, b)
    assert result["status"] == dv.NO_OVERLAP and result["z_a"] is None and result["gap"] is None
    assert dv.compare_pair(z([np.nan]), z([np.nan]))["status"] == dv.NO_OVERLAP


def test_compares_at_latest_common_month_only():
    a = z([1.0, 1.0, 1.0, 1.0])
    b = z([-1.0, -1.0, 1.0, np.nan])        # B has no value for the last month
    result = dv.compare_pair(a, b)
    assert result["month"] == pd.Timestamp("2025-03-01") and result["status"] == dv.AGREE


def test_trend_needs_the_month_three_back():
    a = z([np.nan, 1.0, 1.0, 1.0, 1.0])
    b = z([5.0, np.nan, 1.0, 1.0, 1.0])
    result = dv.compare_pair(a, b)
    assert result["trend_a"] is None   # month t-3 is missing for B at the first of these
    result = dv.compare_pair(z([1.0, 1.0, 1.0, 1.0]), z([1.0, 1.0, 1.0, 1.0]))
    assert result["trend_a"] == 0.0


def test_find_divergences_orders_and_skips_missing_columns():
    wide = pd.DataFrame({"a": z([1.0, 1.0]), "b": z([-2.0, -2.0]), "c": z([1.0, 1.0]), "d": z([np.nan, np.nan])})
    table = dv.find_divergences(wide, [("a", "b"), ("a", "c"), ("a", "d"), ("a", "missing")], {"a": "Alpha"})
    assert list(table["signal_b"]) == ["b", "c", "d"]            # diverge, agree, no overlap
    assert table.loc[0, "label_a"] == "Alpha" and table.loc[0, "label_b"] == "b"
    assert dv.find_divergences(wide, [], {}).empty


def test_pair_selection():
    wide = pd.DataFrame({"msrs_yoy_TOTAL": z([1.0]), "msrs_yoy_441": z([1.0]), "msrs_yoy_442": z([np.nan]),
                         "viirs_avg_radiance": z([0.3]), "indeed_postings_total": z([np.nan])})
    assert dv.within_retail_pairs(wide) == [("msrs_yoy_441", "msrs_yoy_TOTAL")]
    assert dv.cross_source_pairs(wide) == [("msrs_yoy_TOTAL", "viirs_avg_radiance")]
    assert dv.within_retail_pairs(wide.drop(columns="msrs_yoy_TOTAL")) == []


def test_describe_divergence_wording_comes_from_the_numbers():
    row = pd.Series({"month": pd.Timestamp("2026-06-01"), "label_a": "Jobs", "label_b": "Sales", "z_a": 1.25,
                     "z_b": -0.75, "gap": 2.0, "level_diverges": True, "trend_diverges": True,
                     "trend_a": 1.0, "trend_b": -0.5, "severity": "strong", "streak": 3})
    text = dv.describe_divergence(row)
    assert "In Jun 2026, Jobs was above its 12-month baseline (+1.2 standard deviations)" in text or "+1.3" in text
    assert "Sales was below (-0.8)" in text or "Sales was below (-0.7)" in text
    assert "strong gap of 2.0" in text
    assert "3 months in a row" in text
    assert "Jobs rose by 1.0" in text and "Sales fell by 0.5" in text


def test_summarize_is_deterministic_and_reports_missing_sources():
    wide = pd.DataFrame({"msrs_yoy_TOTAL": z([1.0, 1.5]), "msrs_yoy_441": z([-1.0, -1.2]), "msrs_yoy_442": z([0.9, 1.1])})
    labels = {"msrs_yoy_TOTAL": "Total", "msrs_yoy_441": "Cars", "msrs_yoy_442": "Furniture"}
    within = dv.find_divergences(wide, dv.within_retail_pairs(wide), labels)
    cross = dv.find_divergences(wide, dv.cross_source_pairs(wide), labels)
    first = dv.summarize("California", cross, within, ["Indeed job postings"])
    again = dv.summarize("California", cross, within, ["Indeed job postings"])
    assert first == again
    joined = "\n".join(first)
    assert "No data yet for: Indeed job postings" in first[0]
    assert "Across sources: no comparison is possible yet" in joined
    assert "2 pair(s) compared for California" in joined and "1 diverge" in joined and "1 agree" in joined
    assert "Cars was below its 12-month baseline" in joined and "Furniture" not in joined.split("Cars was below")[1]


def test_summarize_with_nothing_to_compare():
    empty = pd.DataFrame()
    paragraphs = dv.summarize("Texas", empty, empty, [])
    assert len(paragraphs) == 2 and all("no comparison is possible" in p for p in paragraphs)


def test_summarize_caps_listed_pairs():
    wide = pd.DataFrame({"msrs_yoy_TOTAL": z([2.0, 2.0])})
    for i in range(7):
        wide[f"msrs_yoy_44{i}"] = z([-2.0, -2.0])
    labels = {c: c for c in wide.columns}
    within = dv.find_divergences(wide, dv.within_retail_pairs(wide), labels)
    text = dv.summarize("California", pd.DataFrame(), within, [], max_listed=3)
    assert "4 more diverging pair(s) are in the table below." in text
