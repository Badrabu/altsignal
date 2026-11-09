"""Tests for src/process/clean.py (cleaning rules)."""

import numpy as np
import pandas as pd
import pytest

from src.process import clean


def row(**overrides):
    """One valid, month-aligned signal row (test data)."""
    base = {
        "signal_id": "s", "source": "msrs", "geo_level": "state", "geo_name": "California",
        "period_start": "2026-01-01", "period_end": "2026-01-31", "value": 1.0,
        "normalized_value": None, "unit": "percent_yoy", "quality_score": 0.5,
        "grain": "monthly", "fetched_at": "2026-11-09T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def frame(*rows):
    """Build a DataFrame from row dicts."""
    return pd.DataFrame(list(rows))


def test_valid_rows_pass_untouched():
    """A clean row is kept and nothing is dropped."""
    result = clean.clean_signals(frame(row()))
    assert len(result.kept) == 1 and result.dropped.empty and result.reasons() == {}


def test_missing_and_infinite_values_are_dropped_not_filled():
    """NaN and inf never become numbers."""
    result = clean.clean_signals(frame(
        row(value=np.nan), row(period_start="2026-02-01", period_end="2026-02-28", value=np.inf),
        row(period_start="2026-03-01", period_end="2026-03-31")))
    assert len(result.kept) == 1
    assert result.reasons() == {"missing or non-finite value": 2}


def test_misaligned_monthly_rows_are_dropped():
    """A 'monthly' row must start on the 1st and end on the month's last day."""
    result = clean.clean_signals(frame(
        row(period_start="2026-01-15", period_end="2026-02-14"),
        row(period_start="2026-02-01", period_end="2026-02-27")))
    assert result.kept.empty
    assert result.reasons() == {"monthly row not aligned to a calendar month": 2}


def test_weekly_rows_are_not_checked_for_month_alignment():
    """Only monthly rows must be month-aligned."""
    result = clean.clean_signals(frame(row(grain="weekly", period_start="2026-01-05", period_end="2026-01-11",
                                           unit="postings")))
    assert len(result.kept) == 1


@pytest.mark.parametrize("unit,value", [("postings", -1), ("percent", 150), ("percent", -0.1),
                                         ("percent_yoy", -120), ("nW/cm2/sr", -5)])
def test_implausible_values_are_dropped(unit, value):
    """Values outside the unit's plausible range are removed."""
    result = clean.clean_signals(frame(row(unit=unit, value=value)))
    assert result.kept.empty
    assert result.reasons() == {"implausible value for its unit": 1}


def test_large_but_possible_values_are_kept():
    """A +200% year-over-year change (seen in real MSRS data) is plausible."""
    assert len(clean.clean_signals(frame(row(value=206.6))).kept) == 1


def test_viirs_low_coverage_rows_are_dropped_and_boundary_is_kept():
    """Quality below 0.80 (more than 20% low coverage) is dropped; exactly 0.80 stays."""
    viirs = dict(source="viirs", unit="nW/cm2/sr")
    result = clean.clean_signals(frame(
        row(quality_score=0.79, **viirs),
        row(period_start="2026-02-01", period_end="2026-02-28", quality_score=1 - 0.2, **viirs),
        row(period_start="2026-03-01", period_end="2026-03-31", quality_score=0.95, **viirs),
        row(period_start="2026-04-01", period_end="2026-04-30", quality_score=None, **viirs)))
    assert sorted(result.kept["period_start"].dt.month) == [2, 3]
    assert result.reasons() == {"VIIRS low coverage (more than 20% of pixels)": 2}


def test_other_sources_are_not_filtered_by_coverage_rule():
    """The 20% rule is VIIRS-only; a low MSRS coverage score is kept as a feature."""
    assert len(clean.clean_signals(frame(row(quality_score=0.05))).kept) == 1


def test_official_sources_get_full_quality_when_missing():
    """FRED values are the published numbers, so quality defaults to 1.0."""
    result = clean.clean_signals(frame(row(source="fred_targets", unit="percent", value=4.0, quality_score=None)))
    assert result.kept.loc[0, "quality_score"] == 1.0


def test_duplicate_keys_keep_last():
    """Two rows with one key collapse to one."""
    result = clean.clean_signals(frame(row(value=1.0), row(value=2.0)))
    assert list(result.kept["value"]) == [2.0]
    assert result.reasons() == {"duplicate key": 1}


def test_check_units_rejects_mixed_units_inside_a_signal():
    """One signal_id with two units is a bug and stops the build."""
    with pytest.raises(clean.ProcessError, match="Mixed units"):
        clean.check_units(frame(row(unit="percent"), row(period_start="2026-02-01", unit="index")))


def test_check_units_rejects_missing_unit():
    """A series without a unit cannot be compared safely."""
    with pytest.raises(clean.ProcessError, match="without a unit"):
        clean.check_units(frame(row(unit=None)))


def test_check_units_allows_different_units_in_different_signals():
    """Different signals may have different units; they are just never mixed."""
    clean.check_units(frame(row(signal_id="a", unit="percent"), row(signal_id="b", unit="index")))
