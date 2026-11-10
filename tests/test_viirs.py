"""Tests for src/ingest/viirs.py. Earth Engine is replaced by simple fakes."""

import pandas as pd
import pytest

import config
from src import db
from src.ingest import viirs


def feature(date, rad, low):
    """Build an Earth Engine style feature dict."""
    return {"properties": {"date": date, "rad": rad, "low": low}}


def test_stats_from_features_converts_none_to_nan():
    """Earth Engine returns None for empty areas; we keep that as NaN."""
    df = viirs.stats_from_features([feature("2024-02-01", None, 0.0), feature("2024-01-01", 2.0, 0.1)])
    assert list(df["period_start"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert pd.isna(df.loc[1, "avg_rad"])


def test_months_over_20_percent_low_coverage_are_dropped():
    """> 20% low coverage drops the month; exactly 20% is kept."""
    stats = viirs.stats_from_features([
        feature("2024-01-01", 5.0, 0.05),
        feature("2024-02-01", 5.0, 0.20),
        feature("2024-03-01", 5.0, 0.21),
        feature("2024-04-01", 5.0, 1.00),
    ])
    kept, dropped = viirs.select_valid_months(stats)
    assert list(kept["period_start"].dt.month) == [1, 2]
    assert list(dropped["period_start"].dt.month) == [3, 4]


def test_unusable_months_are_dropped_not_imputed():
    """A month with no radiance or no coverage figure is dropped."""
    stats = viirs.stats_from_features([feature("2024-01-01", None, 0.0), feature("2024-02-01", 3.0, None)])
    kept, dropped = viirs.select_valid_months(stats)
    assert kept.empty and len(dropped) == 2


def test_coverage_is_stored_as_quality_score():
    """quality_score = share of well-covered pixels."""
    kept, _ = viirs.select_valid_months(viirs.stats_from_features([feature("2024-01-01", 5.0, 0.1)]))
    rows = viirs.build_signal_rows(kept, "metro", "Los Angeles")
    assert rows.loc[0, "quality_score"] == pytest.approx(0.9)
    assert rows.loc[0, "signal_id"] == "viirs_avg_radiance"
    assert rows.loc[0, "period_end"] == pd.Timestamp("2024-01-31")


class FakeEE:
    """Stands in for the ``ee`` module; viirs functions are patched around it."""


def patch_ee(monkeypatch, stats_by_geo, fail_for=()):
    """Replace the two Earth Engine calls with fakes keyed by geometry name."""
    def fake_geometry(ee, fips, counties):
        return "STATE" if counties is None else counties[0]

    def fake_fetch(ee, geometry, start, end=None):
        if geometry in fail_for:
            raise RuntimeError("Earth Engine quota exceeded")
        return stats_by_geo(geometry)

    monkeypatch.setattr(viirs, "get_geometry", fake_geometry)
    monkeypatch.setattr(viirs, "fetch_monthly_stats", fake_fetch)


def good_months(geometry):
    """13 months of good data, then one low-coverage month."""
    months = pd.date_range("2023-01-01", periods=14, freq="MS").strftime("%Y-%m-%d")
    feats = [feature(m, 10.0 + i, 0.02) for i, m in enumerate(months)]
    feats[-1] = feature(months[-1], 99.0, 0.5)
    return viirs.stats_from_features(feats)


def test_ingest_success_stores_rows_and_skips_dropped_month(monkeypatch):
    """Good months are stored, the low-coverage month is absent (never imputed)."""
    patch_ee(monkeypatch, good_months)
    result = viirs.ingest("Wyoming", ee=FakeEE(), start_date="2023-01-01")
    assert result.status == "success"
    assert result.metro_coverage == 1.0
    rows = db.read_signals(source="viirs", geo_name="Cheyenne, WY")
    assert len(rows) == 13
    assert "2024-02-01" not in set(rows["period_start"])
    assert pd.notna(rows["normalized_value"].iloc[-1])
    assert (rows["quality_score"] > 0.9).all()
    assert len(db.read_signals(source="viirs", geo_level="state")) == 13
    assert db.read_scrapes(source="viirs").loc[0, "status"] == "success"


def test_ingest_partial_when_one_area_fails(monkeypatch):
    """One failing metro makes the run partial but keeps the rest."""
    patch_ee(monkeypatch, good_months, fail_for=("Natrona",))
    result = viirs.ingest("Wyoming", ee=FakeEE(), start_date="2023-01-01")
    assert result.status == "partial"
    assert "Casper" in result.error_message and "quota" in result.error_message
    assert result.metro_coverage == 0.5


def test_ingest_fails_when_every_area_fails(monkeypatch):
    """If nothing can be fetched the run is a logged failure."""
    patch_ee(monkeypatch, good_months, fail_for=("Laramie", "Natrona", "STATE"))
    result = viirs.ingest("Wyoming", ee=FakeEE(), start_date="2023-01-01")
    assert result.status == "failed"
    assert db.read_scrapes(source="viirs").loc[0, "status"] == "failed"


def test_ingest_fails_when_nothing_passes_coverage(monkeypatch):
    """All months sparse -> failure, no rows."""
    patch_ee(monkeypatch, lambda g: viirs.stats_from_features([feature("2024-01-01", 5.0, 0.9)]))
    result = viirs.ingest("Wyoming", ee=FakeEE(), start_date="2023-01-01")
    assert result.status == "failed"
    assert len(db.read_signals()) == 0


def test_missing_project_is_a_clear_logged_failure(monkeypatch):
    """Without EE_PROJECT the run fails with setup instructions."""
    monkeypatch.setattr(config, "EE_PROJECT", None)
    result = viirs.ingest("California")
    assert result.status == "failed"
    log = db.read_scrapes(source="viirs")
    assert "EE_PROJECT" in log.loc[0, "error_message"]


def test_unauthenticated_earth_engine_is_a_clear_logged_failure(monkeypatch):
    """With a project but no credentials, the error says how to authenticate."""
    monkeypatch.setattr(config, "EE_PROJECT", "some-project")
    monkeypatch.delenv("EE_SERVICE_ACCOUNT", raising=False)
    import ee

    def refuse(*args, **kwargs):
        raise ee.EEException("Please authorize access to your Earth Engine account")

    monkeypatch.setattr(ee, "Initialize", refuse)
    result = viirs.ingest("California")
    assert result.status == "failed"
    assert "earthengine authenticate" in result.error_message
