"""Tests for the state lookup table in config.py."""

import pytest

import config


def test_state_lookup_by_name_and_abbreviation():
    """Names and abbreviations (any case) resolve to the same state."""
    assert config.get_state_info("California").abbr == "CA"
    assert config.get_state_info("ca").name == "California"
    assert config.get_state_info("  texas ").fips == "48"


def test_active_state_is_used_by_default(monkeypatch):
    """Changing STATE changes what get_state_info() returns."""
    monkeypatch.setattr(config, "STATE", "Texas")
    assert config.get_state_info().abbr == "TX"


def test_unknown_state_raises():
    """An unknown state gives a clear error."""
    with pytest.raises(ValueError, match="Unknown state"):
        config.get_state_info("Atlantis")


def test_every_state_is_complete_and_unique():
    """All 50 states + DC have a unique abbreviation/FIPS and at least one metro."""
    names = config.list_states()
    assert len(names) == 51
    infos = [config.get_state_info(n) for n in names]
    assert len({i.abbr for i in infos}) == 51
    assert len({i.fips for i in infos}) == 51
    for info in infos:
        assert len(info.fips) == 2 and info.fips.isdigit()
        assert info.metros, info.name
        for metro in info.metros:
            assert metro.counties
            assert metro.indeed_location == f"{metro.name}, {info.abbr}"
            assert metro.geo_name == metro.indeed_location
    all_geo = [m.geo_name for i in infos for m in i.metros]
    assert len(all_geo) == len(set(all_geo))   # unique across states (Springfield MA vs MO)
