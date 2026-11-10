"""Tests for src/ingest/fred.py with a fake FRED client (no key needed)."""

import pandas as pd
import pytest

from src import db
from src.ingest import fred


class FakeFred:
    """Fake ``fredapi.Fred``: returns monthly series, or raises like fredapi does."""

    def __init__(self, missing=(), bad_key=False):
        """``missing``: series ids that do not exist; ``bad_key``: reject the key."""
        self.missing = set(missing)
        self.bad_key = bad_key
        self.requested = []

    def get_series(self, series_id):
        """Return 14 monthly values (test data) for any known series."""
        self.requested.append(series_id)
        if self.bad_key:
            raise ValueError("Bad Request.  The value for variable api_key is not registered.")
        if series_id in self.missing:
            raise ValueError("Bad Request.  The series does not exist.")
        index = pd.date_range("2025-01-01", periods=14, freq="MS")
        return pd.Series([float(i) for i in range(14)], index=index)


def test_targets_fetches_state_and_national_series():
    """State series id is built from the abbreviation (CAUR for California)."""
    client = FakeFred()
    result = fred.ingest_targets("California", fred=client)
    assert result.status == "success"
    assert client.requested == ["CAUR", "UNRATE", "RSAFS"]
    rows = db.read_signals(source="fred_targets")
    assert set(rows["geo_level"]) == {"state", "national"}
    assert set(rows[rows.geo_level == "state"]["geo_name"]) == {"California"}
    assert set(rows["grain"]) == {"monthly"}
    assert rows["normalized_value"].notna().any()


def test_targets_switch_state():
    """Changing the state changes the FRED series id requested."""
    client = FakeFred()
    fred.ingest_targets("Texas", fred=client)
    assert client.requested[0] == "TXUR"


def test_missing_state_series_gives_partial_with_national_fallback():
    """No state unemployment series -> partial run, national series still stored."""
    result = fred.ingest_targets("California", fred=FakeFred(missing={"CAUR"}))
    assert result.status == "partial"
    assert "CAUR" in result.error_message
    assert set(db.read_signals(source="fred_targets")["geo_level"]) == {"national"}


def test_missing_api_key_is_a_clear_logged_failure(monkeypatch):
    """No FRED_API_KEY -> failed row explaining how to get one."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    result = fred.ingest_targets("California")
    assert result.status == "failed"
    assert "FRED_API_KEY" in db.read_scrapes(source="fred_targets").loc[0, "error_message"]


def test_rejected_api_key_is_reported_as_key_problem():
    """A bad key is not confused with a missing series."""
    result = fred.ingest_targets("California", fred=FakeFred(bad_key=True))
    assert result.status == "failed"
    assert "rejected the API key" in result.error_message


def test_non_monthly_series_is_rejected():
    """Daily data must be converted first (a later phase), not silently stored."""
    daily = pd.Series([float(i) for i in range(40)], index=pd.date_range("2025-01-01", periods=40, freq="D"))
    with pytest.raises(fred.IngestError, match="not a monthly series"):
        fred.series_to_rows(daily, "x", "national", "United States", "index", "fred_targets")
