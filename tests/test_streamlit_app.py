"""Smoke tests for the Streamlit app using Streamlit's AppTest (no browser, no network)."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from tests.helpers import write_series

APP = str(config.ROOT_DIR / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    """Streamlit caches data between runs in one process; start each test clean."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def run_app(timeout=60) -> AppTest:
    """Run the app script once and return the AppTest handle."""
    return AppTest.from_file(APP, default_timeout=timeout).run()


def texts(at: AppTest) -> str:
    """All markdown, info, warning and caption text on the page, joined."""
    parts = [el.value for el in at.markdown] + [el.value for el in at.info] + [el.value for el in at.warning] + \
            [el.value for el in at.caption] + [el.value for el in at.subheader]
    return "\n".join(parts)


def test_app_runs_on_the_demo_database_without_errors():
    """With no live database the committed demo snapshot is shown."""
    at = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["1. Signals", "2. Nowcast vs official", "3. Divergences"]
    page = texts(at)
    assert "Active state: California (CA)" in page                      # state from config.py, shown prominently
    assert "demo snapshot" in page
    assert len(at.dataframe) >= 4                                       # signals, scoreboard, history, divergences...


def test_active_state_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "STATE", "TX")
    at = run_app()
    assert not at.exception
    page = texts(at)
    assert "Active state: Texas (TX)" in page
    assert "No MSRS rows stored yet" in page                            # the demo holds California's state data only
    assert "Retail: Total retail excl. nonstore retailers" not in page  # California's series are not shown for Texas


def test_blocked_sources_show_no_data_yet_not_fake_data():
    at = run_app()
    page = texts(at)
    for needle in ("No Indeed data yet", "No VIIRS data yet"):
        assert needle in page
    assert "No FRED data yet" not in page                               # the demo holds the real FRED official series
    assert "Do not build anything commercial on scraped Indeed data" in page   # disclaimer is always visible


def test_raw_normalized_toggle_changes_the_values():
    at = run_app()
    normalized = at.dataframe[0].value
    assert (normalized["Unit"] == "z-score").all()
    at.radio(key="value_mode").set_value("Raw (each signal's own unit)").run()
    assert not at.exception
    raw = at.dataframe[0].value
    assert (raw["Unit"] == "percent_yoy").all()
    assert not raw["Latest reading"].equals(normalized["Latest reading"])
    assert any("different units" in w.value for w in at.warning)


def test_nowcast_tab_shows_weakness_warning_scoreboard_and_error_history():
    """Retail: ridge and tree beat naive on the four months every model could forecast."""
    at = run_app()
    assert any("weak and not convincing" in w.value for w in at.warning)
    page = texts(at)
    assert "Full error history" in page and "Test-period accuracy" in page
    assert "Every model is scored on the same 4 months" in page
    frames = [d.value for d in at.dataframe]
    board = next(f for f in frames if "Test RMSE" in f.columns)
    assert {"naive_locf", "naive_seasonal", "ridge", "tree"} == set(board["Model"])
    assert (board["Test months"] == 4).all()
    history = next(f for f in frames if "Error (official - prediction)" in f.columns)
    assert history.groupby("Model").size().to_dict() == {"naive_locf": 6, "naive_seasonal": 6, "ridge": 4, "tree": 4}


def test_unemployment_target_shows_real_official_data_and_models_losing_to_naive():
    at = run_app()
    at.selectbox(key="target").set_value("unemployment").run()
    assert not at.exception
    frames = [d.value for d in at.dataframe]
    board = next(f for f in frames if "Test RMSE" in f.columns)
    scores = board.set_index("Model")
    assert set(scores.index) == {"naive_locf", "naive_seasonal", "ridge", "tree"}
    assert scores.loc["naive_locf", "Test MAE"] == pytest.approx(0.0333, abs=1e-3)
    assert scores.loc["naive_locf", "Test RMSE"] == pytest.approx(0.0577, abs=1e-3)
    assert scores.loc["naive_seasonal", "Test MAE"] == pytest.approx(0.30, abs=1e-6)
    assert (scores.loc[["ridge", "tree"], "Clears 10% margin"] == "no").all()      # naive wins
    assert (scores["Test months"] == 6).all()
    assert not any("FRED_API_KEY" in i.value for i in at.info)


def test_divergence_tab_has_summary_from_rules():
    at = run_app()
    page = texts(at)
    assert "Within retail (each MSRS subsector against total retail)" in page
    assert "not by an LLM" in page
    assert "standard deviations" in page
    assert any("Across sources: no comparison is possible yet" in m.value for m in at.markdown)


def test_empty_database_shows_no_data_everywhere(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEMO_DB_PATH", tmp_path / "missing.db")
    at = run_app()
    assert not at.exception
    page = texts(at)
    assert "No data yet for California" in page
    assert "No database with data was found" in page
    assert "No data yet" in page and "no signal has a z-score" in page
    assert len(at.dataframe) == 0


def test_live_database_is_preferred_and_partial_data_is_not_filled():
    """A live database with 12 months of one series: 12 months of z-scores cannot exist, so none are shown."""
    write_series("msrs_yoy_TOTAL", "msrs", "percent_yoy", "state", "California", 12)
    at = run_app()
    assert not at.exception
    page = texts(at)
    assert "Reading the live database" in page
    assert "No z-scores yet" in page and "12 months of history" in page
    assert not any("Test RMSE" in d.value.columns for d in at.dataframe)      # no models in this database
