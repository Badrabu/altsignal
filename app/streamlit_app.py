"""AltSignal dashboard (phase five).

Run it from the project root:

    streamlit run app/streamlit_app.py

Three tabs for the active state (``STATE`` in ``config.py``):

1. Signals: each signal's current trend and latest reading (raw or normalized).
2. Nowcast vs official: model and naive forecasts against the official release,
   with the full error history and test-period MAE/RMSE.
3. Divergences: where signals disagree, with a plain-language summary written
   by fixed Python rules (no LLM).

The app only reads, and only through ``src/db.py``. A source with no data is
shown as "no data yet"; nothing is ever filled in or invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # lets `streamlit run app/streamlit_app.py` import config, src and app
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

import config
from app import analysis, data_loading as dl, divergence as dv
from src import db

TREND_TEXT = {analysis.RISING: "\u25B2 rising", analysis.FALLING: "\u25BC falling",
              analysis.FLAT: "\u25BA flat", analysis.UNKNOWN: "\u2013 not enough data"}

INDEED_DISCLAIMER = (
    "Indeed has no public API. Indeed data comes from an open-source scraper used for personal research and "
    "education only. **Do not build anything commercial on scraped Indeed data**, and check Indeed's terms of service.")

st.set_page_config(page_title="AltSignal", layout="wide")


# ---------------------------------------------------------------------------
# Cached data access (everything goes through src/db.py via app/data_loading.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_signals(state_name: str, db_path: str, mtime: float) -> pd.DataFrame:
    """Load the active state's monthly signals (cached; ``mtime`` refreshes it when the file changes).

    Args:
        state_name: Full state name.
        db_path: Database file.
        mtime: File modification time (only used as a cache key).

    Returns:
        The table from ``data_loading.load_signals``.
    """
    return dl.load_signals(state_name, db_path)


@st.cache_data(show_spinner=False)
def cached_scrapes(db_path: str, mtime: float) -> pd.DataFrame:
    """Load the scrape log (cached).

    Args:
        db_path: Database file.
        mtime: File modification time (only used as a cache key).

    Returns:
        The ``scrapes`` table, newest first (empty in the demo snapshot).
    """
    return dl._safe(db.read_scrapes, db_path=db_path)


@st.cache_data(show_spinner=False)
def cached_models(state_abbr: str, target: str, db_path: str, mtime: float) -> pd.DataFrame:
    """Load one target's model metrics (cached).

    Args:
        state_abbr: Two-letter abbreviation.
        target: Target name.
        db_path: Database file.
        mtime: File modification time (only used as a cache key).

    Returns:
        The table from ``data_loading.load_models``.
    """
    return dl.load_models(state_abbr, target, db_path)


@st.cache_data(show_spinner=False)
def cached_forecasts(model_ids: tuple[str, ...], db_path: str, mtime: float) -> pd.DataFrame:
    """Load forecasts for some models (cached).

    Args:
        model_ids: Model ids to keep.
        db_path: Database file.
        mtime: File modification time (only used as a cache key).

    Returns:
        The table from ``data_loading.load_forecasts``.
    """
    return dl.load_forecasts(list(model_ids), db_path)


def fmt_month(value) -> str:
    """Format a date as "Jun 2026", or "-" if missing.

    Args:
        value: A timestamp or ``NaT``.

    Returns:
        Text for display.
    """
    return "-" if pd.isna(value) else pd.Timestamp(value).strftime("%b %Y")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header(info: config.StateInfo, choice: dl.DatabaseChoice, signals: pd.DataFrame) -> None:
    """Show the title, the active state (prominently) and where the data comes from.

    Args:
        info: The active state.
        choice: Which database is being read.
        signals: Loaded signals (for the snapshot date).
    """
    st.title("AltSignal")
    st.caption("Monthly nowcasts of official US economic indicators from alternative data.")
    st.markdown(f"## Active state: {info.name} ({info.abbr})")
    st.caption("The state is set by one line, `STATE` in `config.py`. Change it and re-run the pipelines to switch.")
    fetched = dl.snapshot_time(signals)
    st.info(choice.note + (f" Newest data fetched on {fetched}." if fetched else ""))
    if signals.empty:
        st.warning(f"No data yet for {info.name}. Every tab below shows what is missing.")


# ---------------------------------------------------------------------------
# Tab 1: signals
# ---------------------------------------------------------------------------
def with_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an empty row for every missing month of each series, so charts show gaps instead of joining across them.

    Args:
        frame: Columns ``period_start``, ``label`` and ``y``.

    Returns:
        The same data with ``y`` empty for the months a series does not have.
    """
    parts = []
    for label, part in frame.groupby("label"):
        months = pd.date_range(part["period_start"].min(), part["period_start"].max(), freq="MS")
        parts.append(part.set_index("period_start")["y"].reindex(months).rename_axis("period_start").reset_index().assign(label=label))
    return pd.concat(parts, ignore_index=True)


def line_chart(frame: pd.DataFrame, value_title: str, zero_line: bool) -> alt.Chart:
    """Draw monthly lines for one or more series (gaps stay gaps).

    Args:
        frame: Columns ``period_start``, ``label`` and ``y``.
        value_title: Axis title.
        zero_line: Draw a reference line at zero (for z-scores).

    Returns:
        An Altair chart.
    """
    base = alt.Chart(with_gaps(frame)).mark_line(point=True).encode(
        x=alt.X("period_start:T", title="Month"), y=alt.Y("y:Q", title=value_title),
        color=alt.Color("label:N", title="Signal"),
        tooltip=[alt.Tooltip("label:N", title="Signal"), alt.Tooltip("period_start:T", title="Month", format="%b %Y"),
                 alt.Tooltip("y:Q", title=value_title, format=".2f")])
    if zero_line:
        base = alt.layer(base, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q"))
    return base.properties(height=320).interactive()


def render_signals_tab(signals: pd.DataFrame, status: pd.DataFrame) -> None:
    """Tab 1: latest reading and current trend of every signal, with a raw/normalized toggle.

    Args:
        signals: Loaded signals for the active state.
        status: Source status table from ``data_loading.source_status``.
    """
    st.subheader("Signals: latest reading and current trend")
    mode = st.radio("Values", ["Normalized (z-score vs own 12-month baseline)", "Raw (each signal's own unit)"],
                    horizontal=True, key="value_mode")
    normalized = mode.startswith("Normalized")
    value_col = "normalized_value" if normalized else "value"
    if normalized:
        st.caption("A z-score says how many standard deviations a reading is from the signal's own previous 12 months, "
                   "so different signals can be compared. A signal needs 12 months of history before it has one.")
    else:
        st.warning("Raw values are in different units (for example percent, postings, radiance). "
                   "Do not compare them across rows.")
    st.caption("Trend rule: mean of the last 3 months against the mean of the 3 months before, relative to the series' "
               f"own spread (at least {config.TREND_MIN_RATIO} standard deviations = rising or falling). "
               "Months only; gaps in the last 6 months give 'not enough data'. State-level and national series, monthly grain.")

    summary = analysis.summarize_signals(signals, value_col)
    for _, src in status.iterrows():
        st.markdown(f"#### {src['title']}")
        if src["source"] == config.SOURCE_INDEED:
            st.caption(INDEED_DISCLAIMER)
        if not src["has_data"]:
            st.info("**No data yet.** " + src["no_data"])
            continue
        part = summary[summary["source"] == src["source"]]
        shown = part[part["latest"].notna()]
        if not shown.empty:
            table = pd.DataFrame({
                "Signal": shown["label"], "Scope": shown["geo_scope"], "Unit": shown["unit"],
                "Latest month": shown["latest_period"].map(fmt_month), "Latest reading": shown["latest"],
                "Change vs previous month": shown["change"], "Trend": shown["trend"].map(TREND_TEXT),
                "Months with data": shown["n_points"], "Data quality (0-1)": shown["quality"]})
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Latest reading": st.column_config.NumberColumn(format="%.2f"),
                "Change vs previous month": st.column_config.NumberColumn(format="%+.2f"),
                "Data quality (0-1)": st.column_config.NumberColumn(format="%.2f")})
        empty = part[part["latest"].isna()]
        if not empty.empty:
            what = "z-scores" if normalized else "values"
            st.info(f"**No {what} yet** for {len(empty)} series from this source" +
                    (" (a z-score needs 12 months of history)." if normalized else "."))
        if src["last_scrape"]:
            st.caption(f"Last pipeline run: {src['last_scrape']}.")

    available = summary[summary["latest"].notna()]
    if available.empty:
        return
    st.markdown("#### History")
    options = dict(zip(available["label"] + " (" + available["geo_scope"] + ")", available["signal_id"]))
    default = [k for k, v in options.items() if v == dv.REFERENCE_SIGNAL] or list(options)[:1]
    chosen = st.multiselect("Signals to plot", list(options), default=default)
    ids = [options[c] for c in chosen]
    plotted = signals[signals["signal_id"].isin(ids)].dropna(subset=[value_col])
    if plotted.empty:
        st.caption("Pick at least one signal to plot.")
        return
    # Raw values are only ever drawn together when they share a unit.
    groups = [("z-score", plotted)] if normalized else list(plotted.groupby("unit"))
    for unit, part in groups:
        title = "z-score" if normalized else dl.UNIT_LABELS.get(unit, unit)
        frame = part.rename(columns={value_col: "y"})[["period_start", "label", "y"]]
        st.altair_chart(line_chart(frame, title, zero_line=normalized), width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: nowcast vs official
# ---------------------------------------------------------------------------
def render_nowcast_tab(info: config.StateInfo, signals: pd.DataFrame, choice: dl.DatabaseChoice, mtime: float) -> None:
    """Tab 2: model and naive forecasts against the official release, with the error history.

    Args:
        info: The active state.
        signals: Loaded signals (for the official series).
        choice: Which database is being read.
        mtime: Database modification time (cache key).
    """
    st.subheader("Nowcast versus official release")
    st.warning(
        "**Read this first: the results are weak and not convincing.** The test window is only a handful of months, "
        "models that could be fitted are at best barely better than simple naive benchmarks (and for some targets no model could be fitted at all), and nothing here shows that the alternative "
        "signals add information. Treat every number below as a diagnostic, not as proof of nowcasting skill. "
        "The README has the details and caveats.")
    targets = list(config.FEATURE_TARGETS)
    target = st.selectbox("Target", targets, format_func=lambda t: dl.TARGET_TITLES.get(t, t), key="target")
    models = cached_models(info.abbr, target, str(choice.path), mtime)
    if models.empty:
        st.info("**No data yet.** " + dl.TARGET_NO_DATA.get(target, "No model results for this target."))
        return
    ids = tuple(sorted(models["model_id"].unique()))
    forecasts = cached_forecasts(ids, str(choice.path), mtime)
    short = {m: dl.model_short_name(m, info.abbr, target) for m in ids}
    scoreboard = analysis.build_scoreboard(models, forecasts)

    st.markdown("#### Test-period accuracy")
    window = scoreboard.iloc[0]
    st.caption(f"Frozen test window: {fmt_month(window['test_start'])} to {fmt_month(window['test_end'])}, never used for "
               "training, selection or tuning. Units are those of the target. Naive benchmarks were defined before any "
               "result was seen: last observation carried forward (`naive_locf`) and the value 12 months earlier (`naive_seasonal`).")
    table = pd.DataFrame({
        "Model": scoreboard["model_id"].map(short), "Type": scoreboard["kind"],
        "Test MAE": scoreboard["MAE"], "Test RMSE": scoreboard["RMSE"],
        "RMSE vs best naive": scoreboard["rmse_vs_naive"], "MAE vs best naive": scoreboard["mae_vs_naive"],
        f"Clears {config.MEANINGFUL_MARGIN:.0%} margin": scoreboard["clears_margin"].map(
            {True: "yes", False: "no"}).fillna("n/a (benchmark)"),
        "Training CV RMSE": scoreboard["CV_RMSE"], "Test months": scoreboard["test_months"]})
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Test MAE": st.column_config.NumberColumn(format="%.2f"), "Test RMSE": st.column_config.NumberColumn(format="%.2f"),
        "Training CV RMSE": st.column_config.NumberColumn(format="%.2f"),
        "RMSE vs best naive": st.column_config.NumberColumn(format="percent"),
        "MAE vs best naive": st.column_config.NumberColumn(format="percent")})
    st.caption("'vs best naive' is the share by which the error is lower than the best naive benchmark (negative = worse). "
               "A model counts as meaningfully better only if both MAE and RMSE clear the margin, which was fixed in advance.")
    notes = analysis.honesty_notes(scoreboard, forecasts, short)
    if notes:
        st.markdown("**What the numbers say (and do not say):**")
        st.markdown("\n".join(f"- {n}" for n in notes))

    st.markdown("#### Nowcast versus official release")
    official, official_id = dl.official_series(target, signals)
    if official.empty:
        st.info("**No data yet** for the official series, so no comparison line can be drawn.")
    shown_models = st.multiselect("Models to show", list(short.values()), default=list(short.values()), key="shown_models")
    chosen = [m for m, s in short.items() if s in shown_models]
    layers = []
    if not official.empty:
        cutoff = pd.to_datetime(window["test_start"]) - pd.DateOffset(months=18)
        recent = official[official["period_start"] >= cutoff][["period_start", "value"]].assign(series="Official release")
        layers.append(alt.Chart(recent).mark_line(color="black", point=True, strokeWidth=3).encode(
            x=alt.X("period_start:T", title="Month"), y=alt.Y("value:Q", title=dl.TARGET_TITLES.get(target, target)),
            tooltip=[alt.Tooltip("period_start:T", format="%b %Y"), alt.Tooltip("value:Q", format=".2f")]))
    pred = forecasts[forecasts["model_id"].isin(chosen)].assign(model=lambda d: d["model_id"].map(short))
    if not pred.empty:
        layers.append(alt.Chart(pred).mark_line(point=True, strokeDash=[5, 3]).encode(
            x="target_period:T", y=alt.Y("predicted_value:Q", title=dl.TARGET_TITLES.get(target, target)),
            color=alt.Color("model:N", title="Model"),
            tooltip=[alt.Tooltip("model:N"), alt.Tooltip("target_period:T", title="Month", format="%b %Y"),
                     alt.Tooltip("predicted_value:Q", title="Prediction", format=".2f"),
                     alt.Tooltip("actual_value:Q", title="Official", format=".2f")]))
    if layers:
        band = pd.DataFrame({"start": [pd.to_datetime(window["test_start"])], "end": [pd.to_datetime(window["test_end"])]})
        shade = alt.Chart(band).mark_rect(opacity=0.12, color="orange").encode(x="start:T", x2="end:T")
        st.altair_chart(alt.layer(shade, *layers).resolve_scale(color="independent").properties(height=360),
                        width="stretch")
        st.caption("Black line: official release. Dashed lines: model and naive predictions. Shaded: frozen test window. "
                   "Predictions after the last official month are live nowcasts that cannot be scored yet.")

    st.markdown("#### Full error history")
    scored = forecasts[forecasts["actual_value"].notna() & forecasts["model_id"].isin(chosen)].assign(
        model=lambda d: d["model_id"].map(short))
    if scored.empty:
        st.info("**No data yet:** no forecast has been compared with an official value.")
    else:
        err = alt.Chart(scored).mark_line(point=True).encode(
            x=alt.X("target_period:T", title="Month"), y=alt.Y("error:Q", title="Error = official - prediction"),
            color=alt.Color("model:N", title="Model"),
            tooltip=[alt.Tooltip("model:N"), alt.Tooltip("target_period:T", format="%b %Y"), alt.Tooltip("error:Q", format="+.2f")])
        st.altair_chart(alt.layer(err, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q"))
                        .properties(height=280), width="stretch")
        history = pd.DataFrame({
            "Month": scored["target_period"].map(fmt_month), "Model": scored["model"],
            "Prediction": scored["predicted_value"], "Official": scored["actual_value"],
            "Error (official - prediction)": scored["error"], "Absolute error": scored["error"].abs()})
        st.dataframe(history, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in
            ("Prediction", "Official", "Error (official - prediction)", "Absolute error")})
    pending = forecasts[forecasts["actual_value"].isna() & forecasts["model_id"].isin(chosen)]
    if not pending.empty:
        st.markdown("**Live nowcasts (official value not published yet)**")
        live = pending.assign(model=lambda d: d["model_id"].map(short)).pivot_table(
            index="target_period", columns="model", values="predicted_value")
        live.index = [fmt_month(i) for i in live.index]
        st.dataframe(live.rename_axis("Month").reset_index(), hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in live.columns})


# ---------------------------------------------------------------------------
# Tab 3: divergences
# ---------------------------------------------------------------------------
def divergence_table(result: pd.DataFrame) -> pd.DataFrame:
    """Format a ``find_divergences`` result for display.

    Args:
        result: The table from ``divergence.find_divergences``.

    Returns:
        A display table.
    """
    return pd.DataFrame({
        "Signal A": result["label_a"], "Signal B": result["label_b"], "Month compared": result["month"].map(fmt_month),
        "z A": result["z_a"], "z B": result["z_b"], "Gap (A - B)": result["gap"],
        "Result": result["status"], "Severity": result["severity"].fillna(""),
        "Level split": result["level_diverges"].map({True: "yes", False: ""}),
        "Trend split (3 mo)": result["trend_diverges"].map({True: "yes", False: ""}),
        "Months in a row": result["streak"]})


def render_divergence_tab(info: config.StateInfo, signals: pd.DataFrame, status: pd.DataFrame) -> None:
    """Tab 3: where signals disagree, with a rule-based plain-language summary.

    Args:
        info: The active state.
        signals: Loaded signals.
        status: Source status table.
    """
    st.subheader("Divergences: where signals disagree")
    st.caption("Signals are compared as z-scores (each against its own previous 12 months), never as raw values. "
               f"Rules: opposite signs with both at least {config.DIVERGENCE_MIN_Z} standard deviations from zero (level), "
               "or opposite 3-month changes of at least that size (trend). Gap of "
               f"{config.DIVERGENCE_STRONG_GAP} or more is 'strong'. The summary below is generated from the numbers by "
               "fixed rules, not by an LLM.")
    z_wide = dl.wide_series(signals, "normalized_value")
    if z_wide.empty or z_wide.notna().sum().sum() == 0:
        st.info("**No data yet:** no signal has a z-score, so nothing can be compared.")
        return
    labels = dict(zip(signals["signal_id"], signals["label"]))
    cross = dv.find_divergences(z_wide, dv.cross_source_pairs(z_wide), labels)
    within = dv.find_divergences(z_wide, dv.within_retail_pairs(z_wide), labels)
    have_z = {s for s in signals.loc[signals["normalized_value"].notna(), "source"]}
    missing = []
    for _, row in status.iterrows():
        if row["source"] in (config.SOURCE_FRED_TARGETS,):
            continue
        if not row["has_data"]:
            missing.append(row["title"])
        elif row["source"] not in have_z:
            missing.append(row["title"] + " (data stored, but no z-scores yet)")
    st.markdown("#### Summary")
    for paragraph in dv.summarize(info.name, cross, within, missing):
        st.markdown(f"- {paragraph}")
    for title, result in (("Across sources", cross), ("Within retail: subsector vs total", within)):
        st.markdown(f"#### {title}")
        if result.empty:
            st.info("**No data yet** for this comparison.")
            continue
        st.dataframe(divergence_table(result), hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%+.2f") for c in ("z A", "z B", "Gap (A - B)")})
    pairs = pd.concat([within, cross], ignore_index=True) if not (within.empty and cross.empty) else within
    pairs = pairs[pairs["status"] != dv.NO_OVERLAP] if not pairs.empty else pairs
    if pairs.empty:
        return
    options = {f"{r['label_a']}  vs  {r['label_b']}": (r["signal_a"], r["signal_b"]) for _, r in pairs.iterrows()}
    pick = st.selectbox("Plot a pair (z-scores)", list(options), key="pair")
    a, b = options[pick]
    frame = z_wide[[a, b]].dropna(how="all").reset_index().melt("period_start", var_name="signal_id", value_name="y")
    frame["label"] = frame["signal_id"].map(labels)
    st.altair_chart(line_chart(frame.dropna(subset=["y"]), "z-score", zero_line=True), width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    """Build the page: header, three tabs and the data notes."""
    info = config.get_state_info()
    choice = dl.choose_database()
    path = str(choice.path)
    mtime = choice.path.stat().st_mtime if choice.path.exists() else 0.0
    signals = cached_signals(info.name, path, mtime) if choice.path.exists() else pd.DataFrame(
        columns=db.SIGNAL_COLUMNS + ["label", "geo_scope"])
    scrapes = cached_scrapes(path, mtime) if choice.path.exists() else pd.DataFrame()
    status = dl.source_status(signals, scrapes)

    render_header(info, choice, signals)
    tab_signals, tab_nowcast, tab_div = st.tabs(["1. Signals", "2. Nowcast vs official", "3. Divergences"])
    with tab_signals:
        render_signals_tab(signals, status)
    with tab_nowcast:
        render_nowcast_tab(info, signals, choice, mtime)
    with tab_div:
        render_divergence_tab(info, signals, status)

    st.divider()
    st.caption(
        "**Data notes.** Census MSRS is an experimental, modeled product (not a direct state survey) and is revised. "
        "Official FRED series are revised and only the latest vintage is stored. VIIRS is public domain. "
        + INDEED_DISCLAIMER + " Source code and method: see the README.")


main()
