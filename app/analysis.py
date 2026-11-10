"""Plain-Python rules behind tabs 1 and 2 of the app (no Streamlit, no models).

Every statement the app makes about a trend or about model quality is produced
here from the numbers by fixed, deterministic rules, so it can be unit-tested.
No LLM is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

RISING, FALLING, FLAT, UNKNOWN = "rising", "falling", "flat", "not enough data"
TREND_SYMBOLS = {RISING: "up", FALLING: "down", FLAT: "flat", UNKNOWN: "n/a"}
NAIVE_KIND = "naive"


def classify_trend(
    series: pd.Series,
    window: int | None = None,
    min_ratio: float | None = None,
) -> dict:
    """Classify the current trend of a monthly series.

    Rule: compare the mean of the last ``window`` months with the mean of the
    ``window`` months before them. The difference, divided by the standard
    deviation of the whole series, is the ``ratio``. A ratio of at least
    ``min_ratio`` is "rising", at most minus that is "falling", anything between
    is "flat". The ``2 * window`` most recent calendar months must all be
    present (a gap means the trend is "not enough data"; nothing is filled in).

    Args:
        series: Values indexed by month start (``NaN`` = missing).
        window: Months per window. Defaults to ``config.TREND_WINDOW_MONTHS``.
        min_ratio: Threshold. Defaults to ``config.TREND_MIN_RATIO``.

    Returns:
        ``{"direction", "delta", "ratio", "recent_mean", "prior_mean"}``; the
        numbers are ``None`` when the direction is "not enough data".
    """
    window = config.TREND_WINDOW_MONTHS if window is None else window
    min_ratio = config.TREND_MIN_RATIO if min_ratio is None else min_ratio
    empty = {"direction": UNKNOWN, "delta": None, "ratio": None, "recent_mean": None, "prior_mean": None}
    values = series.dropna().sort_index()
    if values.empty:
        return empty
    last = values.index.max()
    months = pd.date_range(end=last, periods=2 * window, freq="MS")
    recent_window = values.reindex(months)
    if recent_window.isna().any() or len(values) < 2 * window:
        return empty
    prior_mean = float(recent_window.iloc[:window].mean())
    recent_mean = float(recent_window.iloc[window:].mean())
    delta = recent_mean - prior_mean
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if not np.isfinite(spread) or spread == 0:
        ratio = 0.0 if delta == 0 else float(np.sign(delta) * np.inf)
    else:
        ratio = delta / spread
    direction = RISING if ratio >= min_ratio else FALLING if ratio <= -min_ratio else FLAT
    return {"direction": direction, "delta": delta, "ratio": ratio,
            "recent_mean": recent_mean, "prior_mean": prior_mean}


def summarize_signals(signals: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Build the tab 1 table: latest reading and current trend of every signal.

    Args:
        signals: Rows from ``data_loading.load_signals``.
        value_col: ``"value"`` (raw) or ``"normalized_value"`` (z-score). The
            latest reading and the trend are both computed on this column, so
            raw values of different units are never compared with each other.

    Returns:
        One row per series with ``signal_id``, ``label``, ``source``,
        ``geo_scope``, ``unit``, ``latest_period``, ``latest``, ``previous``,
        ``change``, ``trend``, ``n_points`` and ``quality``. Series with no
        value in ``value_col`` get ``latest_period`` empty and trend
        "not enough data". Empty input gives an empty table.
    """
    columns = ["signal_id", "label", "source", "geo_scope", "unit", "latest_period", "latest",
               "previous", "change", "trend", "n_points", "quality"]
    rows = []
    for (signal_id, geo_level), part in signals.groupby(["signal_id", "geo_level"], sort=False):
        part = part.sort_values("period_start")
        values = part.set_index("period_start")[value_col]
        known = values.dropna()
        trend = classify_trend(values)
        latest_row = part[part["period_start"] == known.index.max()].iloc[0] if not known.empty else None
        rows.append({
            "signal_id": signal_id, "label": part["label"].iloc[0], "source": part["source"].iloc[0],
            "geo_scope": part["geo_scope"].iloc[0],
            "unit": "z-score" if value_col == "normalized_value" else part["unit"].iloc[0],
            "latest_period": known.index.max() if not known.empty else pd.NaT,
            "latest": float(known.iloc[-1]) if not known.empty else None,
            "previous": float(known.iloc[-2]) if len(known) > 1 else None,
            "change": float(known.iloc[-1] - known.iloc[-2]) if len(known) > 1 else None,
            "trend": trend["direction"], "n_points": int(len(known)),
            "quality": float(latest_row["quality_score"]) if latest_row is not None and pd.notna(latest_row["quality_score"]) else None,
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows, columns=columns)
    # Headline series first (the total), then the rest alphabetically by label.
    table["_order"] = (~table["signal_id"].str.endswith("_TOTAL")).astype(int)
    return table.sort_values(["source", "_order", "label"]).drop(columns="_order").reset_index(drop=True)


def _scored_months(row: pd.Series, forecasts: pd.DataFrame) -> set:
    """Test-window months with an official value that this model forecast.

    Args:
        row: A model's metadata row (``model_id``, ``test_start``, ``test_end``).
        forecasts: The models' forecasts.

    Returns:
        A set of month timestamps.
    """
    mine = forecasts[forecasts["model_id"] == row["model_id"]] if not forecasts.empty else forecasts
    if mine.empty or pd.isna(row["test_start"]):
        return set()
    in_window = (mine["target_period"] >= pd.to_datetime(row["test_start"])) & \
                (mine["target_period"] <= pd.to_datetime(row["test_end"]))
    return set(mine.loc[in_window & mine["actual_value"].notna(), "target_period"])


def build_scoreboard(models: pd.DataFrame, forecasts: pd.DataFrame, margin: float | None = None) -> pd.DataFrame:
    """Compare every model with the best naive benchmark on the test set.

    The benchmark is the naive method with the lowest test RMSE. A model
    "clears the margin" if both its MAE and its RMSE are at least ``margin``
    lower than that benchmark's (same rule as phase four).

    Args:
        models: Metric rows from ``data_loading.load_models`` (one target).
        forecasts: Rows from ``data_loading.load_forecasts`` for the same models.
        margin: Required improvement. Defaults to ``config.MEANINGFUL_MARGIN``.

    Returns:
        One row per model with ``model_id``, ``kind``, ``MAE``, ``RMSE``,
        ``CV_RMSE`` (fitted models only), ``test_months`` (scored forecasts
        inside the test window), ``rmse_vs_naive``, ``mae_vs_naive`` (share
        lower than the best naive; negative = worse; empty for naive rows) and
        ``clears_margin``. Empty if there are no metrics.
    """
    margin = config.MEANINGFUL_MARGIN if margin is None else margin
    columns = ["model_id", "kind", "MAE", "RMSE", "CV_RMSE", "test_months", "rmse_vs_naive",
               "mae_vs_naive", "clears_margin", "train_start", "train_end", "test_start", "test_end"]
    if models.empty:
        return pd.DataFrame(columns=columns)
    wide = models.pivot_table(index="model_id", columns="metric_name", values="metric_value", aggfunc="first")
    meta = models.groupby("model_id").first()[["model_type", "train_start", "train_end", "test_start", "test_end"]]
    table = wide.join(meta).reset_index().rename(columns={"model_type": "kind"})
    for col in ("MAE", "RMSE", "CV_RMSE"):
        if col not in table:
            table[col] = np.nan
    month_sets = [_scored_months(row, forecasts) for _, row in table.iterrows()]
    # Phase four scores every model on the SAME months: those every model could forecast.
    non_empty = [m for m in month_sets if m]
    common = set.intersection(*non_empty) if non_empty else set()
    table["test_months"] = len(common)
    table.attrs["own_test_months"] = {row["model_id"]: len(m) for (_, row), m in zip(table.iterrows(), month_sets)}

    naive = table[(table["kind"] == NAIVE_KIND) & table["RMSE"].notna()]
    table["rmse_vs_naive"] = np.nan
    table["mae_vs_naive"] = np.nan
    table["clears_margin"] = pd.Series([None] * len(table), dtype=object)
    if not naive.empty:
        best = naive.loc[naive["RMSE"].idxmin()]
        fitted = table["kind"] != NAIVE_KIND
        table.loc[fitted, "rmse_vs_naive"] = 1 - table.loc[fitted, "RMSE"] / best["RMSE"]
        table.loc[fitted, "mae_vs_naive"] = 1 - table.loc[fitted, "MAE"] / best["MAE"]
        for i in table.index[fitted]:
            gains = (table.at[i, "rmse_vs_naive"], table.at[i, "mae_vs_naive"])
            table.at[i, "clears_margin"] = bool(not any(pd.isna(g) for g in gains) and min(gains) >= margin)
    table["_order"] = (table["kind"] != NAIVE_KIND).astype(int)
    return table.sort_values(["_order", "RMSE"]).drop(columns="_order").reset_index(drop=True)[columns]


def best_naive_id(scoreboard: pd.DataFrame) -> str | None:
    """Return the id of the naive benchmark with the lowest test RMSE.

    Args:
        scoreboard: The table from ``build_scoreboard``.

    Returns:
        A model id, or ``None`` if no naive benchmark was scored.
    """
    naive = scoreboard[(scoreboard["kind"] == NAIVE_KIND) & scoreboard["RMSE"].notna()]
    return None if naive.empty else str(naive.loc[naive["RMSE"].idxmin(), "model_id"])


def honesty_notes(scoreboard: pd.DataFrame, forecasts: pd.DataFrame, short_names: dict[str, str]) -> list[str]:
    """Write the fact-based caveats shown under the model comparison.

    Each note is produced by a fixed rule from the scoreboard and the forecasts.

    Args:
        scoreboard: The table from ``build_scoreboard``.
        forecasts: The models' forecasts (all rows).
        short_names: ``{model_id: readable name}``.

    Returns:
        A list of sentences (empty if there is nothing to say).
    """
    notes: list[str] = []
    if scoreboard.empty:
        return notes
    fitted = scoreboard[scoreboard["kind"] != NAIVE_KIND]
    n_test = int(scoreboard["test_months"].max()) if len(scoreboard) else 0
    if n_test:
        notes.append(f"The test window has only {n_test} scored months, so any ranking of models can flip easily.")
    own = scoreboard.attrs.get("own_test_months", {})
    if own and max(own.values()) > n_test:
        notes.append(f"Every model is scored on the same {n_test} months. Some models forecast up to {max(own.values())} "
                     "test months, but at least one model had a missing feature in the others (nothing is imputed), "
                     "so those months are left out for all models.")
    best_naive = best_naive_id(scoreboard)
    if fitted.empty:
        notes.append("No ridge, tree or boosting model has test scores for this target: none could be fitted under the "
                     "pre-set rules (too few complete training rows), so only the naive benchmarks are scored. "
                     "See the README and blockers.md.")
    for _, row in fitted.iterrows():
        name = short_names.get(row["model_id"], row["model_id"])
        if pd.isna(row["rmse_vs_naive"]):
            continue
        verdict = "clears" if row["clears_margin"] else "does NOT clear"
        notes.append(
            f"{name}: RMSE {row['rmse_vs_naive']:+.0%} and MAE {row['mae_vs_naive']:+.0%} against the best naive "
            f"benchmark ({short_names.get(best_naive, best_naive)}); it {verdict} the pre-set "
            f"{config.MEANINGFUL_MARGIN:.0%} margin (both must be at least that much lower).")
        if pd.notna(row["CV_RMSE"]) and pd.notna(row["RMSE"]) and row["CV_RMSE"] > row["RMSE"] * 1.5:
            notes.append(f"{name}: its error in cross-validation inside the training window ({row['CV_RMSE']:.2f}) was "
                         f"much higher than its test-set RMSE ({row['RMSE']:.2f}), so the test result was not predicted "
                         "by the training evidence.")
    if not forecasts.empty:
        scored = forecasts[forecasts["error"].notna()]
        for _, row in scoreboard.iterrows():
            mine = scored[scored["model_id"] == row["model_id"]]
            if row["kind"] == NAIVE_KIND or mine.empty or pd.isna(row["test_start"]):
                continue
            mine = mine[(mine["target_period"] >= pd.to_datetime(row["test_start"])) &
                        (mine["target_period"] <= pd.to_datetime(row["test_end"]))]
            name = short_names.get(row["model_id"], row["model_id"])
            if len(mine) >= 3 and (mine["error"] > 0).all():
                notes.append(f"{name}: every test-month error has the same sign (it under-predicts every month).")
            elif len(mine) >= 3 and (mine["error"] < 0).all():
                notes.append(f"{name}: every test-month error has the same sign (it over-predicts every month).")
            distinct = mine["predicted_value"].round(2).nunique()
            if len(mine) >= 3 and distinct <= 2:
                notes.append(f"{name}: it predicts only {distinct} distinct value(s) over the {len(mine)} test months, "
                             "close to a constant forecast.")
    return notes
