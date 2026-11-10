"""Naive benchmarks (phase four), defined before any model result exists.

Spec: "Naive benchmarks defined before any results are seen: last observation
carried forward, and year-over-year seasonal naive."

Both benchmarks are NOWCASTS in the same sense as the models: for target month
T they may only use target values that had already been published at the
prediction time ``as_of(T)``. A value for month M is public from
``target_available(M)`` (end of M plus the series' release lag), so a
benchmark never sees a number the real forecaster would not have had.

Both functions take the feature table rows of ONE target (columns
``target_month``, ``as_of``, ``target_available`` and ``target_value``, as
written by phase three) and return one prediction per target month, or NaN
when the benchmark cannot be computed. Nothing is fitted, so there is nothing
that could learn from the test period.
"""

from __future__ import annotations

import pandas as pd

import config

LOCF = "naive_locf"
SEASONAL = "naive_seasonal"

_REQUIRED = ["target_month", "as_of", "target_available", "target_value"]


def _check(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the input and return it sorted with parsed dates.

    Args:
        table: Rows of one target.

    Returns:
        A copy sorted by ``target_month`` with date columns as timestamps.

    Raises:
        ValueError: If a required column is missing or the table mixes targets.
    """
    missing = [c for c in _REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"benchmark input is missing columns {missing}")
    if "target_name" in table.columns and table["target_name"].nunique() > 1:
        raise ValueError("benchmark input must hold exactly one target")
    out = table.copy()
    for col in ("target_month", "as_of", "target_available"):
        out[col] = pd.to_datetime(out[col])
    return out.sort_values("target_month").reset_index(drop=True)


def last_observation_carried_forward(table: pd.DataFrame) -> pd.Series:
    """Predict each month with the latest value already published at ``as_of``.

    Args:
        table: Rows of one target (see the module docstring).

    Returns:
        Predictions indexed by ``target_month``; NaN if nothing was published yet.
    """
    rows = _check(table)
    observed = rows[rows["target_value"].notna()]
    predictions = []
    for as_of in rows["as_of"]:
        published = observed[observed["target_available"] <= as_of]
        predictions.append(float(published["target_value"].iloc[-1]) if not published.empty else float("nan"))
    return pd.Series(predictions, index=pd.DatetimeIndex(rows["target_month"], name="target_month"),
                     name=LOCF, dtype=float)


def seasonal_naive(table: pd.DataFrame, lag_months: int | None = None) -> pd.Series:
    """Predict each month with the target's own value one year earlier.

    The earlier value is used only if it was published by ``as_of``.

    Args:
        table: Rows of one target (see the module docstring).
        lag_months: Months to look back. Defaults to
            ``config.NAIVE_SEASONAL_LAG_MONTHS`` (12).

    Returns:
        Predictions indexed by ``target_month``; NaN if the earlier month has
        no published value.
    """
    lag = config.NAIVE_SEASONAL_LAG_MONTHS if lag_months is None else lag_months
    rows = _check(table)
    by_month = rows.set_index("target_month")
    predictions = []
    for month, as_of in zip(rows["target_month"], rows["as_of"]):
        earlier = month - pd.DateOffset(months=lag)
        value = float("nan")
        if earlier in by_month.index:
            entry = by_month.loc[earlier]
            if pd.notna(entry["target_value"]) and entry["target_available"] <= as_of:
                value = float(entry["target_value"])
        predictions.append(value)
    return pd.Series(predictions, index=pd.DatetimeIndex(rows["target_month"], name="target_month"),
                     name=SEASONAL, dtype=float)


# name -> function; run_models evaluates every entry the same way as the models.
BENCHMARKS = {
    LOCF: last_observation_carried_forward,
    SEASONAL: seasonal_naive,
}


def all_benchmarks(table: pd.DataFrame) -> pd.DataFrame:
    """Compute every benchmark for one target.

    Args:
        table: Rows of one target.

    Returns:
        A DataFrame indexed by ``target_month`` with one column per benchmark.
    """
    return pd.DataFrame({name: fn(table) for name, fn in BENCHMARKS.items()})
