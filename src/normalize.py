"""Normalization of signals into z-scores.

Signals have different units (percent, index points, counts, radiance), so we
never compare raw values across signals. Instead each value is also stored as
a z-score against that signal's OWN trailing baseline:

    z_t = (x_t - mean of previous N periods) / std of previous N periods

where N is 12 for monthly series (12 months), 52 for weekly series and 4 for
quarterly series (all about 12 months).

Rules (from the project spec):
    * The baseline uses only PAST periods (not the current one), so a z-score
      never uses information from the future.
    * Missing periods are never filled in. If any of the N baseline periods is
      missing, the z-score is left empty (None).
"""

from __future__ import annotations

import pandas as pd

import config

# pandas frequency codes for each grain. Monthly periods start on the 1st;
# weekly periods start on Monday; quarterly on the 1st of Jan/Apr/Jul/Oct.
_FREQ = {"monthly": "MS", "weekly": "W-MON", "quarterly": "QS"}


def rolling_zscore(values: pd.Series, grain: str) -> pd.Series:
    """Compute trailing z-scores for one series.

    Args:
        values: A series indexed by period start date (``DatetimeIndex``),
            holding raw values. Missing periods may simply be absent.
        grain: "monthly", "weekly" or "quarterly". Sets the baseline length.

    Returns:
        A series with the same index as ``values``. Entries are NaN where
        there is not a complete baseline or the baseline has no variation.

    Raises:
        ValueError: If ``grain`` is not supported.
    """
    if grain not in config.ZSCORE_WINDOW:
        raise ValueError(f"Unsupported grain {grain!r}; use one of {list(config.ZSCORE_WINDOW)}")
    window = config.ZSCORE_WINDOW[grain]

    series = values.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]

    # Put the series on a complete calendar so a gap shows up as NaN. The NaN
    # is NOT filled; it just stops that window from producing a z-score.
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=_FREQ[grain])
    on_calendar = series.reindex(full_index)

    baseline = on_calendar.shift(1).rolling(window=window, min_periods=window)
    std = baseline.std()
    z = (on_calendar - baseline.mean()) / std.where(std > 0)
    return z.reindex(series.index)


def add_normalized_values(signals: pd.DataFrame) -> pd.DataFrame:
    """Fill the ``normalized_value`` column for a table of signal rows.

    Each (signal_id, geo_level, geo_name, grain) series is normalized against
    its own history, so different signals are never mixed.

    Args:
        signals: Rows with at least signal_id, geo_level, geo_name, grain,
            period_start and value.

    Returns:
        A copy of ``signals`` with ``normalized_value`` set.
    """
    result = signals.copy()
    result["period_start"] = pd.to_datetime(result["period_start"])
    result["normalized_value"] = float("nan")
    keys = ["signal_id", "geo_level", "geo_name", "grain"]
    for _, group in result.groupby(keys):
        grain = group["grain"].iloc[0]
        series = group.set_index("period_start")["value"].astype(float)
        z = rolling_zscore(series, grain)
        # Write back by row position, so duplicate dates cannot misalign.
        lookup = z.to_dict()
        result.loc[group.index, "normalized_value"] = group["period_start"].map(lookup)
    return result
