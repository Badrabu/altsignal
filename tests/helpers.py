"""Helpers that create synthetic test signals (clearly fake test data, used only in tests)."""

import numpy as np
import pandas as pd

from src import db
from src.normalize import add_normalized_values


def series_rows(signal_id, source, unit, geo_level, geo_name, n_months, start="2019-01-01",
                seed=0, quality=None):
    """Build ``n_months`` of monthly signal rows with random-walk values and real z-scores.

    Args:
        signal_id: Series name.
        source: Source name.
        unit: Unit label.
        geo_level: "state" or "national".
        geo_name: Place name.
        n_months: Number of months.
        start: First month.
        seed: Random seed, so tests are repeatable.
        quality: Optional constant quality score.

    Returns:
        A DataFrame of signal rows (``normalized_value`` computed like phase two).
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="MS")
    rows = pd.DataFrame({
        "signal_id": signal_id, "source": source, "geo_level": geo_level, "geo_name": geo_name,
        "period_start": months, "period_end": months + pd.offsets.MonthEnd(0),
        "value": 5 + np.cumsum(rng.normal(0, 1, n_months)), "unit": unit,
        "quality_score": quality, "grain": "monthly"})
    return add_normalized_values(rows)


def write_series(*args, **kwargs):
    """Create a synthetic series and store it in the ``signals`` table.

    Returns:
        The rows that were written.
    """
    db.init_db()
    rows = series_rows(*args, **kwargs)
    db.write_signals(rows)
    return rows
