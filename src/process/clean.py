"""Cleaning rules for rows of the ``signals`` table (phase two).

Cleaning here means CHECKING and DROPPING, never repairing. A row that breaks
a rule is removed and counted; a missing value is never filled in, smoothed or
interpolated.

The rules, in the order they are applied by ``clean_signals``:

1. No value (NaN/inf) or duplicate key       -> drop
2. Monthly row not aligned to a calendar month -> drop
3. Value outside the plausible range for its unit (``config.PLAUSIBLE_RANGE``)
   e.g. a negative job count                  -> drop
4. VIIRS only: more than 20% of the pixels were low coverage (or coverage is
   unknown)                                   -> drop
5. Official sources (FRED) with no quality score get 1.0 (nothing imputed).

Separately, ``check_units`` guarantees that no series mixes units. It raises
instead of dropping, because a mixed-unit series means a bug upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

KEY_COLUMNS = ["signal_id", "geo_level", "geo_name", "period_start", "grain"]


class ProcessError(Exception):
    """A data problem that must stop the build for a source (e.g. mixed units)."""


@dataclass
class CleanResult:
    """Output of ``clean_signals``.

    Attributes:
        kept: Rows that passed every rule.
        dropped: Rows that failed a rule, with an extra ``reason`` column.
    """

    kept: pd.DataFrame
    dropped: pd.DataFrame

    def reasons(self) -> dict[str, int]:
        """Count dropped rows per reason.

        Returns:
            For example ``{"implausible value": 2}`` (empty if nothing dropped).
        """
        if self.dropped.empty:
            return {}
        return self.dropped["reason"].value_counts().to_dict()


def check_units(signals: pd.DataFrame) -> None:
    """Make sure every ``signal_id`` has exactly one unit.

    Args:
        signals: Rows with ``signal_id`` and ``unit`` columns.

    Raises:
        ProcessError: If a series has no unit or more than one unit.
    """
    if signals.empty:
        return
    missing = signals.loc[signals["unit"].isna(), "signal_id"].unique()
    if len(missing):
        raise ProcessError(f"Signals without a unit: {sorted(missing)}")
    units = signals.groupby("signal_id")["unit"].nunique()
    mixed = units[units > 1]
    if not mixed.empty:
        details = {sid: sorted(signals.loc[signals["signal_id"] == sid, "unit"].unique())
                   for sid in mixed.index}
        raise ProcessError(f"Mixed units inside one signal (never allowed): {details}")


def _split(signals: pd.DataFrame, bad: pd.Series, reason: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate rows flagged ``bad`` from the rest and label the bad ones.

    Args:
        signals: The rows to split.
        bad: Boolean series (same index) that is True for rows to drop.
        reason: Text stored in the ``reason`` column of dropped rows.

    Returns:
        ``(good rows, bad rows with a reason column)``.
    """
    dropped = signals[bad].copy()
    dropped["reason"] = reason
    return signals[~bad].copy(), dropped


def _is_month_aligned(signals: pd.DataFrame) -> pd.Series:
    """Check that monthly rows start on the 1st and end on the month's last day.

    Args:
        signals: Rows with ``period_start``, ``period_end`` and ``grain``.

    Returns:
        A boolean series, True where the row is acceptable (non-monthly rows
        are not checked here).
    """
    start = pd.to_datetime(signals["period_start"])
    end = pd.to_datetime(signals["period_end"])
    aligned = (start.dt.day == 1) & (end == start + pd.offsets.MonthEnd(0))
    return aligned | (signals["grain"] != "monthly")


def _implausible(signals: pd.DataFrame) -> pd.Series:
    """Flag values outside the plausible range for their unit.

    Args:
        signals: Rows with ``value`` and ``unit``.

    Returns:
        A boolean series, True where the value is implausible. Units without a
        configured range are never flagged.
    """
    bad = pd.Series(False, index=signals.index)
    for unit, (low, high) in config.PLAUSIBLE_RANGE.items():
        in_unit = signals["unit"] == unit
        if low is not None:
            bad |= in_unit & (signals["value"] < low)
        if high is not None:
            bad |= in_unit & (signals["value"] > high)
    return bad


def clean_signals(signals: pd.DataFrame) -> CleanResult:
    """Apply every cleaning rule to a table of signal rows.

    Args:
        signals: Rows read from the ``signals`` table (or built in phase two).

    Returns:
        A ``CleanResult`` with the kept rows and the dropped rows (with reasons).
    """
    frame = signals.copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    dropped_parts = []

    def apply(bad: pd.Series, reason: str) -> None:
        """Drop the flagged rows from ``frame`` and remember them."""
        nonlocal frame
        frame, dropped = _split(frame, bad, reason)
        dropped_parts.append(dropped)

    apply(frame["value"].isna() | ~np.isfinite(frame["value"]), "missing or non-finite value")
    apply(frame.duplicated(subset=KEY_COLUMNS, keep="last"), "duplicate key")
    apply(~_is_month_aligned(frame), "monthly row not aligned to a calendar month")
    apply(_implausible(frame), "implausible value for its unit")

    # VIIRS: quality_score is the share of well-covered pixels. More than 20%
    # low coverage means a quality_score below 0.80. Unknown coverage cannot be
    # checked, so it is dropped too.
    is_viirs = frame["source"] == config.SOURCE_VIIRS
    threshold = 1.0 - config.VIIRS_MAX_LOW_COVERAGE_SHARE
    quality = frame["quality_score"]
    apply(is_viirs & (quality.isna() | (quality < threshold - 1e-9)),
          "VIIRS low coverage (more than 20% of pixels)")

    # Official sources carry no coverage information; their values are the
    # published numbers themselves, so full quality is the honest score.
    official = frame["source"].isin(config.OFFICIAL_SOURCES) & frame["quality_score"].isna()
    frame.loc[official, "quality_score"] = 1.0

    parts = [part for part in dropped_parts if not part.empty]
    dropped = pd.concat(parts, ignore_index=True) if parts else dropped_parts[0]
    return CleanResult(kept=frame.reset_index(drop=True), dropped=dropped)
