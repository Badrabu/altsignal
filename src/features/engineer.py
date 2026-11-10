"""Turn the ``signals`` table into one feature table per target (phase three).

The central idea is the PREDICTION TIME. A nowcast for target month T is made
on ``as_of(T)`` = last day of T + ``config.PREDICTION_OFFSET_DAYS``. Every
observation has a release lag (``config.RELEASE_LAG_DAYS``): it becomes public
``lag`` days after its period ends. For each series and each T we therefore
use only the LATEST PERIOD P* that was already published at ``as_of(T)``, and
everything derived from P* looks backwards from P*:

    <series>__z              normalized value (z-score) at P*
    <series>__lag1/2/3       the z-score 1, 2, 3 months before P*
    <series>__chg1/3         change in z over 1 and 3 months up to P*
    <series>__roll3/6_mean   rolling mean of z over the 3/6 months up to P*
    <series>__roll3/6_std    rolling standard deviation over the same window
    <series>__coverage_pct   (VIIRS only) share of well-covered pixels x 100

Rules:
* No look-ahead. Nothing after P* is touched, and P* is chosen by release date,
  not by the calendar (so MSRS, published ~3.5 months late, enters with that
  delay, and a target's own value for T is never one of its features).
* No imputation. A feature is NaN if its value, or any value in its lag/change/
  rolling window, is missing. We never carry an older value forward to fill a
  hole, and a window needs ALL its months. Downstream code drops or flags such
  rows.
* There are no fitted steps here (no scaling, no selection), so there is
  nothing that could learn from the test period. Fitted steps live in
  ``split.py`` and only accept training rows.

Features are built on the NORMALIZED series only, so different units are never
mixed. The raw target value is kept as the label (``target_value``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

import config
from src import db
from src.features import split as splitting
from src.process.build_signals import in_scope

# Separator between a series name and its feature suffix.
SEP = "__"


def month_end(month: pd.Timestamp) -> pd.Timestamp:
    """Return the last day of a month.

    Args:
        month: Any day in the month.

    Returns:
        A ``Timestamp`` for the month's last day.
    """
    return (month + pd.offsets.MonthEnd(0)).normalize()


def release_lag(signal_id: str, source: str) -> int | None:
    """Look up how many days after its period a series is published.

    Args:
        signal_id: Series name (checked first).
        source: Source name (checked second).

    Returns:
        The lag in days, or ``None`` if none is configured (the series is then
        not used, because a guessed lag could leak the future).
    """
    if signal_id in config.RELEASE_LAG_DAYS:
        return config.RELEASE_LAG_DAYS[signal_id]
    return config.RELEASE_LAG_DAYS.get(source)


def as_of(target_month: pd.Timestamp) -> pd.Timestamp:
    """Return the prediction time for a target month.

    Args:
        target_month: First day of the month being nowcast.

    Returns:
        Last day of the month plus ``config.PREDICTION_OFFSET_DAYS``.
    """
    return month_end(target_month) + pd.Timedelta(days=config.PREDICTION_OFFSET_DAYS)


def latest_published_period(target_month: pd.Timestamp, lag_days: int) -> pd.Timestamp:
    """Find P*: the latest period already published at the prediction time.

    Args:
        target_month: First day of the month being nowcast.
        lag_days: The series' release lag in days.

    Returns:
        The first day of the latest month whose end plus ``lag_days`` is not
        after ``as_of(target_month)``.
    """
    deadline = as_of(target_month)
    period = target_month
    while month_end(period) + pd.Timedelta(days=lag_days) > deadline:
        period -= pd.DateOffset(months=1)
    return period


def derive_series_features(z: pd.Series, coverage_pct: pd.Series | None = None) -> pd.DataFrame:
    """Compute lags, changes and rolling statistics for one series.

    Every value at period P uses only periods up to P.

    Args:
        z: Normalized values indexed by month start. Missing months may be absent.
        coverage_pct: Optional coverage percentage (same index).

    Returns:
        A DataFrame on a complete monthly calendar, one column per feature
        suffix (``z``, ``lag1``, ..., ``coverage_pct``). Gaps stay NaN.
    """
    calendar = pd.date_range(z.index.min(), z.index.max(), freq="MS")
    z = z.reindex(calendar)
    columns = {"z": z}
    for k in config.FEATURE_LAGS:
        columns[f"lag{k}"] = z.shift(k)
    for k in config.FEATURE_CHANGES:
        columns[f"chg{k}"] = z - z.shift(k)
    for window in config.FEATURE_ROLLING_WINDOWS:
        rolling = z.rolling(window=window, min_periods=window)   # needs every month
        columns[f"roll{window}_mean"] = rolling.mean()
        columns[f"roll{window}_std"] = rolling.std()
    if coverage_pct is not None:
        columns["coverage_pct"] = coverage_pct.reindex(calendar)
    return pd.DataFrame(columns)


def series_name(signal_id: str, geo_level: str) -> str:
    """Build the column prefix of a series.

    Args:
        signal_id: e.g. "msrs_yoy_TOTAL".
        geo_level: e.g. "state".

    Returns:
        e.g. "msrs_yoy_TOTAL.state".
    """
    return f"{signal_id}.{geo_level}"


@dataclass
class FeatureBuild:
    """Everything ``build_feature_table`` produces.

    Attributes:
        table: One row per (target, target month), meta columns then features.
        boundaries: The frozen split boundary of each built target.
        series_used: ``{series name: release lag in days}`` of every series that became features.
        series_skipped: ``{series name: reason}`` for series left out.
        targets: ``{target name: status text}`` ("built: <series>", "no data", or an error).
    """

    table: pd.DataFrame
    boundaries: dict[str, splitting.SplitBoundary] = field(default_factory=dict)
    series_used: dict[str, int] = field(default_factory=dict)
    series_skipped: dict[str, str] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)


def select_feature_series(signals: pd.DataFrame, series_skipped: dict[str, str]) -> pd.DataFrame:
    """Keep the monthly rows at feature geography levels with a known release lag.

    Args:
        signals: Signal rows for the active state (plus national).
        series_skipped: Filled in with series left out, and why.

    Returns:
        The rows that may become features.
    """
    rows = signals[(signals["grain"] == "monthly") & signals["geo_level"].isin(config.FEATURE_GEO_LEVELS)].copy()
    rows["lag_days"] = [release_lag(sid, src) for sid, src in zip(rows["signal_id"], rows["source"])]
    unknown = rows[rows["lag_days"].isna()]
    for sid, level in unknown[["signal_id", "geo_level"]].drop_duplicates().itertuples(index=False):
        series_skipped[series_name(sid, level)] = "no release lag configured"
    return rows[rows["lag_days"].notna()]


def target_months_range(first_month: pd.Timestamp, today: date) -> pd.DatetimeIndex:
    """List the target months to build rows for.

    Args:
        first_month: Earliest month (first month the target is observed).
        today: The current date; a month is included once its prediction time has passed.

    Returns:
        Month starts from ``first_month`` to the last month whose ``as_of`` is
        not after ``today``.
    """
    months = pd.date_range(first_month, pd.Timestamp(today) + pd.offsets.MonthBegin(0), freq="MS")
    return pd.DatetimeIndex([m for m in months if as_of(m) <= pd.Timestamp(today)])


def build_feature_table(
    signals: pd.DataFrame,
    today: date,
    frozen: dict[str, splitting.SplitBoundary] | None = None,
    refreeze: bool = False,
) -> FeatureBuild:
    """Build the feature table for every target that has data.

    Args:
        signals: Signal rows for the active state (plus national); the ``signals``
            table columns including ``normalized_value`` and ``quality_score``.
        today: Current date; later prediction times are not built.
        frozen: Already frozen split boundaries (``None`` = none yet).
        refreeze: Define new split boundaries even if some are frozen.

    Returns:
        A ``FeatureBuild``. Targets without data are listed in ``targets`` and
        have no rows; a target without enough history for a split reports the
        reason there too.
    """
    build = FeatureBuild(table=pd.DataFrame())
    frozen = frozen or {}
    usable = select_feature_series(signals, build.series_skipped)
    usable = usable.assign(period_start=pd.to_datetime(usable["period_start"]))

    # 1. Per-series feature frames, each on its own calendar.
    per_series: dict[str, tuple[pd.DataFrame, int]] = {}
    for (sid, level, geo), group in usable.groupby(["signal_id", "geo_level", "geo_name"]):
        group = group.set_index("period_start").sort_index()
        coverage = None
        if group["source"].iloc[0] in config.COVERAGE_FEATURE_SOURCES:
            coverage = group["quality_score"] * 100
        name = series_name(sid, level)
        per_series[name] = (derive_series_features(group["normalized_value"], coverage),
                            int(group["lag_days"].iloc[0]))
        build.series_used[name] = int(group["lag_days"].iloc[0])

    tables = []
    for target_name, candidates in config.FEATURE_TARGETS.items():
        chosen = _pick_target(usable, signals, candidates)
        if chosen is None:
            build.targets[target_name] = "no data"
            continue
        sid, level, values, lag_days = chosen
        observed = values.dropna()
        if observed.empty:
            build.targets[target_name] = "no data"
            continue
        try:
            boundary = splitting.resolve_boundary(target_name, sid, observed.index, frozen, refreeze)
        except splitting.SplitError as exc:
            build.targets[target_name] = f"error: {exc}"
            continue
        build.boundaries[target_name] = boundary

        months = target_months_range(observed.index.min(), today)
        rows = pd.DataFrame({"target_month": months})
        rows.insert(0, "target_name", target_name)
        rows.insert(1, "target_series", sid)
        rows["as_of"] = [as_of(m) for m in months]
        rows["target_available"] = [month_end(m) + pd.Timedelta(days=lag_days) for m in months]
        rows["target_value"] = values.reindex(months).to_numpy()
        rows["split"] = [splitting.assign_split(m, pd.notna(v), boundary)
                         for m, v in zip(months, rows["target_value"])]
        tables.append(_attach_features(rows, per_series))
        build.targets[target_name] = f"built: {sid}"

    if tables:
        build.table = pd.concat(tables, ignore_index=True)
    return build


def _pick_target(usable: pd.DataFrame, signals: pd.DataFrame, candidates: list[tuple[str, str]]):
    """Choose the first candidate target series that has data.

    Args:
        usable: Feature-eligible signal rows (not used for targets' values).
        signals: All signal rows for the state.
        candidates: ``[(signal_id, geo_level), ...]`` in order of preference.

    Returns:
        ``(signal_id, geo_level, raw values by month, release lag)`` for the
        first candidate with at least one value, or ``None``.
    """
    monthly = signals[signals["grain"] == "monthly"]
    for sid, level in candidates:
        rows = monthly[(monthly["signal_id"] == sid) & (monthly["geo_level"] == level)]
        if rows.empty:
            continue
        lag = release_lag(sid, rows["source"].iloc[0])
        if lag is None:
            continue
        values = rows.assign(period_start=pd.to_datetime(rows["period_start"])) \
                     .set_index("period_start")["value"].sort_index()
        if values.notna().any():
            return sid, level, values, lag
    return None


def _attach_features(rows: pd.DataFrame, per_series: dict[str, tuple[pd.DataFrame, int]]) -> pd.DataFrame:
    """Add every series' features, taken at its latest published period P*.

    Args:
        rows: Meta columns, one row per target month.
        per_series: ``{series name: (feature frame, release lag)}``.

    Returns:
        ``rows`` with feature columns appended (gaps are NaN).
    """
    pieces = [rows]
    for name, (features, lag_days) in sorted(per_series.items()):
        periods = [latest_published_period(m, lag_days) for m in rows["target_month"]]
        picked = features.reindex(periods)
        picked.columns = [f"{name}{SEP}{c}" for c in picked.columns]
        picked.index = rows.index
        pieces.append(picked)
    return pd.concat(pieces, axis=1)


def load_state_signals(info: config.StateInfo) -> pd.DataFrame:
    """Read the active state's rows (plus national) from the ``signals`` table.

    Args:
        info: The active state's lookup information.

    Returns:
        The ``signals`` rows in scope.
    """
    db.init_db()
    return in_scope(db.read_signals(), info)
