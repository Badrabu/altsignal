"""The train/test split and the guard rails around it.

Spec: "Hold out the last six months as a true test set never used for
training, feature selection, or tuning."

How this module enforces that
-----------------------------
* The boundary is defined from the target's last observed month and then
  FROZEN in ``data/processed/split_<ST>.json``. Later runs reuse it, so the
  test set cannot quietly drift as new months arrive (a drifting test set would
  let earlier "test" months leak into training). Use ``--refreeze-split`` to
  define a new one on purpose.
* Every row gets a ``split`` label: ``train``, ``test``, ``later`` (target
  observed after the frozen test window), ``pre_window`` (target observed but
  before ``config.TRAIN_START``, so excluded from fitting; the row is kept, not
  deleted), ``unlabeled`` (target not published yet or missing).
* ``config.TRAIN_START`` only narrows the TRAINING window from the left. It
  never touches the test window, which stays the frozen last six months.
* Anything that LEARNS from data (scaling, feature selection, and in phase four
  the models) must take a ``TrainSet``. A ``TrainSet`` can only be built from
  rows labelled ``train`` and refuses a table that contains any test-period
  month, so test data cannot reach a fitted step by accident. The feature
  builder itself has no fitted steps at all (see ``engineer.py``).
* The test rows are available only through ``open_test_set``, which returns a
  plain table that the ``fit_*`` functions below do not accept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config

# Columns that describe a row; every other column is a feature.
META_COLUMNS = ["target_name", "target_series", "target_month", "as_of",
                "target_available", "pandemic_regime", "split", "target_value"]

SPLIT_TRAIN, SPLIT_TEST, SPLIT_LATER, SPLIT_UNLABELED = "train", "test", "later", "unlabeled"
SPLIT_PRE_WINDOW = "pre_window"

# Default for ``train_start`` arguments: read ``config.TRAIN_START`` when called.
FROM_CONFIG = "from_config"


def train_start_of(train_start=FROM_CONFIG) -> pd.Timestamp | None:
    """Resolve the first month allowed in training.

    Args:
        train_start: ``FROM_CONFIG`` (read ``config.TRAIN_START``), ``None``
            (no restriction) or a date-like value.

    Returns:
        A ``Timestamp``, or ``None`` when training may use all history.
    """
    if isinstance(train_start, str) and train_start == FROM_CONFIG:
        train_start = config.TRAIN_START
    return None if train_start is None else pd.Timestamp(train_start)


class SplitError(Exception):
    """The split cannot be defined or a rule that protects the test set was broken."""


@dataclass(frozen=True)
class SplitBoundary:
    """The frozen train/test boundary of one target.

    Attributes:
        target_name: e.g. "retail_sales".
        target_series: The signal used as target, e.g. "msrs_yoy_TOTAL".
        test_start: First month of the test set.
        test_end: Last month of the test set.
    """

    target_name: str
    target_series: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def split_path(state_abbr: str) -> Path:
    """Return the file that stores a state's frozen split boundaries.

    Args:
        state_abbr: Two-letter abbreviation.

    Returns:
        A path in ``config.PROCESSED_DIR``.
    """
    return config.PROCESSED_DIR / f"split_{state_abbr}.json"


def load_frozen(state_abbr: str) -> dict[str, SplitBoundary]:
    """Read the frozen boundaries (empty if none were defined yet).

    Args:
        state_abbr: Two-letter abbreviation.

    Returns:
        ``{target_name: SplitBoundary}``.
    """
    path = split_path(state_abbr)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: SplitBoundary(name, entry["target_series"], pd.Timestamp(entry["test_start"]),
                                pd.Timestamp(entry["test_end"]))
            for name, entry in raw.items()}


def save_frozen(state_abbr: str, boundaries: dict[str, SplitBoundary]) -> None:
    """Write the frozen boundaries to disk.

    Args:
        state_abbr: Two-letter abbreviation.
        boundaries: ``{target_name: SplitBoundary}`` to store.
    """
    path = split_path(state_abbr)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: {"target_series": b.target_series,
                      "test_start": b.test_start.strftime("%Y-%m-%d"),
                      "test_end": b.test_end.strftime("%Y-%m-%d")}
               for name, b in sorted(boundaries.items())}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def define_boundary(target_name: str, target_series: str, observed_months: pd.DatetimeIndex,
                    train_start=FROM_CONFIG) -> SplitBoundary:
    """Define a new boundary: the last ``TEST_MONTHS`` calendar months with data.

    Args:
        target_name: e.g. "retail_sales".
        target_series: Signal id of the target.
        observed_months: Months in which the target has a value.
        train_start: First month allowed in training (see ``train_start_of``);
            only months from there to the test window count toward the minimum.

    Returns:
        The new ``SplitBoundary``.

    Raises:
        SplitError: If fewer than ``MIN_TRAIN_MONTHS`` labelled months would
            remain for training.
    """
    months = pd.DatetimeIndex(sorted(set(observed_months)))
    if len(months) == 0:
        raise SplitError(f"{target_name}: the target has no observed months")
    test_end = months.max()
    test_start = test_end - pd.DateOffset(months=config.TEST_MONTHS - 1)
    start = train_start_of(train_start)
    in_window = (months < test_start) if start is None else ((months >= start) & (months < test_start))
    n_train = int(in_window.sum())
    if n_train < config.MIN_TRAIN_MONTHS:
        since = "" if start is None else f" since {start:%Y-%m}"
        raise SplitError(f"{target_name}: only {n_train} labelled months{since} before the test window; "
                         f"need at least {config.MIN_TRAIN_MONTHS}")
    return SplitBoundary(target_name, target_series, test_start, test_end)


def resolve_boundary(
    target_name: str,
    target_series: str,
    observed_months: pd.DatetimeIndex,
    frozen: dict[str, SplitBoundary],
    refreeze: bool = False,
    train_start=FROM_CONFIG,
) -> SplitBoundary:
    """Return the frozen boundary, or define (and return) a new one.

    Args:
        target_name: e.g. "retail_sales".
        target_series: Signal id of the target in use now.
        observed_months: Months in which the target has a value.
        frozen: Boundaries already frozen (``load_frozen``).
        refreeze: Define a new boundary even if one is frozen.
        train_start: First month allowed in training (only used to define a new boundary).

    Returns:
        The boundary to use.

    Raises:
        SplitError: If the frozen boundary was made for a different target
            series (for example national instead of state unemployment), or no
            valid boundary can be defined.
    """
    existing = frozen.get(target_name)
    if existing is not None and not refreeze:
        if existing.target_series != target_series:
            raise SplitError(f"{target_name}: the frozen split was defined for {existing.target_series}, "
                             f"but the target is now {target_series}. Re-run with --refreeze-split "
                             "if this change is intended.")
        return existing
    return define_boundary(target_name, target_series, observed_months, train_start)


def assign_split(target_month: pd.Timestamp, has_target: bool, boundary: SplitBoundary,
                 train_start=None) -> str:
    """Label one row as train, pre_window, test, later or unlabeled.

    Args:
        target_month: The month being nowcast.
        has_target: Whether the target value is observed.
        boundary: The frozen boundary.
        train_start: First month allowed in training (``None`` = no restriction;
            pass ``config.TRAIN_START`` to apply the configured window). It
            cannot affect the test window.

    Returns:
        One of the ``SPLIT_*`` labels.
    """
    if not has_target:
        return SPLIT_UNLABELED
    if target_month < boundary.test_start:
        start = train_start_of(train_start)
        if start is not None and target_month < start:
            return SPLIT_PRE_WINDOW
        return SPLIT_TRAIN
    if target_month <= boundary.test_end:
        return SPLIT_TEST
    return SPLIT_LATER


def feature_columns(table: pd.DataFrame) -> list[str]:
    """List the feature columns of a feature table.

    Args:
        table: A feature table.

    Returns:
        Every column that is not in ``META_COLUMNS``.
    """
    return [c for c in table.columns if c not in META_COLUMNS]


class TrainSet:
    """Rows that are allowed to be used for fitting anything.

    Construction fails unless every row is labelled ``train``, has an observed
    target and lies before the frozen test window. All fitted steps in this
    package (and the phase-four models) accept only this type.
    """

    def __init__(self, table: pd.DataFrame, boundary: SplitBoundary, train_start=FROM_CONFIG):
        """Validate and store the training rows of one target.

        Args:
            table: Feature table rows for ONE target.
            boundary: That target's frozen boundary.
            train_start: First month allowed in training (default
                ``config.TRAIN_START``; ``None`` = no restriction).

        Raises:
            SplitError: If any row is not a clean training row.
        """
        months = pd.to_datetime(table["target_month"])
        if (table["target_name"] != boundary.target_name).any():
            raise SplitError("TrainSet: rows from a different target")
        if (months >= boundary.test_start).any():
            raise SplitError("TrainSet: the table contains test-period or later months")
        start = train_start_of(train_start)
        if start is not None and (months < start).any():
            raise SplitError(f"TrainSet: the table contains months before the training start {start:%Y-%m}")
        if (table["split"] != SPLIT_TRAIN).any():
            raise SplitError("TrainSet: every row must be labelled 'train'")
        if table["target_value"].isna().any():
            raise SplitError("TrainSet: every row needs an observed target")
        self._table = table.reset_index(drop=True)
        self.boundary = boundary

    @property
    def frame(self) -> pd.DataFrame:
        """Return a copy of the training table.

        Returns:
            A DataFrame with meta and feature columns.
        """
        return self._table.copy()

    @property
    def features(self) -> pd.DataFrame:
        """Return the feature columns only (gaps stay NaN).

        Returns:
            A DataFrame of features.
        """
        return self._table[feature_columns(self._table)].copy()

    @property
    def target(self) -> pd.Series:
        """Return the target values.

        Returns:
            A series of raw target values.
        """
        return self._table["target_value"].copy()


def make_train_set(table: pd.DataFrame, target_name: str, boundary: SplitBoundary,
                   train_start=FROM_CONFIG) -> TrainSet:
    """Pick one target's training rows from a feature table.

    Args:
        table: The full feature table (all targets, all splits).
        target_name: Which target.
        boundary: That target's frozen boundary.
        train_start: First month allowed in training (default ``config.TRAIN_START``).

    Returns:
        A ``TrainSet`` with the rows labelled ``train`` and an observed target.
    """
    start = train_start_of(train_start)
    rows = table[(table["target_name"] == target_name) & (table["split"] == SPLIT_TRAIN)]
    if start is not None:
        rows = rows[pd.to_datetime(rows["target_month"]) >= start]
    return TrainSet(rows, boundary, train_start)


def open_test_set(table: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Return the held-out test rows of one target, for FINAL evaluation only.

    The result is a plain DataFrame, not a ``TrainSet``, so it cannot be passed
    to ``fit_standardizer`` or ``select_columns``.

    Args:
        table: The full feature table.
        target_name: Which target.

    Returns:
        The rows labelled ``test``.
    """
    return table[(table["target_name"] == target_name) & (table["split"] == SPLIT_TEST)].copy()


def _require_train_set(train: object) -> TrainSet:
    """Reject anything that is not a ``TrainSet``.

    Args:
        train: The object passed to a fitted step.

    Returns:
        The same object, typed as ``TrainSet``.

    Raises:
        TypeError: If it is not a ``TrainSet``.
    """
    if not isinstance(train, TrainSet):
        raise TypeError("Fitted steps only accept a TrainSet (training rows before the test window).")
    return train


@dataclass(frozen=True)
class Standardizer:
    """Column means and standard deviations learned from training rows only.

    Attributes:
        mean: Mean of each feature in the training rows.
        std: Standard deviation (ddof=1) of each feature; columns with no
            variation are left out of ``transform``'s division by NaN-ing them.
    """

    mean: pd.Series
    std: pd.Series

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features with the TRAINING mean and std.

        Missing values stay missing (nothing is filled in).

        Args:
            features: Any feature table with the same columns (train or test).

        Returns:
            The standardized features.
        """
        std = self.std.where(self.std > 0)
        return (features[self.mean.index] - self.mean) / std


def fit_standardizer(train: TrainSet) -> Standardizer:
    """Learn column means/std from the training rows.

    Args:
        train: The training rows.

    Returns:
        A ``Standardizer`` that can be applied to any table.

    Raises:
        TypeError: If ``train`` is not a ``TrainSet``.
    """
    features = _require_train_set(train).features
    return Standardizer(mean=features.mean(), std=features.std(ddof=1))


def select_columns(train: TrainSet, max_missing_share: float = 0.5) -> list[str]:
    """Choose feature columns using the training rows only.

    A column is kept if at most ``max_missing_share`` of the training rows are
    missing for it. (Nothing is imputed; this only decides which columns to use.)

    Args:
        train: The training rows.
        max_missing_share: Largest allowed share of missing training values.

    Returns:
        The kept column names, in table order.

    Raises:
        TypeError: If ``train`` is not a ``TrainSet``.
    """
    features = _require_train_set(train).features
    keep = features.isna().mean() <= max_missing_share
    return list(features.columns[keep.to_numpy()])
