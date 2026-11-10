"""Fit the phase-four models on the TRAINING window only.

Spec: "Primary model: ridge regression with cross-validation, trained only on
the training window. Comparison models: a shallow decision tree and optionally
gradient boosting, only if ridge works."

How the test set is kept out
----------------------------
Everything that learns takes a ``TrainSet`` (``src/features/split.py``), which
cannot hold a test-period row. Column choice (``select_columns``), scaling
(``fit_standardizer``), hyperparameter tuning (cross-validation) and the final
fit all use only those rows. Test rows are touched only by ``predict``, after
the model is fixed.

Cross-validation
----------------
Time-series CV inside the training window: an expanding training part followed
by a block of validation months, repeated ``MODEL_CV_FOLDS`` times (no random
shuffling, which would let the model train on the future). A GAP of months
between the training and validation parts mimics the release lag: when month V
is nowcast, the labels of the most recent months are not public yet, so they
must not be in the training part. The scaler is re-fitted inside every fold.

Nothing is imputed. Columns that are mostly missing in training are dropped,
and training rows that still have a gap are dropped. A row to predict with a
missing feature gets NO prediction (NaN), never a filled-in one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

import config
from src.features import split as sp

RIDGE, TREE, GBM = "ridge", "tree", "gbm"


class ModelError(Exception):
    """A model cannot be fitted (for example too few complete training rows)."""


@dataclass
class FittedModel:
    """A model fitted on the training window.

    Attributes:
        name: "ridge", "tree" or "gbm".
        columns: Feature columns used, in order.
        standardizer: Scaler fitted on the training rows.
        estimator: The fitted scikit-learn model.
        params: The hyperparameters chosen by cross-validation.
        cv_rmse: Cross-validated RMSE of the chosen parameters (training window only).
        cv_predictions: Out-of-fold predictions ``{target_month: value}``.
        n_train_rows: Complete training rows used.
        n_columns_dropped: Columns left out for missing training values.
    """

    name: str
    columns: list[str]
    standardizer: sp.Standardizer
    estimator: object
    params: dict
    cv_rmse: float
    cv_predictions: pd.Series
    n_train_rows: int
    n_columns_dropped: int
    cv_grid: list[dict] = field(default_factory=list)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        """Predict rows of a feature table (train, test or unlabeled).

        Args:
            table: Feature-table rows with the model's columns and ``target_month``.

        Returns:
            Predictions indexed by ``target_month``; NaN where any feature is missing.
        """
        x = self.standardizer.transform(table)[self.columns]
        out = pd.Series(np.nan, index=pd.DatetimeIndex(pd.to_datetime(table["target_month"]), name="target_month"),
                        name=self.name)
        ok = x.notna().all(axis=1).to_numpy()
        if ok.any():
            out.iloc[np.flatnonzero(ok)] = self.estimator.predict(x.to_numpy()[ok])
        return out


def _restrict(train: sp.TrainSet, columns: list[str]) -> sp.TrainSet:
    """Build a TrainSet that holds only the given feature columns.

    The rows were already checked against ``config.TRAIN_START`` when the
    original ``TrainSet`` was built, so this one skips that check.

    Args:
        train: The training rows.
        columns: Feature columns to keep.

    Returns:
        A new ``TrainSet`` (same rows, same boundary).
    """
    frame = train.frame
    keep = [c for c in frame.columns if c in sp.META_COLUMNS or c in columns]
    return sp.TrainSet(frame[keep], train.boundary, train_start=None)


def _subset(train: sp.TrainSet, positions: np.ndarray) -> sp.TrainSet:
    """Build a TrainSet from some rows (by position) of another TrainSet.

    Args:
        train: The training rows.
        positions: Row positions to keep.

    Returns:
        A new ``TrainSet`` (still only training rows).
    """
    return sp.TrainSet(train.frame.iloc[positions], train.boundary, train_start=None)


def prepare_training_rows(train: sp.TrainSet) -> tuple[sp.TrainSet, int]:
    """Choose usable columns and complete rows, using the training rows only.

    Args:
        train: The training rows of one target.

    Returns:
        ``(complete training set restricted to the chosen columns, number of
        columns dropped)``.

    Raises:
        ModelError: If too few complete rows or usable columns remain.
    """
    columns = sp.select_columns(train, config.MODEL_MAX_MISSING_SHARE)
    n_dropped = len(sp.feature_columns(train.frame)) - len(columns)
    restricted = _restrict(train, columns)
    complete = restricted.features.notna().all(axis=1).to_numpy()
    complete_set = _subset(restricted, np.flatnonzero(complete))
    scaler = sp.fit_standardizer(complete_set)
    usable = [c for c in columns if scaler.std[c] > 0]
    if not usable:
        raise ModelError("no usable feature columns in the training window")
    needed = config.MODEL_CV_FOLDS * config.MODEL_CV_VALIDATION_MONTHS + config.MODEL_MIN_TRAIN_ROWS_FIRST_FOLD
    if complete.sum() < needed:
        raise ModelError(f"only {int(complete.sum())} complete training rows; need at least {needed}")
    return _restrict(complete_set, usable), n_dropped + (len(columns) - len(usable))


def label_gap(train: sp.TrainSet) -> int:
    """Count how many recent labels are not yet public when a month is nowcast.

    Args:
        train: Complete training rows (contiguous months).

    Returns:
        The largest number of earlier rows whose target had not been published
        at a row's prediction time.
    """
    frame = train.frame
    available = pd.to_datetime(frame["target_available"]).to_numpy()
    as_of = pd.to_datetime(frame["as_of"]).to_numpy()
    gap = 0
    for i in range(len(frame)):
        gap = max(gap, int((available[:i] > as_of[i]).sum()))
    return gap


def _make_ridge(params: dict):
    """Create a ridge regressor.

    Args:
        params: ``{"alpha": float}``.

    Returns:
        An unfitted ``Ridge``.
    """
    return Ridge(alpha=params["alpha"])


def _make_tree(params: dict):
    """Create a shallow regression tree.

    Args:
        params: ``max_depth`` and ``min_samples_leaf``.

    Returns:
        An unfitted ``DecisionTreeRegressor``.
    """
    return DecisionTreeRegressor(random_state=config.MODEL_RANDOM_STATE, **params)


def _make_gbm(params: dict):
    """Create a small gradient boosting regressor.

    Args:
        params: ``n_estimators``, ``max_depth`` and ``learning_rate``.

    Returns:
        An unfitted ``GradientBoostingRegressor``.
    """
    return GradientBoostingRegressor(subsample=0.8, random_state=config.MODEL_RANDOM_STATE, **params)


def _grid(options: dict) -> list[dict]:
    """Expand ``{"a": [1, 2], "b": [3]}`` into every combination.

    Args:
        options: Parameter name to list of values.

    Returns:
        A list of parameter dictionaries.
    """
    combos = [{}]
    for key, values in options.items():
        combos = [{**c, key: v} for c in combos for v in values]
    return combos


MODEL_TYPES: dict[str, tuple[Callable[[dict], object], list[dict]]] = {
    RIDGE: (_make_ridge, [{"alpha": a} for a in config.RIDGE_ALPHAS]),
    TREE: (_make_tree, _grid(config.TREE_GRID)),
    GBM: (_make_gbm, _grid(config.GBM_GRID)),
}


def cross_validate(train: sp.TrainSet, make: Callable[[dict], object], params: dict, gap: int) -> pd.Series:
    """Out-of-fold predictions of one parameter setting (training window only).

    Args:
        train: Complete training rows, in time order.
        make: Estimator factory.
        params: Hyperparameters.
        gap: Rows between the training and validation parts of a fold.

    Returns:
        Predictions indexed by ``target_month`` for the validation rows.
    """
    splitter = TimeSeriesSplit(n_splits=config.MODEL_CV_FOLDS, test_size=config.MODEL_CV_VALIDATION_MONTHS, gap=gap)
    months = pd.to_datetime(train.frame["target_month"]).to_numpy()
    pieces = []
    for fit_rows, val_rows in splitter.split(np.arange(len(months))):
        fold_train = _subset(train, fit_rows)
        scaler = sp.fit_standardizer(fold_train)        # re-fitted in every fold
        usable = scaler.std[scaler.std > 0].index
        x_fit = scaler.transform(fold_train.features)[usable].to_numpy()
        x_val = scaler.transform(train.features.iloc[val_rows])[usable].to_numpy()
        model = make(params).fit(x_fit, fold_train.target.to_numpy())
        pieces.append(pd.Series(model.predict(x_val), index=pd.DatetimeIndex(months[val_rows])))
    return pd.concat(pieces)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error.

    Args:
        actual: Observed values.
        predicted: Predicted values.

    Returns:
        The RMSE.
    """
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def fit_model(name: str, train: sp.TrainSet, n_columns_dropped: int = 0) -> FittedModel:
    """Tune a model by time-series CV and fit it on the whole training window.

    Args:
        name: "ridge", "tree" or "gbm".
        train: Complete training rows from ``prepare_training_rows``.
        n_columns_dropped: Columns left out earlier (for the report).

    Returns:
        The fitted model with its chosen hyperparameters and CV score.

    Raises:
        TypeError: If ``train`` is not a ``TrainSet``.
    """
    if not isinstance(train, sp.TrainSet):
        raise TypeError("fit_model only accepts a TrainSet")
    make, grid = MODEL_TYPES[name]
    gap = label_gap(train)
    actual = train.target.set_axis(pd.DatetimeIndex(pd.to_datetime(train.frame["target_month"])))

    results = []
    for params in grid:
        oof = cross_validate(train, make, params, gap)
        results.append({"params": params, "cv_rmse": rmse(actual.loc[oof.index], oof), "oof": oof})
    best = min(results, key=lambda r: r["cv_rmse"])      # first one wins a tie (simplest settings come first)

    scaler = sp.fit_standardizer(train)
    columns = [c for c in train.features.columns if scaler.std[c] > 0]
    model = make(best["params"]).fit(scaler.transform(train.features)[columns].to_numpy(), train.target.to_numpy())
    return FittedModel(
        name=name, columns=columns, standardizer=scaler, estimator=model, params=best["params"],
        cv_rmse=best["cv_rmse"], cv_predictions=best["oof"], n_train_rows=len(train.frame),
        n_columns_dropped=n_columns_dropped,
        cv_grid=[{"params": r["params"], "cv_rmse": r["cv_rmse"]} for r in results])
