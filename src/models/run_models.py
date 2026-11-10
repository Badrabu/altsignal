"""Phase four entry point: benchmarks and models for the active state.

    python -m src.models.run_models

Run ``python -m src.features.build_features`` first. For every target in the
feature table (retail sales; unemployment once FRED data exists) this

1. computes the naive benchmarks (``benchmarks.py``),
2. fits ridge regression with time-series cross-validation on the TRAINING
   window only, then a shallow decision tree, and gradient boosting only if
   ridge "works" (it beats the best naive benchmark in cross-validation inside
   the training window; the test set plays no part in that decision),
3. scores every model on the frozen test set (MAE and RMSE) and writes
   ``models`` and ``forecasts`` rows via ``db.py``,
4. saves a JSON report to ``data/processed/models_<ST>.json``.

Idempotent: the same features give the same results, and rerunning updates the
existing database rows (unique on model_id + metric, and on model_id + target
period) instead of adding new ones.

``forecasts`` holds only out-of-sample rows: the test months, any later
labelled months and the unlabeled months still to be nowcast (``actual_value``
empty). ``error`` = actual - predicted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

import config
from src import db
from src.features import split as sp
from src.features.build_features import features_path
from src.models import benchmarks, fit

logger = logging.getLogger("altsignal")

OUT_OF_SAMPLE = (sp.SPLIT_TEST, sp.SPLIT_LATER, sp.SPLIT_UNLABELED)


def model_id(state_abbr: str, target: str, name: str) -> str:
    """Build the id under which a model is stored.

    Args:
        state_abbr: Two-letter abbreviation.
        target: Target name, e.g. "retail_sales".
        name: Model name, e.g. "ridge" or "naive_locf".

    Returns:
        For example "CA_retail_sales_ridge".
    """
    return f"{state_abbr}_{target}_{name}"


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute error.

    Args:
        actual: Observed values.
        predicted: Predictions on the same index.

    Returns:
        The MAE.
    """
    return float(np.mean(np.abs(actual - predicted)))


def score(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE and RMSE.

    Args:
        actual: Observed values.
        predicted: Predictions on the same index.

    Returns:
        ``{"MAE": ..., "RMSE": ...}``.
    """
    return {"MAE": mae(actual, predicted), "RMSE": fit.rmse(actual.to_numpy(), predicted.to_numpy())}


def ridge_beats_naive_in_cv(ridge: fit.FittedModel, naive: pd.DataFrame, actual: pd.Series) -> tuple[bool, dict]:
    """Decide whether ridge "works", using cross-validation inside the training window.

    Ridge works if its out-of-fold RMSE is lower than that of the best naive
    benchmark on the same validation months (months where the benchmark exists).

    Args:
        ridge: The fitted ridge model.
        naive: Benchmark predictions by target month.
        actual: Observed targets by target month.

    Returns:
        ``(works, details)``.
    """
    months = ridge.cv_predictions.index
    rows = naive.loc[months].dropna()
    ridge_rmse = fit.rmse(actual.loc[rows.index].to_numpy(), ridge.cv_predictions.loc[rows.index].to_numpy())
    naive_rmse = {c: fit.rmse(actual.loc[rows.index].to_numpy(), rows[c].to_numpy()) for c in rows.columns}
    best = min(naive_rmse.values()) if naive_rmse else float("nan")
    works = bool(len(rows) > 0 and ridge_rmse < best)
    return works, {"validation_months": int(len(rows)), "ridge_cv_rmse": ridge_rmse,
                   "naive_cv_rmse": naive_rmse, "ridge_works": works}


def run_target(table: pd.DataFrame, target: str, boundary: sp.SplitBoundary, state_abbr: str) -> dict:
    """Run benchmarks and models for one target.

    Args:
        table: The full feature table.
        target: Target name.
        boundary: That target's frozen split boundary.
        state_abbr: Two-letter abbreviation.

    Returns:
        A report dictionary (also used to write the database rows).
    """
    rows = table[table["target_name"] == target].sort_values("target_month").reset_index(drop=True)
    rows["target_month"] = pd.to_datetime(rows["target_month"])
    by_month = rows.set_index("target_month")
    actual = by_month["target_value"]

    naive = benchmarks.all_benchmarks(rows)
    predictions: dict[str, pd.Series] = {name: naive[name] for name in naive.columns}
    report: dict = {"target": target, "target_series": boundary.target_series,
                    "train_start": None, "train_end": None,
                    "test_start": boundary.test_start.strftime("%Y-%m-%d"),
                    "test_end": boundary.test_end.strftime("%Y-%m-%d"), "models": {}, "notes": []}

    train_all = sp.make_train_set(rows, target, boundary)
    report["train_start"] = pd.to_datetime(train_all.frame["target_month"]).min().strftime("%Y-%m-%d")
    report["train_end"] = pd.to_datetime(train_all.frame["target_month"]).max().strftime("%Y-%m-%d")
    fitted: dict[str, fit.FittedModel] = {}
    try:
        train, dropped = fit.prepare_training_rows(train_all)
    except fit.ModelError as exc:
        report["notes"].append(f"models not fitted: {exc}")
        train = None
    if train is not None:
        report["training"] = {"complete_rows": len(train.frame), "columns_used": len(train.features.columns),
                              "columns_dropped": dropped,
                              "first_month": pd.to_datetime(train.frame["target_month"]).min().strftime("%Y-%m-%d"),
                              "label_gap_months": fit.label_gap(train)}
        fitted[fit.RIDGE] = fit.fit_model(fit.RIDGE, train, dropped)
        works, details = ridge_beats_naive_in_cv(fitted[fit.RIDGE], naive, actual)
        report["ridge_gate"] = details
        fitted[fit.TREE] = fit.fit_model(fit.TREE, train, dropped)
        if works or not config.GRADIENT_BOOSTING_REQUIRES_RIDGE_CV_BEATS_NAIVE:
            fitted[fit.GBM] = fit.fit_model(fit.GBM, train, dropped)
        else:
            report["notes"].append("gradient boosting not run: ridge did not beat the best naive "
                                   "benchmark in cross-validation inside the training window")
        for name, model in fitted.items():
            predictions[name] = model.predict(rows)

    # Evaluate every model on the SAME test months: those with an observed
    # target and a prediction from every model and benchmark.
    test_months = rows.loc[rows["split"] == sp.SPLIT_TEST, "target_month"]
    frame = pd.DataFrame(predictions).loc[test_months]
    frame.insert(0, "actual", actual.loc[test_months])
    common = frame.dropna()
    report["test_months_total"] = int(len(test_months))
    report["test_months_scored"] = int(len(common))
    if len(common) < len(test_months):
        report["notes"].append(f"{len(test_months) - len(common)} test month(s) not scored: "
                               "a model had a missing feature (nothing is imputed)")
    report["test_errors"] = {m.strftime("%Y-%m"): {c: float(common.loc[m, "actual"] - common.loc[m, c])
                                                   for c in common.columns if c != "actual"}
                             for m in common.index}

    for name, series in predictions.items():
        metrics = score(common["actual"], common[name]) if len(common) else {"MAE": None, "RMSE": None}
        entry = {"kind": "naive" if name in benchmarks.BENCHMARKS else name, **metrics}
        if name in fitted:
            model = fitted[name]
            entry.update({"params": model.params, "cv_rmse": model.cv_rmse, "n_train_rows": model.n_train_rows})
        report["models"][name] = entry

    report["verdict"] = verdict(report)
    report["_predictions"] = predictions            # stripped before the JSON is written
    report["_actual"] = actual
    report["_split"] = rows.set_index("target_month")["split"]
    return report


def verdict(report: dict) -> dict:
    """Compare each fitted model with the best naive benchmark on the test set.

    A model "meaningfully beats naive" if both its RMSE and its MAE are at
    least ``config.MEANINGFUL_MARGIN`` lower than the best naive benchmark's.

    Args:
        report: The report produced by ``run_target`` (before the verdict).

    Returns:
        ``{"best_naive": name, "margin_required": x, "models": {name: {...}},
        "any_model_meaningfully_beats_naive": bool}``.
    """
    models = report["models"]
    naive = {n: m for n, m in models.items() if m["kind"] == "naive" and m["RMSE"] is not None}
    if not naive:
        return {"any_model_meaningfully_beats_naive": False, "note": "no scored benchmark"}
    best_naive = min(naive, key=lambda n: naive[n]["RMSE"])
    out = {"best_naive": best_naive, "margin_required": config.MEANINGFUL_MARGIN, "models": {}}
    for name, m in models.items():
        if m["kind"] == "naive" or m["RMSE"] is None:
            continue
        rmse_gain = 1 - m["RMSE"] / naive[best_naive]["RMSE"]
        mae_gain = 1 - m["MAE"] / naive[best_naive]["MAE"]
        out["models"][name] = {"rmse_improvement": rmse_gain, "mae_improvement": mae_gain,
                               "meaningfully_beats_naive": bool(min(rmse_gain, mae_gain) >= config.MEANINGFUL_MARGIN)}
    out["any_model_meaningfully_beats_naive"] = any(v["meaningfully_beats_naive"] for v in out["models"].values())
    return out


def store_results(report: dict, state_abbr: str) -> tuple[int, int]:
    """Write one target's metrics and out-of-sample forecasts to the database.

    Args:
        report: The report from ``run_target``.
        state_abbr: Two-letter abbreviation.

    Returns:
        ``(metric rows written, forecast rows written)``.
    """
    db.init_db()
    target = report["target"]
    metric_rows, forecast_rows = [], []
    for name, entry in report["models"].items():
        mid = model_id(state_abbr, target, name)
        base = {"model_id": mid, "model_type": entry["kind"],
                "target": target, "test_start": report["test_start"], "test_end": report["test_end"]}
        if entry["kind"] != "naive":
            base.update({"train_start": report["train_start"], "train_end": report["train_end"]})
        metrics = {"MAE": entry["MAE"], "RMSE": entry["RMSE"]}
        if "cv_rmse" in entry:
            metrics["CV_RMSE"] = entry["cv_rmse"]
        metric_rows += [{**base, "metric_name": k, "metric_value": v} for k, v in metrics.items()]

    db.write_model_metrics(metric_rows)
    for name, series in report["_predictions"].items():
        mid = model_id(state_abbr, target, name)
        for month, value in series.items():
            if report["_split"].get(month) not in OUT_OF_SAMPLE or pd.isna(value):
                continue
            actual = report["_actual"].get(month)
            has_actual = actual is not None and pd.notna(actual)
            forecast_rows.append({"model_id": mid, "target_period": month, "predicted_value": float(value),
                                  "actual_value": float(actual) if has_actual else None,
                                  "error": float(actual - value) if has_actual else None})
    db.write_forecasts(forecast_rows)
    return len(metric_rows), len(forecast_rows)


def format_report(report: dict) -> str:
    """Render a target's test-set results as a plain-text table.

    Args:
        report: The report from ``run_target``.

    Returns:
        A multi-line string.
    """
    lines = [f"  test window {report['test_start'][:7]} to {report['test_end'][:7]}: "
             f"{report['test_months_scored']} of {report['test_months_total']} months scored",
             f"  {'model':16s} {'MAE':>9s} {'RMSE':>9s}  notes"]
    for name, m in report["models"].items():
        extra = f"params {m['params']}, CV RMSE {m['cv_rmse']:.3f}" if "params" in m else ""
        mae_text = f"{m['MAE']:9.3f}" if m["MAE"] is not None else f"{'n/a':>9s}"
        rmse_text = f"{m['RMSE']:9.3f}" if m["RMSE"] is not None else f"{'n/a':>9s}"
        lines.append(f"  {name:16s} {mae_text} {rmse_text}  {extra}")
    v = report["verdict"]
    if "best_naive" in v:
        for name, d in v["models"].items():
            lines.append(f"  {name} vs {v['best_naive']}: RMSE {d['rmse_improvement']:+.1%}, MAE "
                         f"{d['mae_improvement']:+.1%} -> "
                         f"{'meaningfully better' if d['meaningfully_beats_naive'] else 'NOT meaningfully better'}"
                         f" (need {v['margin_required']:.0%})")
    lines += [f"  note: {n}" for n in report["notes"]]
    return "\n".join(lines)


def run_models(state: str | None = None) -> dict[str, dict]:
    """Run benchmarks and models for every target in the saved feature table.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.

    Returns:
        ``{target name: report}``; targets that cannot be modelled are omitted.

    Raises:
        FileNotFoundError: If the feature table has not been built.
    """
    info = config.get_state_info(state)
    path = features_path(info.abbr)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.features.build_features` first")
    table = pd.read_csv(path, parse_dates=["target_month", "as_of", "target_available"])
    frozen = sp.load_frozen(info.abbr)
    db.init_db()

    reports = {}
    for target in sorted(table["target_name"].unique()):
        if target not in frozen:
            logger.warning("no frozen split for %s; skipped", target)
            continue
        reports[target] = run_target(table, target, frozen[target], info.abbr)
        store_results(reports[target], info.abbr)
    return reports


def save_json(reports: dict[str, dict], state_abbr: str) -> None:
    """Save the reports (without raw prediction series) as JSON.

    Args:
        reports: ``{target: report}``.
        state_abbr: Two-letter abbreviation.
    """
    clean = {t: {k: v for k, v in r.items() if not k.startswith("_")} for t, r in reports.items()}
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / f"models_{state_abbr}.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m src.models.run_models``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        0 if at least one target was modelled, otherwise 1.
    """
    argparse.ArgumentParser(description="Run the AltSignal benchmarks and models.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    info = config.get_state_info()
    print(f"Running models for {info.name} ({info.abbr})")
    try:
        reports = run_models()
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    table = pd.read_csv(features_path(info.abbr), usecols=["target_name"])
    for target in config.FEATURE_TARGETS:
        if target not in reports:
            status = "no data in the feature table" if target not in set(table["target_name"]) else "not modelled"
            print(f"target {target}: {status}")
    for target, report in reports.items():
        print(f"target {target} ({report['target_series']}):")
        print(format_report(report))
    save_json(reports, info.abbr)
    print(f"wrote {config.PROCESSED_DIR / f'models_{info.abbr}.json'}")
    return 0 if reports else 1


if __name__ == "__main__":
    sys.exit(main())
