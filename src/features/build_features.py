"""Phase three entry point: build the feature table for the active state.

    python -m src.features.build_features
    python -m src.features.build_features --refreeze-split    # define a new test window

Reads the ``signals`` table (run ``python -m src.process.build_signals`` first)
and writes, in ``data/processed/``:

    features_<ST>.csv        one row per target and target month (see engineer.py);
                             missing values are empty cells, never filled
    features_<ST>_meta.json  what was built: lags used, split boundaries, counts
    split_<ST>.json          the FROZEN train/test boundary per target

Re-running with unchanged inputs rewrites identical files (idempotent), and the
frozen test window never moves unless ``--refreeze-split`` is given.

Targets: ``retail_sales`` (MSRS total retail, state) and ``unemployment``
(state, with the national rate as fallback). A target without data yet (for
example unemployment before FRED credentials exist) is reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import config
from src.features import engineer, split as splitting

logger = logging.getLogger("altsignal")


def features_path(state_abbr: str):
    """Return the CSV path of a state's feature table.

    Args:
        state_abbr: Two-letter abbreviation.

    Returns:
        A path in ``config.PROCESSED_DIR``.
    """
    return config.PROCESSED_DIR / f"features_{state_abbr}.csv"


def meta_path(state_abbr: str):
    """Return the JSON path describing a state's feature table.

    Args:
        state_abbr: Two-letter abbreviation.

    Returns:
        A path in ``config.PROCESSED_DIR``.
    """
    return config.PROCESSED_DIR / f"features_{state_abbr}_meta.json"


def describe(build: engineer.FeatureBuild, info: config.StateInfo, today) -> dict:
    """Summarise a feature build as a JSON-friendly dictionary.

    Args:
        build: The result of ``engineer.build_feature_table``.
        info: The active state's lookup information.
        today: The date the table was built for.

    Returns:
        A dictionary with settings, targets, boundaries, row counts and the
        number of non-missing values per feature.
    """
    table = build.table
    features = splitting.feature_columns(table) if not table.empty else []
    return {
        "state": info.name,
        "built_for_date": today.isoformat(),
        "prediction_offset_days": config.PREDICTION_OFFSET_DAYS,
        "test_months": config.TEST_MONTHS,
        "targets": build.targets,
        "split_boundaries": {
            name: {"target_series": b.target_series,
                   "test_start": b.test_start.strftime("%Y-%m-%d"),
                   "test_end": b.test_end.strftime("%Y-%m-%d")}
            for name, b in sorted(build.boundaries.items())},
        "rows_per_target_and_split": (
            table.groupby(["target_name", "split"]).size().unstack(fill_value=0).to_dict("index")
            if not table.empty else {}),
        "release_lag_days_used": dict(sorted(build.series_used.items())),
        "series_skipped": dict(sorted(build.series_skipped.items())),
        "non_missing_per_feature": ({c: int(table[c].notna().sum()) for c in features}),
    }


def build_features(state: str | None = None, today=None, refreeze: bool = False,
                   write: bool = True) -> tuple[engineer.FeatureBuild, dict]:
    """Build (and optionally save) the feature table for one state.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.
        today: Date to build for. Defaults to today (UTC).
        refreeze: Define new split boundaries instead of reusing frozen ones.
        write: Save the CSV/JSON files and freeze any new boundaries.

    Returns:
        ``(build, description)``.
    """
    info = config.get_state_info(state)
    today = today or datetime.now(timezone.utc).date()
    signals = engineer.load_state_signals(info)
    frozen = splitting.load_frozen(info.abbr)
    build = engineer.build_feature_table(signals, today, frozen=frozen, refreeze=refreeze)
    description = describe(build, info, today)

    if write:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if build.boundaries:
            splitting.save_frozen(info.abbr, {**frozen, **build.boundaries})
        build.table.to_csv(features_path(info.abbr), index=False, date_format="%Y-%m-%d")
        meta_path(info.abbr).write_text(json.dumps(description, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return build, description


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m src.features.build_features``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        0 if at least one target was built, otherwise 1.
    """
    parser = argparse.ArgumentParser(description="Build the AltSignal feature table.")
    parser.add_argument("--refreeze-split", action="store_true",
                        help="define new train/test boundaries (the old ones are frozen otherwise)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    info = config.get_state_info()
    print(f"Building features for {info.name} ({info.abbr})")
    build, description = build_features(refreeze=args.refreeze_split)

    for name, status in build.targets.items():
        print(f"  target {name:14s} {status}")
    for name, boundary in build.boundaries.items():
        counts = description["rows_per_target_and_split"].get(name, {})
        print(f"  {name}: test window {boundary.test_start:%Y-%m} to {boundary.test_end:%Y-%m}; rows {counts}")
    if build.series_skipped:
        print(f"  series skipped: {build.series_skipped}")
    if not build.table.empty:
        print(f"  wrote {features_path(info.abbr)} ({len(build.table)} rows, "
              f"{len(splitting.feature_columns(build.table))} features)")
    return 0 if not build.table.empty else 1


if __name__ == "__main__":
    sys.exit(main())
