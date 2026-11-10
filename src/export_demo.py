"""Export a small demo database for the Streamlit app.

    python -m src.export_demo                 # live database -> data/demo/altsignal_demo.db

The deployed app (Streamlit Community Cloud) has no credentials and cannot run
the ingestion pipelines, so it reads this committed snapshot instead. The
snapshot holds, for the active state only:

* ``signals``: state-level and national monthly rows of the sources listed in
  ``config.DEMO_EXPORT_SOURCES`` (public data only: Census MSRS, VIIRS, FRED
  official series). **Indeed is never exported** (scraped data, non-commercial
  use only) and neither is the card spending index (third-party terms).
* ``models`` and ``forecasts``: the rows of that state's models.

The ``scrapes`` table is not exported. All reading and writing goes through
``db.py``. The destination file is rebuilt from scratch every time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import config
from src import db


def export_demo(
    state: str | None = None,
    source_db: str | Path | None = None,
    dest: str | Path | None = None,
) -> dict[str, int]:
    """Copy the public part of the active state's data into a fresh demo database.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.
        source_db: Database to read. Defaults to ``config.DB_PATH``.
        dest: File to (re)create. Defaults to ``config.DEMO_DB_PATH``. It is left
            untouched when there is nothing to export.

    Returns:
        Row counts written: ``{"signals": n, "models": n, "forecasts": n}``.
    """
    info = config.get_state_info(state)
    final = Path(dest) if dest is not None else Path(config.DEMO_DB_PATH)
    # Build next to the final file and swap it in only if something was exported,
    # so a run against an empty database never wipes a good committed snapshot.
    dest = final.with_name(final.name + ".tmp")
    final.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    db.init_db(dest)

    parts = [
        db.read_signals(geo_level="state", geo_name=info.name, db_path=source_db),
        db.read_signals(geo_level="national", db_path=source_db),
    ]
    signals = pd.concat(parts, ignore_index=True)
    signals = signals[signals["source"].isin(config.DEMO_EXPORT_SOURCES) & (signals["grain"] == "monthly")]
    written_signals = db.write_signals(signals, db_path=dest)

    prefix = f"{info.abbr}_"
    models = db.read_models(db_path=source_db)
    models = models[models["model_id"].str.startswith(prefix)]
    written_models = db.write_model_metrics(models, db_path=dest)
    forecasts = db.read_forecasts(db_path=source_db)
    forecasts = forecasts[forecasts["model_id"].str.startswith(prefix)]
    written_forecasts = db.write_forecasts(forecasts.drop(columns=["forecast_id"]), db_path=dest)
    if written_signals > 0:
        dest.replace(final)
    else:
        dest.unlink()
    return {"signals": written_signals, "models": written_models, "forecasts": written_forecasts}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m src.export_demo``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        0 on success, 1 if the source database has no signals to export.
    """
    parser = argparse.ArgumentParser(description="Export the demo database used by the Streamlit app.")
    parser.add_argument("--source", help="database to read (default: config.DB_PATH)")
    parser.add_argument("--dest", help="file to write (default: config.DEMO_DB_PATH)")
    args = parser.parse_args(argv)
    counts = export_demo(source_db=args.source, dest=args.dest)
    dest = args.dest or config.DEMO_DB_PATH
    if counts["signals"] == 0:
        print(f"error: no exportable signals for {config.get_state_info().name}; "
              "run the ingest and build steps first (see README)")
        return 1
    print(f"wrote {dest}: {counts['signals']} signal rows, {counts['models']} model rows, "
          f"{counts['forecasts']} forecast rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
