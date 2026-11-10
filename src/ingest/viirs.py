"""Ingest VIIRS nighttime lights from Google Earth Engine.

Dataset: NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG (monthly average radiance, public
domain). For every metro of the active state - and for the state as a whole -
we compute the average radiance per month.

Data-quality rules (from the project spec):
    * The dataset has a quality band, ``cf_cvg``: the number of cloud-free
      nights that went into each pixel. A pixel is "low coverage" if it has
      fewer than ``config.VIIRS_LOW_COVERAGE_MIN_OBS`` such nights (or no data).
    * If more than 20% of a metro's pixels are low coverage in a month, that
      metro-month is DROPPED.
    * Missing data is NEVER imputed or interpolated. A dropped month simply
      has no row.
    * The share of good pixels is stored as ``quality_score`` (1.0 = every
      pixel well covered) so later phases can use coverage as a feature.
    * The radiance of kept months is averaged over good-coverage pixels only.

Earth Engine setup (free noncommercial "Community" tier):
    1. Sign up at https://earthengine.google.com/ and create/choose a Google
       Cloud project registered for Earth Engine.
    2. ``export EE_PROJECT=your-project-id``
    3. Either run ``earthengine authenticate`` once, or for a service account
       set ``EE_SERVICE_ACCOUNT`` and ``EE_PRIVATE_KEY_FILE``.

Run it:  python -m src.ingest.viirs
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

import config
from src import db
from src.ingest.common import IngestError, ScrapeResult, exit_code, run_logged, setup_logging
from src.normalize import add_normalized_values

logger = logging.getLogger("altsignal")

RADIANCE_UNIT = "nW/cm2/sr"


def init_earth_engine():
    """Initialise the Earth Engine client, or explain how to set it up.

    Returns:
        The imported ``ee`` module, ready to use.

    Raises:
        IngestError: If the library is missing, no project is configured, or
            authentication fails.
    """
    try:
        import ee
    except ImportError as exc:
        raise IngestError("The earthengine-api package is not installed "
                          "(pip install -r requirements.txt).") from exc

    if not config.EE_PROJECT:
        raise IngestError(
            "Earth Engine project not set. Set the EE_PROJECT environment variable "
            "to your Google Cloud project id (see the setup steps in src/ingest/viirs.py)."
        )
    try:
        account, key_file = os.environ.get("EE_SERVICE_ACCOUNT"), os.environ.get("EE_PRIVATE_KEY_FILE")
        if account and key_file:
            credentials = ee.ServiceAccountCredentials(account, key_file)
            ee.Initialize(credentials, project=config.EE_PROJECT)
        else:
            ee.Initialize(project=config.EE_PROJECT)
    except Exception as exc:  # EEException, auth errors, network errors...
        raise IngestError(
            "Earth Engine authentication/initialisation failed. Run `earthengine authenticate` "
            "or set EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc
    return ee


def get_geometry(ee, state_fips: str, county_names: list[str] | None):
    """Build the Earth Engine geometry for a state or a group of its counties.

    Args:
        ee: The initialised ``ee`` module.
        state_fips: Two-digit state FIPS code, e.g. "06".
        county_names: County names to combine, or ``None`` for the whole state.

    Returns:
        An ``ee.Geometry``.

    Raises:
        IngestError: If some county name is not found in the Census file
            (usually a typo in the lookup table in ``config.py``).
    """
    if county_names is None:
        states = ee.FeatureCollection(config.STATES_DATASET).filter(ee.Filter.eq("STATEFP", state_fips))
        return states.geometry()

    counties = (ee.FeatureCollection(config.COUNTIES_DATASET)
                .filter(ee.Filter.eq("STATEFP", state_fips))
                .filter(ee.Filter.inList("NAME", list(county_names))))
    found = counties.size().getInfo()
    if found < len(set(county_names)):
        raise IngestError(f"Only {found} of {len(set(county_names))} counties found for "
                          f"{county_names} in state FIPS {state_fips}; check config._STATE_TABLE.")
    return counties.geometry()


def fetch_monthly_stats(ee, geometry, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Compute monthly radiance and low-coverage share for one geometry.

    Args:
        ee: The initialised ``ee`` module.
        geometry: Area to summarise (``ee.Geometry``).
        start_date: First day to include, "YYYY-MM-DD".
        end_date: Day after the last day to include. Defaults to today.

    Returns:
        DataFrame with columns ``period_start``, ``avg_rad`` (mean radiance of
        good-coverage pixels, NaN if none) and ``low_coverage_share`` (0-1).
    """
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    collection = ee.ImageCollection(config.VIIRS_DATASET).filterDate(start_date, end_date)

    def summarise(image):
        """Reduce one monthly image over the geometry (runs on Google's servers)."""
        radiance = image.select(config.VIIRS_RADIANCE_BAND)
        coverage = image.select(config.VIIRS_COVERAGE_BAND)
        enough = coverage.gte(config.VIIRS_LOW_COVERAGE_MIN_OBS)
        # 1 where the pixel is low coverage; pixels with no data count as low too.
        low = enough.Not().unmask(1).rename("low")
        good_radiance = radiance.updateMask(enough).rename("rad")
        stats = low.addBands(good_radiance).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry,
            scale=config.VIIRS_SCALE_METERS, maxPixels=1e10, tileScale=4,
        )
        return ee.Feature(None, {
            "date": image.date().format("YYYY-MM-dd"),
            "rad": stats.get("rad"),
            "low": stats.get("low"),
        })

    features = collection.map(summarise).getInfo()["features"]
    return stats_from_features(features)


def stats_from_features(features: list[dict]) -> pd.DataFrame:
    """Convert Earth Engine feature dicts into a tidy DataFrame.

    Args:
        features: The ``features`` list returned by Earth Engine; each has a
            ``properties`` dict with ``date``, ``rad`` and ``low``.

    Returns:
        DataFrame with ``period_start``, ``avg_rad`` and ``low_coverage_share``
        (``None`` values become NaN).
    """
    records = [
        {
            "period_start": pd.to_datetime(f["properties"]["date"]),
            "avg_rad": f["properties"].get("rad"),
            "low_coverage_share": f["properties"].get("low"),
        }
        for f in features
    ]
    df = pd.DataFrame(records, columns=["period_start", "avg_rad", "low_coverage_share"])
    df["avg_rad"] = pd.to_numeric(df["avg_rad"], errors="coerce")
    df["low_coverage_share"] = pd.to_numeric(df["low_coverage_share"], errors="coerce")
    return df.sort_values("period_start").reset_index(drop=True)


def select_valid_months(stats: pd.DataFrame, max_low_share: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split months into kept and dropped according to the coverage rule.

    A month is dropped if more than ``max_low_share`` of the pixels are low
    coverage, or if the coverage or radiance could not be computed at all.
    Nothing is imputed.

    Args:
        stats: Output of ``fetch_monthly_stats``.
        max_low_share: Largest allowed low-coverage share (0-1). Defaults to
            ``config.VIIRS_MAX_LOW_COVERAGE_SHARE`` (0.20).

    Returns:
        ``(kept, dropped)`` DataFrames.
    """
    limit = config.VIIRS_MAX_LOW_COVERAGE_SHARE if max_low_share is None else max_low_share
    unusable = stats["avg_rad"].isna() | stats["low_coverage_share"].isna()
    too_sparse = stats["low_coverage_share"] > limit
    drop = unusable | too_sparse
    return stats[~drop].copy(), stats[drop].copy()


def build_signal_rows(kept: pd.DataFrame, geo_level: str, geo_name: str) -> pd.DataFrame:
    """Turn kept months into ``signals`` rows (without normalization).

    Args:
        kept: Months that passed ``select_valid_months``.
        geo_level: "metro" or "state".
        geo_name: Place name, e.g. "Los Angeles, CA" or "California".

    Returns:
        A DataFrame of signal rows.
    """
    return pd.DataFrame({
        "signal_id": "viirs_avg_radiance",
        "source": config.SOURCE_VIIRS,
        "geo_level": geo_level,
        "geo_name": geo_name,
        "period_start": kept["period_start"],
        "period_end": kept["period_start"] + pd.offsets.MonthEnd(0),
        "value": kept["avg_rad"],
        "unit": RADIANCE_UNIT,
        # Share of pixels with adequate coverage - the stored coverage feature.
        "quality_score": 1.0 - kept["low_coverage_share"],
        "grain": "monthly",
    })


def ingest(state: str | None = None, ee=None, start_date: str | None = None) -> ScrapeResult:
    """Fetch VIIRS for the active state, store it, and log the scrape.

    Args:
        state: State name or abbreviation. Defaults to ``config.STATE``.
        ee: Optional Earth Engine module (used by tests); by default the real
            client is initialised.
        start_date: First month to fetch. Defaults to ``config.VIIRS_START_DATE``.

    Returns:
        The ``ScrapeResult`` (also written to the ``scrapes`` table).
    """

    def job(result: ScrapeResult) -> None:
        info = config.get_state_info(state)
        client = ee or init_earth_engine()
        start = start_date or config.VIIRS_START_DATE

        # (geo_level, geo_name, counties or None for the whole state)
        areas = [("metro", m.geo_name, list(m.counties)) for m in info.metros]
        areas.append(("state", info.name, None))

        frames, errors, dropped_notes = [], {}, []
        metros_with_data = 0
        for geo_level, geo_name, counties in areas:
            try:
                geometry = get_geometry(client, info.fips, counties)
                stats = fetch_monthly_stats(client, geometry, start)
            except IngestError as exc:
                errors[geo_name] = str(exc)
                continue
            except Exception as exc:  # Earth Engine quota, timeout, ...
                errors[geo_name] = f"{type(exc).__name__}: {exc}"
                continue
            kept, dropped = select_valid_months(stats)
            if not dropped.empty:
                dropped_notes.append(f"{geo_name}: dropped {len(dropped)} of {len(stats)} months (low coverage)")
            if not kept.empty:
                frames.append(build_signal_rows(kept, geo_level, geo_name))
                if geo_level == "metro":
                    metros_with_data += 1

        result.metro_coverage = metros_with_data / len(info.metros)
        if not frames:
            first_error = next(iter(errors.values()), "no month passed the coverage rule")
            raise IngestError(f"No VIIRS rows produced. {first_error}")

        rows = add_normalized_values(pd.concat(frames, ignore_index=True))
        result.rows_returned = db.write_signals(rows)

        if errors:
            result.status = db.STATUS_PARTIAL
            result.error_message = ("Some areas failed: " + "; ".join(
                f"{name}: {msg}" for name, msg in errors.items()))[:1000]
        for note in dropped_notes:
            logger.info("[viirs] %s", note)

    return run_logged(config.SOURCE_VIIRS, job)


def main() -> int:
    """Command-line entry point: ``python -m src.ingest.viirs``.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging()
    return exit_code(ingest())


if __name__ == "__main__":
    sys.exit(main())
