"""AltSignal configuration.

This is the ONLY file you need to edit to switch the pilot state.

    STATE = "California"      ->   STATE = "Texas"

Every module reads the active state from here (through ``get_state_info``),
so changing the single ``STATE`` value and re-running the pipelines is enough
to collect data for a different state.

The file has four parts:
    1. The active state (``STATE``).
    2. A lookup table of all 50 states + DC (abbreviation, FIPS code, metros).
    3. Small helper functions/classes to read that table.
    4. Settings for each data source (paths, URLs, thresholds).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. THE ACTIVE STATE
# ---------------------------------------------------------------------------
# Full state name (e.g. "California") or two-letter abbreviation (e.g. "CA").
STATE = "California"

# ---------------------------------------------------------------------------
# 2. STATE LOOKUP TABLE
# ---------------------------------------------------------------------------
# Format:  "State name": ("ABBR", "FIPS", [metros])
# Each metro is:  ("Metro name", ["County", "County", ...])
#
#   * Metro name  -> used as the geo_name in the database and, together with
#                    the state abbreviation, as the Indeed search location
#                    (e.g. "Los Angeles, CA").
#   * Counties    -> the principal counties of the metro (names exactly as in
#                    the Census TIGER file). VIIRS nighttime lights are
#                    averaged over these counties.
#
# The metro lists are deliberately short: the largest 2-4 metro areas per
# state, built from the principal counties of each Census metropolitan
# statistical area. They are an approximation, not the official MSA
# definition. Add or edit entries freely; nothing else needs to change.
_STATE_TABLE: dict[str, tuple[str, str, list[tuple[str, list[str]]]]] = {
    "Alabama": ("AL", "01", [
        ("Birmingham", ["Jefferson", "Shelby"]),
        ("Huntsville", ["Madison", "Limestone"]),
        ("Mobile", ["Mobile"]),
        ("Montgomery", ["Montgomery"]),
    ]),
    "Alaska": ("AK", "02", [
        ("Anchorage", ["Anchorage"]),
        ("Fairbanks", ["Fairbanks North Star"]),
    ]),
    "Arizona": ("AZ", "04", [
        ("Phoenix", ["Maricopa", "Pinal"]),
        ("Tucson", ["Pima"]),
    ]),
    "Arkansas": ("AR", "05", [
        ("Little Rock", ["Pulaski", "Saline"]),
        ("Fayetteville", ["Washington", "Benton"]),
    ]),
    "California": ("CA", "06", [
        ("Los Angeles", ["Los Angeles", "Orange"]),
        ("San Francisco", ["San Francisco", "Alameda", "Contra Costa", "San Mateo", "Marin"]),
        ("San Diego", ["San Diego"]),
        ("San Jose", ["Santa Clara", "San Benito"]),
        ("Sacramento", ["Sacramento", "Placer", "El Dorado", "Yolo"]),
        ("Riverside", ["Riverside", "San Bernardino"]),
        ("Fresno", ["Fresno"]),
    ]),
    "Colorado": ("CO", "08", [
        ("Denver", ["Denver", "Adams", "Arapahoe", "Jefferson", "Douglas"]),
        ("Colorado Springs", ["El Paso"]),
        ("Fort Collins", ["Larimer"]),
    ]),
    "Connecticut": ("CT", "09", [
        ("Hartford", ["Hartford", "Tolland", "Middlesex"]),
        ("New Haven", ["New Haven"]),
        ("Bridgeport", ["Fairfield"]),
    ]),
    "Delaware": ("DE", "10", [
        ("Wilmington", ["New Castle"]),
        ("Dover", ["Kent"]),
    ]),
    "District of Columbia": ("DC", "11", [
        ("Washington", ["District of Columbia"]),
    ]),
    "Florida": ("FL", "12", [
        ("Miami", ["Miami-Dade", "Broward", "Palm Beach"]),
        ("Tampa", ["Hillsborough", "Pinellas", "Pasco", "Hernando"]),
        ("Orlando", ["Orange", "Seminole", "Osceola", "Lake"]),
        ("Jacksonville", ["Duval", "Clay", "St. Johns", "Nassau", "Baker"]),
    ]),
    "Georgia": ("GA", "13", [
        ("Atlanta", ["Fulton", "DeKalb", "Cobb", "Gwinnett", "Clayton", "Cherokee", "Forsyth", "Henry"]),
        ("Savannah", ["Chatham"]),
        ("Augusta", ["Richmond", "Columbia"]),
    ]),
    "Hawaii": ("HI", "15", [
        ("Honolulu", ["Honolulu"]),
        ("Kahului", ["Maui"]),
    ]),
    "Idaho": ("ID", "16", [
        ("Boise", ["Ada", "Canyon"]),
        ("Coeur d'Alene", ["Kootenai"]),
    ]),
    "Illinois": ("IL", "17", [
        ("Chicago", ["Cook", "DuPage", "Lake", "Will", "Kane", "McHenry"]),
        ("Rockford", ["Winnebago"]),
        ("Peoria", ["Peoria", "Tazewell"]),
    ]),
    "Indiana": ("IN", "18", [
        ("Indianapolis", ["Marion", "Hamilton", "Hendricks", "Johnson", "Hancock"]),
        ("Fort Wayne", ["Allen"]),
        ("Evansville", ["Vanderburgh"]),
    ]),
    "Iowa": ("IA", "19", [
        ("Des Moines", ["Polk", "Dallas", "Warren"]),
        ("Cedar Rapids", ["Linn"]),
        ("Davenport", ["Scott"]),
    ]),
    "Kansas": ("KS", "20", [
        ("Wichita", ["Sedgwick"]),
        ("Topeka", ["Shawnee"]),
    ]),
    "Kentucky": ("KY", "21", [
        ("Louisville", ["Jefferson", "Bullitt", "Oldham"]),
        ("Lexington", ["Fayette", "Jessamine"]),
    ]),
    "Louisiana": ("LA", "22", [
        ("New Orleans", ["Orleans", "Jefferson", "St. Tammany", "St. Bernard"]),
        ("Baton Rouge", ["East Baton Rouge", "Livingston"]),
        ("Shreveport", ["Caddo", "Bossier"]),
    ]),
    "Maine": ("ME", "23", [
        ("Portland", ["Cumberland", "York", "Sagadahoc"]),
        ("Bangor", ["Penobscot"]),
    ]),
    "Maryland": ("MD", "24", [
        ("Baltimore", ["Baltimore", "Anne Arundel", "Howard", "Harford"]),
        ("Rockville", ["Montgomery", "Prince George's"]),
        ("Frederick", ["Frederick"]),
    ]),
    "Massachusetts": ("MA", "25", [
        ("Boston", ["Suffolk", "Middlesex", "Norfolk", "Essex", "Plymouth"]),
        ("Worcester", ["Worcester"]),
        ("Springfield", ["Hampden"]),
    ]),
    "Michigan": ("MI", "26", [
        ("Detroit", ["Wayne", "Oakland", "Macomb"]),
        ("Grand Rapids", ["Kent", "Ottawa"]),
        ("Lansing", ["Ingham"]),
    ]),
    "Minnesota": ("MN", "27", [
        ("Minneapolis", ["Hennepin", "Ramsey", "Dakota", "Anoka", "Washington", "Scott", "Carver"]),
        ("Rochester", ["Olmsted"]),
        ("Duluth", ["St. Louis"]),
    ]),
    "Mississippi": ("MS", "28", [
        ("Jackson", ["Hinds", "Rankin", "Madison"]),
        ("Gulfport", ["Harrison"]),
    ]),
    "Missouri": ("MO", "29", [
        ("Kansas City", ["Jackson", "Clay", "Platte", "Cass"]),
        ("St. Louis", ["St. Louis", "St. Charles", "Jefferson"]),
        ("Springfield", ["Greene"]),
    ]),
    "Montana": ("MT", "30", [
        ("Billings", ["Yellowstone"]),
        ("Missoula", ["Missoula"]),
    ]),
    "Nebraska": ("NE", "31", [
        ("Omaha", ["Douglas", "Sarpy"]),
        ("Lincoln", ["Lancaster"]),
    ]),
    "Nevada": ("NV", "32", [
        ("Las Vegas", ["Clark"]),
        ("Reno", ["Washoe"]),
    ]),
    "New Hampshire": ("NH", "33", [
        ("Manchester", ["Hillsborough"]),
        ("Portsmouth", ["Rockingham"]),
    ]),
    "New Jersey": ("NJ", "34", [
        ("Newark", ["Essex", "Union", "Morris", "Passaic"]),
        ("Jersey City", ["Hudson", "Bergen"]),
        ("Trenton", ["Mercer"]),
        ("Camden", ["Camden", "Burlington"]),
    ]),
    "New Mexico": ("NM", "35", [
        ("Albuquerque", ["Bernalillo", "Sandoval"]),
        ("Santa Fe", ["Santa Fe"]),
    ]),
    "New York": ("NY", "36", [
        ("New York", ["New York", "Kings", "Queens", "Bronx", "Richmond"]),
        ("Long Island", ["Nassau", "Suffolk"]),
        ("Buffalo", ["Erie", "Niagara"]),
        ("Rochester", ["Monroe"]),
        ("Albany", ["Albany", "Rensselaer", "Saratoga", "Schenectady"]),
    ]),
    "North Carolina": ("NC", "37", [
        ("Charlotte", ["Mecklenburg", "Union", "Cabarrus", "Gaston"]),
        ("Raleigh", ["Wake", "Johnston"]),
        ("Greensboro", ["Guilford"]),
        ("Durham", ["Durham", "Orange"]),
    ]),
    "North Dakota": ("ND", "38", [
        ("Fargo", ["Cass"]),
        ("Bismarck", ["Burleigh", "Morton"]),
    ]),
    "Ohio": ("OH", "39", [
        ("Columbus", ["Franklin", "Delaware", "Licking"]),
        ("Cleveland", ["Cuyahoga", "Lake", "Lorain", "Geauga", "Medina"]),
        ("Cincinnati", ["Hamilton", "Butler", "Warren", "Clermont"]),
    ]),
    "Oklahoma": ("OK", "40", [
        ("Oklahoma City", ["Oklahoma", "Cleveland", "Canadian"]),
        ("Tulsa", ["Tulsa", "Rogers", "Wagoner"]),
    ]),
    "Oregon": ("OR", "41", [
        ("Portland", ["Multnomah", "Washington", "Clackamas"]),
        ("Eugene", ["Lane"]),
        ("Salem", ["Marion", "Polk"]),
    ]),
    "Pennsylvania": ("PA", "42", [
        ("Philadelphia", ["Philadelphia", "Montgomery", "Bucks", "Delaware", "Chester"]),
        ("Pittsburgh", ["Allegheny", "Washington", "Westmoreland", "Butler", "Beaver"]),
        ("Harrisburg", ["Dauphin", "Cumberland", "Perry"]),
    ]),
    "Rhode Island": ("RI", "44", [
        ("Providence", ["Providence", "Kent", "Bristol", "Washington", "Newport"]),
    ]),
    "South Carolina": ("SC", "45", [
        ("Charleston", ["Charleston", "Berkeley", "Dorchester"]),
        ("Columbia", ["Richland", "Lexington"]),
        ("Greenville", ["Greenville", "Pickens"]),
    ]),
    "South Dakota": ("SD", "46", [
        ("Sioux Falls", ["Minnehaha", "Lincoln"]),
        ("Rapid City", ["Pennington"]),
    ]),
    "Tennessee": ("TN", "47", [
        ("Nashville", ["Davidson", "Williamson", "Rutherford", "Wilson"]),
        ("Memphis", ["Shelby"]),
        ("Knoxville", ["Knox"]),
        ("Chattanooga", ["Hamilton"]),
    ]),
    "Texas": ("TX", "48", [
        ("Houston", ["Harris", "Fort Bend", "Montgomery", "Brazoria", "Galveston"]),
        ("Dallas", ["Dallas", "Tarrant", "Collin", "Denton"]),
        ("Austin", ["Travis", "Williamson", "Hays"]),
        ("San Antonio", ["Bexar", "Comal"]),
        ("El Paso", ["El Paso"]),
    ]),
    "Utah": ("UT", "49", [
        ("Salt Lake City", ["Salt Lake", "Tooele"]),
        ("Provo", ["Utah"]),
        ("Ogden", ["Weber", "Davis"]),
    ]),
    "Vermont": ("VT", "50", [
        ("Burlington", ["Chittenden", "Franklin"]),
    ]),
    "Virginia": ("VA", "51", [
        ("Virginia Beach", ["Virginia Beach", "Norfolk", "Chesapeake", "Newport News", "Hampton"]),
        ("Richmond", ["Richmond", "Henrico", "Chesterfield", "Hanover"]),
        ("Arlington", ["Fairfax", "Arlington", "Loudoun", "Prince William", "Alexandria"]),
    ]),
    "Washington": ("WA", "53", [
        ("Seattle", ["King", "Snohomish", "Pierce"]),
        ("Spokane", ["Spokane"]),
    ]),
    "West Virginia": ("WV", "54", [
        ("Charleston", ["Kanawha"]),
        ("Huntington", ["Cabell"]),
        ("Morgantown", ["Monongalia"]),
    ]),
    "Wisconsin": ("WI", "55", [
        ("Milwaukee", ["Milwaukee", "Waukesha", "Washington", "Ozaukee"]),
        ("Madison", ["Dane"]),
        ("Green Bay", ["Brown"]),
    ]),
    "Wyoming": ("WY", "56", [
        ("Cheyenne", ["Laramie"]),
        ("Casper", ["Natrona"]),
    ]),
}


# ---------------------------------------------------------------------------
# 3. HELPERS TO READ THE LOOKUP TABLE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Metro:
    """One metro area inside a state.

    Attributes:
        name: Metro name, e.g. "Los Angeles". Used as ``geo_name`` in the DB.
        counties: Principal county names used for the VIIRS aggregation.
        indeed_location: Search text for Indeed, e.g. "Los Angeles, CA".
        geo_name: Name stored in the ``signals`` table, e.g. "Los Angeles, CA".
            The state is included because metro names repeat across states
            (Springfield MA/MO, Portland ME/OR, ...).
    """

    name: str
    counties: tuple[str, ...]
    indeed_location: str
    geo_name: str


@dataclass(frozen=True)
class StateInfo:
    """Everything the pipelines need to know about the active state.

    Attributes:
        name: Full state name, e.g. "California".
        abbr: Two-letter postal abbreviation, e.g. "CA".
        fips: Two-digit FIPS code as text, e.g. "06".
        metros: The state's metro areas (see ``Metro``).
    """

    name: str
    abbr: str
    fips: str
    metros: tuple[Metro, ...]


def get_state_info(state: str | None = None) -> StateInfo:
    """Look up a US state and return its ``StateInfo``.

    Args:
        state: Full state name or two-letter abbreviation (case-insensitive).
            If omitted, the active ``STATE`` from this file is used.

    Returns:
        A ``StateInfo`` with abbreviation, FIPS code and metros.

    Raises:
        ValueError: If the state is not in the lookup table.
    """
    wanted = (state if state is not None else STATE).strip().lower()
    for name, (abbr, fips, metros) in _STATE_TABLE.items():
        if wanted in (name.lower(), abbr.lower()):
            return StateInfo(
                name=name,
                abbr=abbr,
                fips=fips,
                metros=tuple(
                    Metro(
                        name=metro_name,
                        counties=tuple(counties),
                        indeed_location=f"{metro_name}, {abbr}",
                        geo_name=f"{metro_name}, {abbr}",
                    )
                    for metro_name, counties in metros
                ),
            )
    raise ValueError(
        f"Unknown state {state if state is not None else STATE!r}. "
        "Use a full US state name or abbreviation listed in config._STATE_TABLE."
    )


# ---------------------------------------------------------------------------
# 4. SETTINGS
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# SQLite database file. Can be overridden with the ALTSIGNAL_DB environment
# variable (the unit tests use this to work on a temporary database).
DB_PATH = Path(os.environ.get("ALTSIGNAL_DB", DATA_DIR / "altsignal.db"))

# --- Staleness alert -------------------------------------------------------
# Alert if a source has not had a successful scrape for this many hours.
STALENESS_HOURS = 48

# Names used in the ``source`` column of the ``scrapes`` and ``signals`` tables.
SOURCE_MSRS = "msrs"
SOURCE_INDEED = "indeed"
SOURCE_VIIRS = "viirs"
SOURCE_FRED_TARGETS = "fred_targets"
ALL_SOURCES = (
    SOURCE_MSRS,
    SOURCE_INDEED,
    SOURCE_VIIRS,
    SOURCE_FRED_TARGETS,
)

# Rolling window used for each signal's z-score (number of periods).
# Spec: "rolling 12-month baseline"; weekly series use 52 weeks (~12 months).
ZSCORE_WINDOW = {"monthly": 12, "weekly": 52}
# Fewest OBSERVED periods the baseline window must contain for a z-score to be
# computed. One missing period is tolerated (monthly: 11 of 12, weekly: 51 of
# 52); two or more leave the z-score empty. Nothing is imputed or filled: the
# baseline mean and standard deviation use only the observed periods. Reason:
# the BLS published no unemployment release for October 2025 (federal
# shutdown), and under a strict "every baseline month present" rule that single
# hole emptied every z-score for the following 12 months.
# The window must also be entirely inside the series' history, so a series
# still needs a full window of months before its first z-score.
BASELINE_MIN_PERIODS = {"monthly": 11, "weekly": 51}

# --- Census MSRS -----------------------------------------------------------
MSRS_BASE_URL = "https://www.census.gov/retail/mrts/www/statedata"
MSRS_FILES = {
    "yy": "state_retail_yy.csv",              # year-over-year % changes
    "coverage": "state_retail_coverage.csv",  # non-imputed coverage class A-D
}
# The Census site rejects requests without a browser-like User-Agent string.
HTTP_USER_AGENT = "AltSignal research project (python-requests)"

# --- Indeed ----------------------------------------------------------------
# The omkarcloud/indeed-scraper project runs as a small local web server
# (``python run.py`` inside its repo). We call that server over HTTP.
INDEED_API_URL = os.environ.get("INDEED_API_URL", "http://localhost:8000")
INDEED_RADIUS_MILES = 25      # search radius around each metro's name
INDEED_MAX_PAGES = 50         # the scraper serves at most 50 pages x 20 jobs
INDEED_PAUSE_SECONDS = 0.5    # polite pause between page requests
INDEED_TIMEOUT_SECONDS = 60
INDEED_SCHEDULE_HOUR = 6      # daily run time (24h clock, local time)
INDEED_SCHEDULE_MINUTE = 0
# Raw (deduplicated) postings are kept here so job IDs are never counted twice.
INDEED_POSTINGS_DIR = RAW_DIR / "indeed"

# --- VIIRS nighttime lights (Google Earth Engine) --------------------------
VIIRS_DATASET = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
VIIRS_RADIANCE_BAND = "avg_rad"
VIIRS_COVERAGE_BAND = "cf_cvg"   # number of cloud-free nights behind each pixel
# A pixel counts as "low coverage" if fewer than this many cloud-free
# observations went into it (cf_cvg < threshold, or no data at all).
VIIRS_LOW_COVERAGE_MIN_OBS = 3
# Drop a metro-month if more than this share of its pixels is low coverage.
VIIRS_MAX_LOW_COVERAGE_SHARE = 0.20
VIIRS_START_DATE = "2019-01-01"   # same first month as MSRS
VIIRS_SCALE_METERS = 463.83       # native pixel size (15 arc-seconds)
COUNTIES_DATASET = "TIGER/2018/Counties"
STATES_DATASET = "TIGER/2018/States"
# Earth Engine needs a Google Cloud project. Set EE_PROJECT (or the standard
# GOOGLE_CLOUD_PROJECT). For a service account also set EE_SERVICE_ACCOUNT and
# EE_PRIVATE_KEY_FILE; otherwise run ``earthengine authenticate`` once.
EE_PROJECT = os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

# --- FRED -------------------------------------------------------------------
# Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
# Put it in the FRED_API_KEY environment variable (never commit it).
FRED_TARGET_SERIES = {
    # signal_id: (series id template, geo level, unit)
    # "{abbr}" is replaced with the state abbreviation (e.g. "CAUR").
    "unemployment_rate_state": ("{abbr}UR", "state", "percent"),
    "unemployment_rate_national": ("UNRATE", "national", "percent"),
    "retail_sales_national": ("RSAFS", "national", "millions_usd"),
}

# --- Phase two: cleaning and alignment ------------------------------------
# Plausible range per unit: (lowest, highest); None = no limit on that side.
# Values outside the range are DROPPED (never "fixed") and counted in the
# build report. Ranges are deliberately loose: they catch broken data (a
# negative job count, an unemployment rate of 400%), not unusual-but-real data.
PLAUSIBLE_RANGE = {
    "percent": (0.0, 100.0),          # e.g. unemployment rate
    "percent_yoy": (-100.0, None),    # a level cannot fall by more than 100%
    "postings": (0.0, None),
    "nW/cm2/sr": (0.0, None),         # radiance
    "millions_usd": (0.0, None),
    "index": (None, None),
}
# Sources whose values are official published numbers with nothing imputed on
# our side; their quality_score is 1.0 when the source gives none.
OFFICIAL_SOURCES = (SOURCE_FRED_TARGETS,)
# Months in which Indeed was scraped on less than this share of days are kept
# (with their low quality_score) but excluded from z-score baselines, so a
# 9-day count is never compared with a full-month count.
INDEED_MIN_MONTH_COVERAGE = 0.8
# Per-source minimum quality_score for a row to take part in the z-score.
MIN_QUALITY_FOR_ZSCORE = {SOURCE_INDEED: INDEED_MIN_MONTH_COVERAGE}

# --- Phase three: feature engineering -------------------------------------
# A nowcast for target month T is made on the day after T ends
# (``PREDICTION_OFFSET_DAYS`` after the last day of T). Only observations that
# had ALREADY BEEN PUBLISHED by then may be used.
PREDICTION_OFFSET_DAYS = 1

# Days between the end of an observation's period and the day it becomes
# public. Looked up by signal_id first, then by source. A series with no entry
# here is NOT used (we never guess a release lag). The values are deliberately
# on the conservative (longer) side: too long loses a little information, too
# short would leak the future.
RELEASE_LAG_DAYS = {
    # Census MSRS: the release schedule (msrs_release_schedule.pdf) shows lags of
    # 81-114 days after the end of the data month.
    SOURCE_MSRS: 115,
    # Own daily scrape of Indeed: a month is complete the day after it ends.
    SOURCE_INDEED: 1,
    # VIIRS monthly composites: ASSUMED ~3 months; not verified (no Earth
    # Engine access yet, see blockers.md).
    SOURCE_VIIRS: 90,
    # BLS state unemployment (LAUS) comes out about 3 weeks after month end.
    "fred_unemployment_rate_state": 30,
    # BLS national unemployment: first Friday of the next month (at most ~7 days).
    "fred_unemployment_rate_national": 14,
    # Census advance retail sales: about the 15th of the next month.
    "fred_retail_sales_national": 21,
}

# Which series become features: monthly rows at these geography levels only
# (metro-level series would multiply the columns; use them in a later phase).
FEATURE_GEO_LEVELS = ("state", "national")
# Lags, changes and rolling windows (in months) built on the normalized series.
FEATURE_LAGS = (1, 2, 3)
FEATURE_CHANGES = (1, 3)
FEATURE_ROLLING_WINDOWS = (3, 6)
# Series whose coverage (quality_score x 100) becomes a feature too.
COVERAGE_FEATURE_SOURCES = (SOURCE_VIIRS,)

# Targets to nowcast: name -> candidate (signal_id, geo_level), first one with
# data wins (state unemployment, then national as the fallback).
FEATURE_TARGETS = {
    "retail_sales": [("msrs_yoy_TOTAL", "state")],
    "unemployment": [("fred_unemployment_rate_state", "state"),
                     ("fred_unemployment_rate_national", "national")],
}

# Train/test split: the last TEST_MONTHS months with an observed target are the
# true test set. The boundary is frozen in a file the first time it is defined.
TEST_MONTHS = 6
# Refuse to define a split with fewer labelled months than this in the
# training window (TRAIN_START up to the test set).
MIN_TRAIN_MONTHS = 24

# First target month allowed in the TRAINING set (None = use all history).
# Months before it keep their rows in the feature table (label "pre_window")
# but are never fitted on. Nothing is deleted or changed in the signals table.
# Why 2022-01-01: year-over-year targets such as MSRS swing from about -20% to
# +80% in 2020-21 purely from the pandemic shock and its base effects (the
# 2021 comparison month is the 2020 collapse, the 2022 one is stimulus-inflated
# 2021). Those few extreme months would dominate any fit while describing a
# regime that is not the one being nowcast. This is a first-principles choice,
# not tuned on test results; change it here to widen or narrow the window.
TRAIN_START = "2022-01-01"

# Months flagged by the `pandemic_regime` column of the feature table (inclusive).
# The flag is informational (a meta column, not a model feature): inside the
# default training window it is always 0.
PANDEMIC_REGIME = ("2020-03-01", "2021-12-01")

# Target-own-history features (the target's last PUBLISHED raw values, in the
# target's own units, built at each target's latest published period).
TARGET_HISTORY_PREFIX = "target_hist"

# --- Phase four: benchmarks and models -------------------------------------
# These rules are fixed BEFORE any model result is computed, so they cannot be
# tuned to flatter a model.
#
# Naive benchmarks (src/models/benchmarks.py):
#   * last observation carried forward (LOCF): the latest value of the target
#     that was already published at the prediction time;
#   * year-over-year seasonal naive: the target's own value 12 months earlier.
NAIVE_SEASONAL_LAG_MONTHS = 12
# "Meaningfully better than naive" means a test RMSE at least this share lower
# than the best naive benchmark's RMSE. With a six-month test set even this is
# weak evidence, and the README says so.
MEANINGFUL_MARGIN = 0.10
# Gradient boosting is only run if ridge "works". Ridge works if, in the
# cross-validation inside the TRAINING window (never the test set), its RMSE is
# lower than that of the best naive benchmark on the same validation months.
GRADIENT_BOOSTING_REQUIRES_RIDGE_CV_BEATS_NAIVE = True

# Model settings (set before results were computed; see README, "Phase four").
# Columns with more than this share of missing TRAINING values are not used;
# training rows that are still incomplete are dropped (nothing is imputed).
# 0.5 was the first setting; with the 2022+ training window it left only 27 of
# 48 complete rows because three sparse MSRS subsectors (448, 451, 452) have
# suppressed months. Any share up to 0.2 keeps the same 98 complete columns and
# all 48 rows, so 0.1 was chosen from training-window missingness alone, before
# any model was fitted on the new window.
# Changed again to 0.0 (only columns complete in every training row) after the
# real FRED data and the one-gap z-score rule (BASELINE_MIN_PERIODS) arrived. The
# BLS has no October 2025 release, and every lag/change/rolling window that
# touches that month is empty, as are the z-scores of sparse MSRS subsectors that
# now exist. At 0.1 only 43 (retail) and 40 (unemployment) training rows were
# complete, below the 48 the CV needs; at any value from 0.02 to 0.1 unemployment
# had 40-43 complete rows. 0.0 is the only value that keeps all 48 (retail) and
# all 50 (unemployment) training rows complete, so it follows the same rule as
# before (the loosest setting that keeps every training row), chosen from
# training-window missingness alone before any model was fitted on this data.
MODEL_MAX_MISSING_SHARE = 0.0
# Cross-validation INSIDE the training window: expanding window, this many
# folds, each validating on this many months. A gap (derived from the target's
# release lag) keeps labels that were not yet public out of the training part.
MODEL_CV_FOLDS = 5
MODEL_CV_VALIDATION_MONTHS = 6
MODEL_MIN_TRAIN_ROWS_FIRST_FOLD = 18
MODEL_RANDOM_STATE = 0
RIDGE_ALPHAS = [10 ** (e / 4) for e in range(-8, 17)]          # 0.01 ... 10,000
TREE_GRID = {"max_depth": [2, 3], "min_samples_leaf": [3, 5, 8]}
GBM_GRID = {"n_estimators": [50, 100], "max_depth": [1, 2], "learning_rate": [0.05]}

# --- Phase five: Streamlit app and demo database ---------------------------
# The app reads ``DB_PATH`` when it exists and has data; otherwise it falls back
# to this small committed snapshot so a deployed copy (Streamlit Community Cloud)
# has data without any credentials. Rebuild it with ``python -m src.export_demo``.
DEMO_DB_PATH = DATA_DIR / "demo" / "altsignal_demo.db"
# Only sources whose data is public and may be redistributed go into the demo
# snapshot. Indeed is scraped (non-commercial use only, terms of service unclear)
# and is never exported.
DEMO_EXPORT_SOURCES = (SOURCE_MSRS, SOURCE_VIIRS, SOURCE_FRED_TARGETS)
# A divergence needs both signals at least this many standard deviations from
# their own baseline (z-score), or a 3-month z-score change of at least this much.
DIVERGENCE_MIN_Z = 0.5
# A gap (difference of the two z-scores) of at least this is "strong"; below it "moderate".
DIVERGENCE_STRONG_GAP = 2.0
# Trend rule used by the app: mean of the last 3 months against the mean of the 3
# months before, in units of the series' own standard deviation.
TREND_WINDOW_MONTHS = 3
TREND_MIN_RATIO = 0.25
