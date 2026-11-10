# AltSignal

Monthly nowcasts of official US economic indicators from three alternative
data sources, for any US state: state retail sales (consumption proxy), job
postings, and satellite nighttime lights. The full specification is in
[`docs/altsignal-spec.md`](docs/altsignal-spec.md).

**Status: phases one (ingestion), two (cleaning, alignment, final signals
table) and three (feature engineering) are built.** Models, the Streamlit app and the final write-up
come in later phases. Open blockers are listed in [`blockers.md`](blockers.md).

## Choosing the state

Open `config.py` and change one line:

```python
STATE = "California"      # any US state name or abbreviation, e.g. "Texas" or "TX"
```

All pipelines read the state (abbreviation, FIPS code and metro areas) from the
lookup table in `config.py`, which covers all 50 states and DC. Nothing else
needs editing.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                       # unit tests (no network or keys needed)

python -m src.ingest.msrs              # Census MSRS (no key needed)
python -m src.ingest.indeed            # Indeed (needs the local scraper, below)
python -m src.ingest.viirs             # VIIRS (needs Earth Engine, below)
python -m src.ingest.fred              # FRED: official targets + card spending
python -m src.ingest.run_all           # everything, then the staleness check
python -m src.ingest.staleness         # only the 48-hour staleness check

python -m src.process.build_signals    # phase two: clean + align + z-scores
python -m src.features.build_features  # phase three: feature table + frozen split
```

Data goes into the SQLite database `data/altsignal.db` (created automatically).
Every run, successful or not, adds a row to the `scrapes` table; read it with
`src.db.read_scrapes()`. Exit code 0 means success.

### Credentials and services

| Source | What you need |
| --- | --- |
| Census MSRS | Nothing. |
| Indeed | Run the open-source [omkarcloud/indeed-scraper](https://github.com/omkarcloud/indeed-scraper) locally: `git clone` it, `pip install -r requirements.txt`, `python run.py` (serves `http://localhost:8000`; change with `INDEED_API_URL`). No API key. Use a residential IP if you can; Indeed blocks datacenter traffic. |
| VIIRS | Free noncommercial [Earth Engine](https://earthengine.google.com/) account and a Google Cloud project: `export EE_PROJECT=<project id>`, then `earthengine authenticate` (or `EE_SERVICE_ACCOUNT` + `EE_PRIVATE_KEY_FILE`). |
| FRED | Free API key in `FRED_API_KEY` ([get one](https://fred.stlouisfed.org/docs/api/api_key.html)). |

### Scheduling

`python -m src.ingest.indeed --schedule` runs the Indeed scrape now and then
daily (APScheduler, time set by `INDEED_SCHEDULE_HOUR` in `config.py`).
`python -m src.ingest.run_all --schedule` does the same for every source.
Because the staleness alert fires when a source has no successful scrape in 48
hours, schedule `run_all` (or cron) at least daily.

## Phase two: building the signals table

`python -m src.process.build_signals` turns what ingestion stored into the
final, analysis-ready `signals` table for the active `STATE` (plus the national
series). Run it after the ingest commands, as often as you like: it is
**idempotent** (the table is unique on signal, geography, period and grain, so
reruns update rows in place and never duplicate them).

What it does, per source:

1. **Checks units**: a series with more than one unit stops the build for that
   source with an error. Z-scores are computed per series (signal, geography,
   grain), so values with different units are never mixed.
2. **Cleans** (`src/process/clean.py`) by dropping, never repairing: rows with
   no value, monthly rows not aligned to a calendar month, values outside the
   plausible range for their unit (`config.PLAUSIBLE_RANGE`, e.g. negative job
   counts), and VIIRS months with more than 20% low-coverage pixels. Dropped rows
   are also deleted from the table. **Nothing is imputed or interpolated.**
3. **Aligns to monthly** (`src/process/align.py`). Native monthly sources are
   untouched. Indeed postings are re-counted per calendar month (same
   `signal_id` as the weekly series, `grain = "monthly"`; the weekly rows remain
   with `grain = "weekly"`). Series published less often than monthly would be
   repeated across their months and tagged `grain = "quarterly_repeated"`; no
   current source needs this yet.
4. **Normalizes**: `normalized_value` = z-score against the series' own previous
   12 months (empty until 12 months of history exist). **One missing month in the
   baseline is tolerated** (`config.BASELINE_MIN_PERIODS`: 11 of 12 months, 51 of
   52 weeks): the mean and standard deviation use only the observed months,
   nothing is imputed or filled, and two or more missing months leave the z-score
   empty. The rule is the same for every source. Why: the BLS published no
   unemployment release for October 2025 (federal shutdown), and requiring every
   baseline month emptied all FRED z-scores for the following 12 months. Indeed
   months scraped on fewer than 80% of their days
   (`config.INDEED_MIN_MONTH_COVERAGE`) still get no z-score and are left out of
   baselines (they count as missing months, so at most one may fall in a window).
5. **Reports** per source: `built`, `no data` (not ingested yet, e.g. VIIRS or
   FRED without credentials), `skipped` (optional card spending) or `error`.

The command exits 0 if at least one source was built and nothing is wrong.
Missing-credential sources are reported, not treated as failures.

### `quality_score` (0 to 1) means different things per source

Scores are features, not filters, and are **not comparable across sources**.

| Source | Meaning |
| --- | --- |
| `msrs` | Census non-imputed coverage class A-D, as the middle of its range (0.05, 0.175, 0.375, 0.75). Low values mean the estimate is mostly modeled. |
| `viirs` | Share of pixels with adequate cloud-free coverage (at least 0.80 after cleaning). |
| `indeed` | Share of the period's days on which Indeed was scraped. |
| `fred_*` | 1.0: the published official number itself. |

## Phase three: features and the train/test split

`python -m src.features.build_features` reads the `signals` table for the active
`STATE` and writes three files to `data/processed/` (git-ignored):

* `features_<ST>.csv`: one row per **target** and **target month**. Meta columns
  (`target_name`, `target_series`, `target_month`, `as_of`, `target_available`,
  `pandemic_regime`, `split`, `target_value`) followed by the features. Missing values are empty
  cells.
* `features_<ST>_meta.json`: targets built, split boundaries, release lags used,
  row counts per split and the number of non-missing values per feature.
* `split_<ST>.json`: the **frozen** train/test boundary.

Targets: `retail_sales` (Census MSRS total retail, state) and `unemployment`
(state rate, falling back to the national rate). A target with no data yet
(unemployment, until FRED credentials exist) is reported and skipped. Re-running
with unchanged inputs rewrites identical files (idempotent).

### No look-ahead: prediction time and release lags

A nowcast for target month T is made the day after T ends (`as_of`,
`config.PREDICTION_OFFSET_DAYS`). Each series has a release lag
(`config.RELEASE_LAG_DAYS`, days after its period ends until it is public), and
for T we use only the latest period already published at `as_of`, never the
calendar month:

| Series | Lag (days) | Effect for target month T |
| --- | --- | --- |
| Census MSRS | 115 (the schedule shows 81-114) | latest data is about 4 months old |
| Indeed (own daily scrape) | 1 | month T itself is complete |
| VIIRS | 90 (**assumed**, unverified) | about 3 months old |
| State / national unemployment | 30 / 14 | month T-1 |
| National retail sales (FRED) | 21 | month T-1 |
| Card spending (optional) | 45 (assumed) | only if a series is configured |

Lags are on the cautious side (a too-long lag loses a little information, a
too-short one leaks the future). A series with no configured lag is skipped, not
guessed. The target's own value for T therefore never appears in row T, and
`target_available` records when T's label itself becomes public so later phases
can respect it when training.

### Features (built on the normalized series only)

For each state-level and national series, at its latest published period P*:
`__z` (the z-score), `__lag1/2/3`, `__chg1/3` (changes), `__roll3/6_mean` and
`__roll3/6_std` (rolling statistics over the months up to P*). VIIRS also gets
`__coverage_pct` (share of well-covered pixels x 100). Metro-level series are not
used yet. Different units are never mixed: z-scores are used for the other
series, and the raw target is kept as the label.

**The target's own history (`target_hist__*`).** Each row also carries the
target's own last published raw values, in the same units as the label (for
example percent year-over-year): `last` (the latest published value), `lag1/2/3`,
`chg1/3` (changes), `chg1_lag1` (the previous month's change, a lagged change) and
`roll3_mean`. They follow the target's own release lag exactly like every other
series: for MSRS, row T holds values up to about T-4 months, never T itself. A
nowcast that cannot see the last official print is handicapped against a naive
"repeat the last value" baseline, so this is the information a fair nowcast is
entitled to. (The z-scored `<target>.state__*` columns also remain.) The tests
break if the release-lag logic is removed.

**Nothing is imputed.** A feature is empty if its value, or any month in its
lag, change or rolling window, is missing; an older value is never carried
forward and a rolling window needs all its months. Early months have no features
until 12 months of z-score history exist. Downstream code drops or flags those
rows.

### The training window and the pandemic regime

`config.TRAIN_START` (default `2022-01-01`; `None` = all history) is the first
target month that may be fitted on. Labelled months before it are **kept** in the
feature table with `split = pre_window`; nothing is deleted or altered in the
`signals` table, and they can still be inspected. The default is a
first-principles choice made before looking at any model result: year-over-year
targets such as MSRS swing from about -20% to +80% in 2020-21 purely because of
the pandemic shock and its base effects (the 2021 comparison month is the 2020
collapse; 2022 compares against stimulus-inflated 2021). A handful of such
extreme months would dominate any fit while describing a regime that is not the
one being nowcast. Limits to be aware of: the rolling z-scores of early-2022 rows
still use 2021 months in their 12-month baseline, and features of the first
training rows look back into 2021 (they are inputs, not fitted rows). The column
`pandemic_regime` is 1 for target months in `config.PANDEMIC_REGIME` (default
2020-03 to 2021-12) so anyone who widens the window can flag or stratify those
rows; it is a meta column, not a model feature (it is always 0 inside the default
window). `TRAIN_START` never changes the test window: the frozen boundary is
reused as is, and `TrainSet` also rejects months before the training start.

### The test set

The last six months with an observed target are held out as the true test set
(`config.TEST_MONTHS`), per target. The boundary is **frozen** in
`split_<ST>.json` the first time it is defined, so new months do not move it
(they are labelled `later`); `--refreeze-split` defines a new one on purpose. A
target needs at least 24 labelled months in the training window (from
`TRAIN_START`) before the test window, or no split is defined. Row labels in
`split`: `train`, `pre_window` (before `TRAIN_START`), `test`, `later`, `unlabeled` (target not
published yet or missing; these are the rows to nowcast).

The feature builder has no fitted steps (no scaling or selection), so the table
holds unscaled features. Anything that learns from data must take a `TrainSet`
(`src/features/split.py`): it can only be built from rows labelled `train` (from `TRAIN_START`, before
the test window) and raises on any test-period row, and `fit_standardizer` and
`select_columns` reject everything else. The test rows are available only through
`open_test_set`, as a plain table that those functions do not accept.

Limitation: FRED returns the latest revised values, not what was known at the
time, so official series carry some revision look-ahead (see `blockers.md`).

## What is stored

Four SQLite tables (`src/db.py`, the only module that touches the database):
`signals`, `scrapes`, `models`, `forecasts`.

Every signal row has the raw `value` and a `normalized_value`: a z-score
against that signal's own trailing baseline (previous 12 months; 52 weeks for
weekly series). The baseline excludes the current period, and a z-score is left
empty when the baseline has two or more missing periods (one is tolerated and the
statistics use the observed periods only); nothing is ever imputed. Raw values of
different signals are never compared directly. The `grain` column tags each
row `monthly` or `weekly`.

| Source | `signal_id` | Grain | Notes |
| --- | --- | --- | --- |
| Census MSRS | `msrs_yoy_<NAICS or TOTAL>` | monthly | YoY % change by retail subsector; `quality_score` is the Census coverage class A-D mapped to a number. Suppressed ("S") cells are skipped. |
| Indeed | `indeed_postings_<sector>`, `indeed_postings_total` | weekly | New unique postings per metro and sector (see caveats). `quality_score` = share of the week's days scraped. |
| VIIRS | `viirs_avg_radiance` | monthly | Mean radiance per metro and for the state; `quality_score` = share of pixels with adequate coverage; months with more than 20% low-coverage pixels are dropped. |
| FRED targets | `fred_unemployment_rate_state`, `fred_unemployment_rate_national`, `fred_retail_sales_national` | monthly | Official numbers to be nowcast later. |
| FRED card spending | `card_spending_index` | monthly | National cross-check only. Optional: needs a configured series ID (see `blockers.md`); the build skips it when absent. |

Metro rows are named with their state (`Springfield, MA`) because metro names
repeat across states; the database can therefore hold several states at once,
and each build only touches the active state's rows.

## Data sources, access terms and limitations

* **Census MSRS is experimental and modeled.** The Monthly State Retail Sales
  are an experimental Census product blending the Monthly Retail Trade Survey,
  administrative data and third-party data. They are modeled year-over-year
  changes (not seasonally adjusted, from January 2019), not a direct state
  survey, and are subject to revision (a revision was announced for the July
  2026 release). Treat them accordingly.
* **Indeed disclaimer.** Indeed has no public API. This project uses an
  open-source scraper for personal research and education. **Do not build
  anything commercial on scraped Indeed data**, and check Indeed's terms of
  service before using it. Counts are a sample, not a census: the scraper only
  sees postings Indeed returns (large metros come back well below the count
  Indeed reports); a metro is searched by city name plus a 25-mile radius, which
  only approximates the metro; the "sector" is Indeed's own occupation grouping;
  and a posting is counted in the week (or month) we first saw it, not when it
  was published. A month's `quality_score` is computed from the scrape log, which
  does not record the state, so scrapes of different states on different days
  would all count as observed days; scrape one state at a time per database.
* **VIIRS** (NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG) is public domain; Earth Engine
  is free for noncommercial use. A pixel counts as low coverage when fewer than
  3 cloud-free nights contributed to it (`config.VIIRS_LOW_COVERAGE_MIN_OBS`);
  this threshold is a judgement call.
* **FRED** data is subject to each series' own terms; official series are
  revised, and the latest vintage is stored.
* **Metro definitions** in `config.py` are an approximate selection of the
  largest 2-5 metro areas per state, built from principal counties.

## Methodological backing

Using retail sales as a state consumption proxy is supported by recent work on
credit card data: Aladangady, Gabriel and Wix, "Measuring Consumption with
Credit Card Data: Benchmarking and Beyond", Federal Reserve Board *Finance and
Economics Discussion Series* (FEDS), October 2026
([Fed page](https://www.federalreserve.gov/econres/feds/measuring-consumption-with-credit-card-data-benchmarking-and-beyond.htm)).
The paper uses Y-14M credit card data and reports a pooled state-year
regression of annual card spending growth on annual adjusted PCE growth with a
slope of 0.93 and adjusted R-squared of 0.88, i.e. 88% of the variation.
Caveats: this is a FEDS working paper (not peer reviewed); the 88% pools
time-series and cross-state variation and is driven by the 2020-21 pandemic
swings, while the within-year cross-sectional fit is weaker (not significant in
2019, 2022 and 2023); and it concerns card spending, not Census retail sales,
so it supports the proxy idea only indirectly.

## Project layout

```
config.py        active state + lookup table + settings
src/db.py        the only module that touches SQLite
src/normalize.py rolling z-scores
src/ingest/      msrs.py, indeed.py, viirs.py, fred.py, staleness.py, run_all.py
src/process/     clean.py, align.py, build_signals.py (phase two)
src/features/    engineer.py, split.py, build_features.py (phase three)
tests/           unit tests (python -m pytest)
data/            raw/ and processed/ files (git-ignored) and the SQLite file
notebooks/ app/  placeholders for later phases
docs/            the specification
```
