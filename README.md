# AltSignal

Monthly nowcasts of official US economic indicators from three alternative
data sources, for any US state: state retail sales (consumption proxy), job
postings, and satellite nighttime lights. The full specification is in
[`docs/altsignal-spec.md`](docs/altsignal-spec.md).

**Status: phases one (ingestion) and two (cleaning, alignment, final signals
table) are built.** Features, models, the Streamlit app and the final write-up
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
tests/           unit tests (python -m pytest)
data/            raw/ and processed/ files (git-ignored) and the SQLite file
notebooks/ app/  placeholders for later phases
docs/            the specification
```
