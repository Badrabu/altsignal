# AltSignal

Monthly nowcasts of official US economic indicators from three alternative
data sources, for any US state: state retail sales (consumption proxy), job
postings, and satellite nighttime lights. The full specification is in
[`docs/altsignal-spec.md`](docs/altsignal-spec.md).

**Status: phase one only (ingestion pipelines).** Cleaning/alignment, features,
models, the Streamlit app and the final write-up come in later phases.
Phase-one blockers are listed in [`blockers.md`](blockers.md).

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

## What is stored

Four SQLite tables (`src/db.py`, the only module that touches the database):
`signals`, `scrapes`, `models`, `forecasts`.

Every signal row has the raw `value` and a `normalized_value`: a z-score
against that signal's own trailing baseline (previous 12 months; 52 weeks for
weekly series). The baseline excludes the current period, and a z-score is left
empty when the baseline has gaps; nothing is ever imputed. Raw values of
different signals are never compared directly. The `grain` column tags each
row `monthly` or `weekly`.

| Source | `signal_id` | Grain | Notes |
| --- | --- | --- | --- |
| Census MSRS | `msrs_yoy_<NAICS or TOTAL>` | monthly | YoY % change by retail subsector; `quality_score` is the Census coverage class A-D mapped to a number. Suppressed ("S") cells are skipped. |
| Indeed | `indeed_postings_<sector>`, `indeed_postings_total` | weekly | New unique postings per metro and sector (see caveats). `quality_score` = share of the week's days scraped. |
| VIIRS | `viirs_avg_radiance` | monthly | Mean radiance per metro and for the state; `quality_score` = share of pixels with adequate coverage; months with more than 20% low-coverage pixels are dropped. |
| FRED targets | `fred_unemployment_rate_state`, `fred_unemployment_rate_national`, `fred_retail_sales_national` | monthly | Official numbers to be nowcast later. |
| FRED card spending | `card_spending_index` | monthly | National cross-check only. Needs a configured series ID (see `blockers.md`). |

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
  and a posting is counted in the week we first saw it, not the week it was
  published.
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
tests/           unit tests (python -m pytest)
data/            raw/ and processed/ files (git-ignored) and the SQLite file
notebooks/ app/  placeholders for later phases
docs/            the specification
```
