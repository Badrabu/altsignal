# AltSignal specification (from Deen Os)

You are a senior data engineer and econometrician building a public economic nowcasting project called AltSignal. The goal: combine three alternative data sources (state-level retail sales as a consumption proxy, job postings, and satellite nighttime lights) to produce monthly nowcasts of official US economic indicators before they are released, and to surface divergences between signals.

The project starts with California as the pilot state but is architected to be fully dynamic: any US state can be selected via a single configuration value, and all pipelines, aggregations, and models switch automatically. National-level scaling is a future option, not the current scope.

The builder is an experienced economist with a central banking background who is learning software engineering, so write clean, well-commented, beginner-friendly code with docstrings on every function. Use Python 3.12. All dependencies go in a requirements.txt.

The project lives in a single GitHub repository with this structure: `data/` for raw and processed files, `src/` for all Python modules, `notebooks/` for exploration only, `app/` for the Streamlit dashboard, `tests/` for unit tests, and a README at the root. A config file called `config.py` at the project root defines the active state as a single variable, for example `STATE = "California"`. Every module reads from this config. Changing the state value and rerunning should produce a complete nowcast for the new state without editing any other file.

## Data sources and access methods

1. Census Monthly State Retail Sales (MSRS). Free experimental data product from the Census Bureau providing modeled year-over-year percentage changes by state and retail subsector, monthly, starting January 2019. Download the CSV files directly from census.gov/retail/state_retail_sales.html. This is the primary consumption signal because it is state-level and official.
2. Indeed job postings. There is no public Indeed API. Use the open-source scraper from GitHub repository omkarcloud/indeed-scraper, which runs locally with no API key and no proxies. Schedule it daily via APScheduler. Deduplicate postings by job ID. Track posting counts by sector and metro area weekly, filtered to the active state's metros.
3. VIIRS nighttime lights. Access through Google Earth Engine using the Python client library earthengine-api. Dataset: NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG. Aggregate average radiance by county or metro area within the active state, monthly. Public domain. Earth Engine requires a free noncommercial Community tier signup with 150 compute hours per month.
4. SF Fed card spending index via the FRED API using fredapi. National only; secondary cross-check signal, never the primary state-level consumption input.
5. Official targets from FRED: next month's unemployment rate from BLS and retail sales from Census.

## Critical constraints

- Never impute missing VIIRS data. Use the quality band to drop months where more than 20 percent of a metro's pixels are flagged as low coverage, and store the coverage percentage as a feature.
- Never mix raw values across signals with different units. Always store both raw value and a normalized z-score against that signal's own rolling 12-month baseline.
- The analysis grain is monthly. Tag any repeated values with a `grain` column.
- The Indeed scraper must write a row to a scrapes log table on every run with status, rows returned, metro coverage, and error message, even on failure. Alert if any source has not had a successful scrape in 48 hours.
- Run the scraper from a residential IP if possible because Indeed blocks datacenter traffic.
- Do not build anything commercial on scraped Indeed data. Include a clear disclaimer in the README.
- MSRS data is experimental and modeled; document that limitation.
- Cite the Federal Reserve's October 2026 working paper on Y-14M credit card data, which shows card spending growth explains 88 percent of annual state-level PCE growth variation, as methodological backing for using retail sales as a consumption proxy. (Verify the citation actually exists and note any uncertainty rather than fabricating details.)

## Database (SQLite), four tables

- `signals`: signal_id text, source text, geo_level text, geo_name text, period_start date, period_end date, value real, normalized_value real, unit text, quality_score real, grain text, fetched_at timestamp.
- `scrapes`: scrape_id integer primary key, source text, started_at timestamp, finished_at timestamp, status text, rows_returned integer, error_message text nullable, metro_coverage real.
- `models`: model_id text, model_type text, target text, train_start date, train_end date, test_start date, test_end date, metric_name text, metric_value real, created_at timestamp.
- `forecasts`: forecast_id integer, model_id text foreign key, target_period date, predicted_value real, actual_value real nullable, error real nullable, UNIQUE (model_id, target_period) so retraining updates rather than duplicates.

All database access goes through one module, `db.py`, with clear read and write functions so it can be swapped for Postgres later without touching the rest of the code.

## Modeling (phase four)

Primary model: ridge regression with cross-validation, trained only on the training window. Comparison models: a shallow decision tree and optionally gradient boosting, only if ridge works. Naive benchmarks defined before any results are seen: last observation carried forward, and year-over-year seasonal naive. Hold out the last six months as a true test set never used for training, feature selection, or tuning. Targets: next month's state-level unemployment where available (national as fallback), and state retail sales from MSRS. Report MAE and RMSE on the test set for every model. If no model beats naive by a meaningful margin, the README must say so honestly.

## Interface (phase five)

Streamlit app with three tabs. Tab 1: each signal's current trend and latest reading for the active state, with raw/normalized toggle. Tab 2: nowcast versus official release with full error history. Tab 3: divergences where signals disagree, with a plain-language summary generated from the numbers, not from an LLM. The app reads the active state from config.py and displays it prominently. Deployable to Streamlit Community Cloud for free.

## Documentation (phase six)

README covers methodology, data sources, access terms, the state-switching design, limitations, honest results, and a link to the live demo. One short LinkedIn post highlighting one counterintuitive finding.

## Build order

Phase 1: ingestion pipelines for all sources with the scrapes log, parameterized by the active state from config. Phase 2: cleaning, alignment, and the signals table. Phase 3: feature engineering. Phase 4: models and benchmarks. Phase 5: Streamlit app. Phase 6: README and packaging. Each phase must produce something runnable before the next begins.

If you hit a blocker (a broken scraper, an Earth Engine quota issue, a FRED rate limit, a missing MSRS file), stop, document the problem and your attempted fix in `blockers.md`, and ask before proceeding.

**Start with phase one.**
