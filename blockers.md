# Blockers

_Status after phase five: nothing new blocks the dashboard (see "Phase five notes"). Status after the real FRED run and the one-gap z-score rule: FRED works (item 2 below is resolved) and models run for both targets (section "Real FRED run: October 2025 gap"). Items 1 and 3 are unchanged._

# Phase one blockers

Phase one is built and its unit tests pass (`python -m pytest`). Three pieces
could not be fully verified or completed without things I do not have. Nothing
was faked: every failure below is logged to the `scrapes` table with a clear
error message. Phase two has **not** been started.

## 1. SF Fed card spending: no FRED series ID found (needs a decision)

**Problem.** The spec says to get the "SF Fed card spending index via the FRED
API". I could not find any such series.

**What I tried**
* Searched FRED's site for "SF Fed card spending", "frbsf card", "card spending
  index", "credit card spending", "consumer spending tracker" and the San
  Francisco Fed as a source: no card spending series came back (only unrelated
  series such as credit card delinquency rates and the San Francisco Tech
  Pulse).
* Requested a guessed ID (`SF_FED_CARD_SPENDING_INDEX`, which a web search tool
  suggested) from FRED: it does not exist (the search tool's answer looks
  fabricated).
* Web searches: the San Francisco Fed does not appear to publish a standalone
  card spending index data release. The paper cited in the spec is from the
  Federal Reserve **Board** (FEDS), not the SF Fed, and I found no sign that
  its county-level Y-14M dataset is published as a FRED series.

**What the code does now.** `python -m src.ingest.fred` runs the card spending
pipeline against `config.CARD_SPENDING_SERIES_ID` (or the env var of the same
name). It is unset, so the run is logged as `failed` with a message pointing
here. No ID is guessed. The pipeline itself is unit-tested with a fake FRED
client.

**Options (please choose)**
1. You give me the intended series ID or another source for it.
2. Use the Visa Spending Momentum Index on FRED (`VISASMIHSA`, headline,
   monthly, national; exists and is current to Sept 2026). It is not an SF Fed
   series and has its own terms, so I did not substitute it without approval.
3. Drop the card spending cross-check.

## 2. FRED targets: no API key, so the fredapi path is untested with real data

**Problem.** No `FRED_API_KEY` is available in this environment.

**What I did.** Confirmed the target series exist and are current using FRED's
public CSV download (no key): `CAUR`, `TXUR`, `DCUR` (state unemployment),
`UNRATE`, `RSAFS`. Ran `python -m src.ingest.fred` with no key (logged failure
"FRED_API_KEY is not set") and with a made-up key (logged failure "FRED
rejected the API key", reproduced against the live FRED API). Unit tests use a
fake client. **Not verified:** a real `fredapi` download and write into
`signals`.

**To unblock.** Export a free `FRED_API_KEY` and run
`python -m src.ingest.fred`.

## 3. VIIRS via Earth Engine: no credentials, so nothing run live

**Problem.** No Earth Engine account/project or authentication here.

**What I did.** Ran `python -m src.ingest.viirs` with no project (logged failure
telling the user to set `EE_PROJECT`) and with a fake project (the real
`ee.Initialize` fails; logged failure telling the user to run
`earthengine authenticate`). The coverage rule (drop months with more than 20%
low-coverage pixels, never impute, store coverage), partial-failure handling and
row building are unit-tested with Earth Engine replaced by fakes.

**Not verified live (assumptions to check on the first real run)**
* The dataset has bands `avg_rad` and `cf_cvg` (my understanding of
  `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG`), and `cf_cvg` (number of cloud-free
  observations) is the right quality band.
* "Low coverage" means fewer than 3 cloud-free observations
  (`VIIRS_LOW_COVERAGE_MIN_OBS`); the spec does not define the threshold, so
  this is my choice.
* The county names in the `config.py` metro table match `TIGER/2018/Counties`
  `NAME` values (the code fails loudly if a county is missing), and a
  `reduceRegion` over a whole state per month fits Earth Engine's memory and
  time limits.

**To unblock.** Create a free noncommercial Earth Engine project, then
`export EE_PROJECT=<id>` and `earthengine authenticate` (or a service account),
and run `python -m src.ingest.viirs`.

## Verified live

* **Census MSRS**: real download for California (1,065 rows) and Texas (1,073
  rows) with z-scores and coverage-based quality scores.
* **Indeed**: the omkarcloud scraper ran in this sandbox and was **not**
  blocked at the time (datacenter IP); a real full California run stored
  deduplicated postings and weekly counts, and a re-run added 0 duplicates.
  Indeed may block this later; a blocked or unreachable scraper is logged as a
  failed run.
* **Failure paths and staleness**: failed runs write `scrapes` rows with clear
  errors; `run_all` and the 48-hour staleness check report the three sources
  above as never having succeeded.

## Phase two notes

* **Rule change (decided by the user after the real FRED run): one missing
  month in a z-score baseline is tolerated.** The BLS published no unemployment
  release for October 2025 (federal shutdown), so CAUR and UNRATE have no value for
  that month. Under the original rule (all 12 baseline months required) that single
  hole emptied every FRED z-score from Nov 2025 to Oct 2026 and, through the
  features built on them, left no model fittable or scorable. The baseline now needs
  at least `config.BASELINE_MIN_PERIODS` observed periods (monthly 11 of 12, weekly
  51 of 52); the statistics use observed periods only; nothing is imputed; two
  missing periods still give no z-score; the series' first 12 periods still have
  none. The rule is uniform across sources, and the Indeed under-80%-days
  exclusion is unchanged (an excluded month counts as one missing month). It
  changes z-scores of any series with an isolated gap (for example MSRS subsectors
  with suppressed months), so results computed before the change are not
  comparable.

* Verified on real data: `python -m src.process.build_signals` on a database
  holding real MSRS (California) and real Indeed postings gives MSRS 1,065 rows
  with z-scores and quality scores, and Indeed monthly rows; a second run
  leaves the table unchanged. VIIRS and FRED report `no data` there (blocked
  above); their processing is tested with the phase-one fakes.
* The real Indeed data covers a single scrape day, so its monthly rows have a
  tiny `quality_score` and no z-score. That is expected until the scraper has
  run daily for a while, and 12 months of history are needed for a z-score.
* Fix to a phase-one weakness: metro `geo_name`s are now `"<Metro>, <ST>"`
  (they collided across states, e.g. Springfield MA/MO). This touches
  `config.py`, `ingest/indeed.py` and `ingest/viirs.py`; old rows with the
  previous names in an existing database are ignored by the build and can be
  deleted.

## Phase three notes

* Verified on real data: `python -m src.features.build_features` on the real
  California MSRS series gave 94 rows (84 train, 6 test = Jan-Jun 2026, 4
  unlabeled months to nowcast) and 130 features, and a rerun produced
  byte-identical files. Unemployment reports "no data" (FRED blocked) and all
  Indeed features are empty (only one scrape day, no 12-month z-score yet).
* **Release lags are my estimates.** MSRS (115 days) comes from the Census
  release schedule; the VIIRS lag (90 days) is an unverified assumption until
  Earth Engine access exists; BLS/Census lags are typical release timing. All
  are on the cautious side and live in `config.RELEASE_LAG_DAYS`.
* **Revision look-ahead in FRED.** `fredapi` `get_series` returns the latest
  revised values, so official series (unemployment, retail sales) are not exactly
  what was known on the prediction date. Fixing this needs vintage data
  (`fredapi` offers `get_series_first_release` / `get_series_as_of_date`), which
  I cannot test without an API key. Phase four should consider it before
  reporting backtest results.
* **Feature/split revision (after phase four found no model beat naive).** Added,
  on first-principles grounds and without looking at test-set results: the
  target's own published history (`target_hist__*`, release-lag respecting) and a
  configurable `TRAIN_START` (default 2022-01-01) that excludes the 2020-21
  pandemic months from fitting (rows kept as `pre_window`; signals table
  untouched). The frozen test window is unchanged (California: 2026-01 to
  2026-06), now 48 train / 36 pre_window / 6 test rows. Caveat: 48 training
  months is small, and early-2022 z-scores still have a 2021 baseline. Whether
  this helps is for phase four to measure; I did not evaluate models.
* MSRS can also be revised after first release, and only the current vintage is
  stored, so it carries the same kind of (smaller, unquantified) revision risk.

## Phase four notes

* **FRED_API_KEY was not available in this environment** (the variable was not
  set when phase four was built; its value was never read or printed). So the
  real FRED ingest and the **unemployment target were not run**; only the
  retail-sales target (Census MSRS, real data, no key needed) was modelled.
  Unemployment is reported as "no data in the feature table", not faked. The
  code path is the same for both targets and is unit-tested with synthetic
  tables only. **To unblock:** export `FRED_API_KEY`, then run
  `python -m src.ingest.fred`, `python -m src.process.build_signals`,
  `python -m src.features.build_features` and `python -m src.models.run_models`
  (the unemployment train/test split is defined and frozen on the first build).
* **Revision look-ahead is still open** (phase three note above). FRED
  vintages (`get_series_first_release` / `get_series_as_of_date`) would fix it
  for unemployment but cannot be tested without a key. MSRS vintages are not
  stored at all.
* **Rerun after the phase-three revision.** With the 2022+ training window and
  target-own lags, ridge and the tree edge out naive on the six test months
  (details and caveats in the README); the first run (all history, no own lags)
  had no model near naive. `MODEL_MAX_MISSING_SHARE` was lowered from 0.5 to 0.1
  because the sparse MSRS subsectors (448, 451, 452) left only 27 of 48 complete
  training rows. The naive rules, the 10% margin and the gradient-boosting gate
  are unchanged. FRED_API_KEY was checked again and was still not set.

## Real FRED run: October 2025 gap (resolved by the one-gap z-score rule)

**Resolved:** item 2 above. With a real key, `python -m src.ingest.fred` logged
`fred_targets` = `success` (1,970 rows: `CAUR`, `UNRATE`, `RSAFS`; the key was never
printed or stored); `fred_card` is still `failed`/skipped (item 1). `build_signals`
and `build_features` then built the unemployment target (state series `CAUR`,
national fallback not needed): 50 train, 6 test (2026-04 to 2026-09), 552
pre_window and 2 unlabeled rows.

**Problem.** The BLS did not publish unemployment for October 2025 (no household
survey during the federal shutdown); FRED has no value for `CAUR` or `UNRATE` at
2025-10-01 (`RSAFS` is complete). Under the existing rules:

* a z-score needs a complete 12-month baseline, so all FRED z-scores are empty
  from 2025-11 to 2026-10, and every lag/change/rolling feature (built on z-scores)
  is empty with them. **All FRED-derived features are empty in all six test months
  of both targets.**
* training rows with any missing feature are dropped: 45 of 50 unemployment rows
  and 46 of 48 retail rows remain, below the 48 the CV needs, so ridge and tree are
  not fitted ("only 45 complete training rows; need at least 48"). This also removes
  the retail-sales models that ran before FRED data existed (those results still
  reproduce exactly on a database without FRED data).

**What I tried.** Lowering `MODEL_MIN_TRAIN_ROWS_FIRST_FOLD` from 18 to 12 so that
the models would fit: they fit, but no test month had a full set of features, so
nothing could be scored (and the CLI crashed formatting empty scores). Reverted; no
setting was changed. Naive benchmarks (LOCF, seasonal), the 10% margin, the
boosting gate and the frozen split are untouched. Naive test scores: unemployment
LOCF MAE 0.033 / RMSE 0.058, seasonal 0.300 / 0.321; retail LOCF 1.57 / 2.18,
seasonal 2.65 / 3.25.

**Decision taken (by the user): option 1, tolerate one missing month in a z-score
baseline** (`config.BASELINE_MIN_PERIODS`, phase-two rule, documented in the README
and the phase two notes; two missing months still give no z-score; nothing is
imputed). The full chain was then rerun on a fresh database with real data.

**What the rule did not fix, and the disclosed setting change.** Lag, change and
rolling windows that touch October 2025 stay empty for up to six months, and the
rule also gives z-scores to sparse MSRS subsectors. With `MODEL_MAX_MISSING_SHARE`
= 0.1 only 43 (retail) and 40 (unemployment) training rows were complete, below the
48 the CV needs, so no model was fitted. `MODEL_MAX_MISSING_SHARE` was therefore
set to 0.0 (only columns with no missing training value): 0.0 is the only value
that keeps all 48 (retail) and all 50 (unemployment) training rows complete, chosen
from training-window missingness alone before any model was fitted on this data.
Side effects: unemployment models use 103 columns that are mostly MSRS and national
retail-sales features, with almost all FRED unemployment features dropped; and two
retail test months (Jan, Feb 2026) cannot be forecast by any model (a feature
window touches the hole), so retail is scored on four months (Mar-Jun 2026).
Naive benchmarks, the 10% margin, the boosting gate and the frozen test split are
untouched. Results: README, "Current results".

**Bug fixed (exposed by the real data):** `run_models` crashed printing the report when a target had no scorable test month (empty scores are now shown as `n/a`; test added). Also fixed: `run_models` left stale model and forecast rows from earlier runs (for example ridge rows after a rerun in which ridge was no longer fitted); a rerun now replaces the target's stored models and forecasts (`db.delete_models`; tests added).

Also open: **revision look-ahead.** FRED returns the latest vintage; first-release
or as-of-date vintages (`get_series_first_release` / `get_series_as_of_date`) would
remove it but were not used. All scores are somewhat optimistic for that reason.

## Phase five notes

* **Not deployed.** No Streamlit Community Cloud account or GitHub-linked
  deployment was available here, so the app was run locally
  (`streamlit run app/streamlit_app.py`), smoke-tested with Streamlit's `AppTest`,
  and checked in a bare environment holding only `app/requirements.txt`, but the
  hosted deployment itself is untested. The README's live-demo link is a
  placeholder. **To unblock:** create the app on share.streamlit.io (main file
  `app/streamlit_app.py`, Python 3.12) and put its URL in the README. One thing to
  verify on first deploy: that Community Cloud picks `app/requirements.txt` (next
  to the entry point) rather than the root file; if it picks the root file the app
  still works, only the install is slower.
* **The demo database holds MSRS (California), the real FRED official series
  and the phase-four results of both targets** (naive benchmarks, ridge, tree),
  refreshed after the real FRED run and the one-gap z-score rule. VIIRS (Earth
  Engine) and card spending (no verified series) are still blocked, and Indeed has
  too little history for a z-score and is deliberately never exported to the
  public demo (scraped data, non-commercial use only). The app shows each of these
  as "No data yet" with the reason; nothing is faked. Cross-source divergences still
  cannot be assessed (the official FRED targets are not compared as signals); only
  subsector-versus-total divergences within retail (MSRS) are shown.
* **Scoreboard months.** Phase four scores every model on the months all of them
  could forecast, so retail is scored on 4 of its 6 test months (two have a
  feature window touching the missing October 2025 release). The app's "Test
  months" column shows that common number and a note says so.
* **Manual refresh.** The demo snapshot is rebuilt by hand with
  `python -m src.export_demo` and committed; nothing refreshes it automatically.
  It is a binary SQLite file in git, so each refresh adds about its size (~250 KB)
  to the history.
* **Decision to review: the app's trend and divergence thresholds** (3-month
  windows, 0.25 standard deviations for a trend, 0.5 for a divergence, 2.0 for
  "strong") are my choices, set in `config.py` before looking at any divergence
  output; they are not estimated from data.

## Decisions to review (not blockers)

* `forecasts.model_id` is **not** a SQL foreign key. The spec stores one row per
  metric in `models` (so `model_id` repeats there), which SQLite cannot
  reference. `write_forecasts` checks the model exists in Python instead.
* Job-ID deduplication needs memory between runs, so deduplicated Indeed
  postings are kept in `data/raw/indeed/<ST>_postings.csv` (git-ignored) rather
  than adding a fifth database table.
* `signals` has a unique index on (signal_id, geo_level, geo_name, period_start,
  grain) so reruns update rows instead of duplicating them.
* The scrapes `status` values are `success`, `partial` and `failed`; only
  `success` counts for the staleness alert.
