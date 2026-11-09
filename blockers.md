# Blockers

_Status after phase two: no new blockers. The three items below are unchanged; phase two for VIIRS and FRED is implemented and unit-tested against fixtures/mocks only._

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
