# Adversarial review: leakage and overfitting

Reviewer scope: data leakage and overfitting / eval-window contamination only. Style,
performance and features were out of scope. Reviewed at HEAD `e2af3fc`. No file was
modified except this one. `make backtest` was **not** run, and the eval ledger
(`reports/eval_runs.json`, 1 run) was not touched. Every empirical check below either
reads committed artefacts or runs strictly inside the dev window.

## Summary

| # | Severity | Finding | Location |
|---|---|---|---|
| 1 | **HIGH** | Fighter attributes are as-of scrape time, and **missing reach (also DOB and height) encodes the fighter's future career**. In 2005–2017 the fighter with a missing reach wins only 0–11% of bouts. LightGBM learns this. It produces **all** of the market+model variant's eval "improvement", all of its dev improvement, and most of the dev-window signal. The headline result (the model loses to the market) **survives**. | `features/pit.py:333-345`, `ingest/normalize.py:174-206` |
| 2 | MEDIUM | The leakage suite cannot detect finding 1. The mutation and shuffle tests perturb only `fights` and `fight_stats`, never the `fighters` table. All 27 `leakage` tests pass while the leak is live. | `tests/test_leakage.py:83-160, 470-530` |
| 3 | MEDIUM | The frozen hyperparameters were tuned on a leaky objective. 17.4% of the tuning validation tail (150 of 860 bouts) carries the reach leak. The README's "real (control)" AUC of 0.69 is mostly the leak: it drops to 0.59 on clean rows. | `models/tune.py:57-90`, `config/config.yaml:84-110`, `README.md:247` |
| 4 | LOW | The market+model stacker can be refit on same-date outcomes. When two events share a date, the second event's refit includes the first event's labels. This happened once (fold 14, 2014-08-23, dev window). | `backtest/walkforward.py:154-205` |
| 5 | LOW | The walk-forward cutoff test is self-referential. It checks `train_date_max` as reported by the engine, and the engine's own guard is tautological with its filter. A model fitted on a different frame than the one reported would still pass. | `tests/test_backtest.py:275-300`, `backtest/walkforward.py:124-133` |
| 6 | LOW | Eval-ledger provenance is weak. The run is stamped with git `1c1e1f0`, but the walk-forward and report code were uncommitted at that time. The ledger can be deleted and re-created. Parlay runs over the eval window are not ledgered at all, and several parlay settings were added after the eval result was known. | `reports/eval_runs.json`, `backtest/report.py:255-282`, `config/config.yaml:149-158` |
| 7 | INFO | Early stopping and isotonic calibration share the same chronological validation tail. This adds mild optimism to the calibration, but is not cross-time leakage. | `models/lgbm.py:122-160` |
| 8 | INFO | Elo: if a fighter fights twice on one date, the second update overwrites the first. This is a correctness bug, not leakage. | `features/elo.py:75-80` |
| 9 | INFO | 57 "Overturned" results are coded NC from the start, but an overturn happens months later. This removes a little information; it does not add any. | `ingest/normalize.py:58-70, 296` |
| 10 | INFO | Demo "hypothetical matchup" uses a model fitted on all data. Pairs that met only in unpriced bouts are labelled "never fought" but the model trained on their bout. | `export_app.py:178-183`, `app/streamlit_app.py:253-290` |
| 11 | INFO | Bankroll replay stakes at the closing price and compounds bets sequentially within a card. This is slightly optimistic, and irrelevant to a negative result. | `sim/bankroll.py:43-55`, `sim/pipeline.py:200-215` |

**Bottom line.** The headline is robust: on the leak-free subset of the eval window,
LightGBM still loses to the vig-free close by **+0.0653** nats, CI [+0.0529, +0.0794]. The
secondary claims need correcting:

- The market+model variant result is pure leakage. It goes from −0.0033 (p≈0.10) to
  +0.0001 (p≈0.98) on clean rows.
- The dev-window variant figure (−0.022, CI excluding zero) is also leakage.
- The README's leakage-diagnostic "real control" and the tuning objective are both
  leak-inflated.

---

## Finding 1 — HIGH: missingness of scrape-time attributes leaks each fighter's future career

**Mechanism.** `build_features` takes `height_cm`, `reach_cm` and `dob` from the
`fighters` table (`pit.py:333-345`). That table is the UFCStats tale of the tape as
scraped on 2026-10-09 (`normalize.py:174-206`). The values themselves are near-static, so
using them is mostly harmless. Their **missingness** is the problem: UFCStats records reach
for fighters who stayed on the roster into the era when reach was tracked. A fighter who
lost and was cut early never had reach recorded. `reach_1` being NaN while `reach_2` is
present therefore says "fighter 1 did not have a long UFC career after this", which is
information from after the fight. LightGBM splits on NaN natively (the README presents
this as a feature), so it learns the pattern directly.

**Empirical evidence.** All labelled bouts. Win rate of the fighter whose reach is
missing, when exactly one side is missing:

```
uv run python -c "...features.parquet; miss_side = one-sided reach NaN; groupby year..."
year   n   missing_fighter_won   (debut share)
2005  32   0.031
2007  56   0.036
2008  44   0.000
2011  35   0.000
2012  57   0.053
2015  75   0.013
2016  45   0.022
2017  17   0.059
...
2025  15   0.467
2026  67   0.537        <- recent debutants not yet populated: ~50%, as it should be
```

The 2025–26 rows are the control. Where missingness simply reflects "not yet populated",
the missing-reach fighter wins about half the time. In the historical era the rate is
0–6%, which no pre-fight variable could produce. DOB and height show the same pattern
before 2010, on smaller counts (for example, pre-2005 `age_1` missing gives a fighter-1
win rate of 0.158, n=57).

**Share of bouts affected** (one-sided missing reach): about 12–22% of bouts per year in
2008–2015, 9.4% in 2016, 3.8% in 2017, about 0% in 2019–2024, then rising again in
2025–26. In the walk-forward predictions this is 327 of 2,154 dev bouts and 94 of 4,636
eval bouts.

**Impact on the committed predictions** (`reports/walkforward_predictions.csv`):

| Window / subset | n | Market LL | LightGBM LL | Market+model LL (market on same rows) |
|---|---:|---:|---:|---:|
| Dev, one-sided missing reach | 327 | 0.5577 | **0.4272** | **0.4071** (0.5585) |
| Dev, rest | 1,827 | 0.6191 | 0.7105 | 0.6199 (0.6178) |
| Eval, one-sided missing reach | 94 | 0.5169 | 0.8627 | **0.3481** (0.5169) |
| Eval, rest | 4,542 | 0.6081 | 0.6734 | 0.6082 (0.6081) |

In the eval window, when fighter 1's reach is missing (n=43), fighter 1 wins 4.7% of the
time. LightGBM says 2.5% and the market says 39.4%.

Paired event bootstrap (repo's `paired_bootstrap_by_event`, 10,000 resamples, seed 42) on
the eval window:

```
eval all    market_plus_model 4636  diff -0.0033 CI [-0.0069, +0.0006] p=0.097
eval clean  market_plus_model 4542  diff +0.0001 CI [-0.0024, +0.0031] p=0.980
eval all    lightgbm          4636  diff +0.0710 CI [+0.0548, +0.0887]
eval clean  lightgbm          4542  diff +0.0653 CI [+0.0529, +0.0794]
```

What this means for each claim:

- **The headline survives.** The model loses to the market with or without the leak. In
  the eval era the leak actually hurts LightGBM on the "fighter 2 missing" side: the
  dev-era pattern no longer holds as strongly, and the model is overconfident (0.881
  predicted against 0.686 realised).
- **The market+model variant has no real signal.** Its whole near-significant eval
  improvement, and its −0.022 dev improvement (which excluded zero), come from leaked
  rows. On clean rows the features add **nothing** beyond the close.
- The 2015 by-year row, where LightGBM (0.5925) beats the market (0.6319), is a leak
  artefact.
- The parlay analysis is affected only marginally. 111 of 1,262 staked LightGBM parlays
  include a leaky leg. Excluding them, realised ROI is −0.206 against −0.226 overall, so
  the conclusion is unchanged.

**Fix.**

1. Drop the leak at the source. Either remove `reach`, `height` and `age` entirely, or,
   better, neutralise missingness: when either side is missing, set both sides and the
   diff to NaN (a symmetric mask, so one-sided NaN cannot occur). Also add explicit
   "attribute unknown" handling that does not depend on which side is missing.
   Alternatively, impute a missing reach from height (reach ≈ height plus noise) using a
   regression fitted on the training slice only, and drop the raw missingness.
2. Rebuild features and re-tune in the dev window (see finding 3). Then re-run the
   backtest **once**: that is eval run #2, and the README should state why.
3. Rewrite the README's market+model paragraph and the dev-window numbers, and add a
   limitations bullet on scrape-time attributes.

## Finding 2 — MEDIUM: the leakage suite is structurally blind to the `fighters` table

**Mechanism.**

- The mutation tests (`test_leakage.py:83-160, 163-183`) change future `fights` and
  `fight_stats` rows and assert the past is unchanged. A static, scrape-time attribute
  never changes under that mutation, so a leak through it is invisible.
- The shuffle test (`test_leakage.py:470-530`) randomises `winner_id` but passes the real
  `fighters` table. Reach missingness correlates with the *real* outcomes, not the
  shuffled ones, so the shuffle test cannot fire on it either.
- The "real control" upper bound (AUC < 0.80) is too loose to catch it.
- The column-name tests (`test_no_outcome_derived_column_reaches_the_feature_set`) are
  name-based and cannot see value-level leakage.

**Empirical check.** `uv run pytest -q -m leakage` gives `27 passed, 244 deselected in
106.00s`, with the leak in finding 1 live.

**Fix.** Add a behavioural test for static attributes. For example, inside the dev window,
for each of `reach`, `height` and `age`, compute the win rate of the side with a missing
value among one-side-missing bouts, and assert it lies in [0.35, 0.65] (or that a
logistic regression of label on the six missingness indicators has AUC < 0.55). More
generally, assert that a model trained only on the `fighters`-derived columns cannot
beat chance by more than a tolerance. Height, reach and DOB carry little real signal
beyond Elo; a large lift means leakage.

## Finding 3 — MEDIUM: hyperparameters and leakage diagnostics were selected on a leaky objective

**Mechanism.** `tune.py` uses a chronological holdout (2014-03-23 → 2015-12-19, 860 bouts).
That holdout is correctly confined to the dev window, but 150 of its 860 rows (17.4%)
carry the near-deterministic reach leak. The training rows up to 2015 contain 701 leaky
bouts out of 3,441. The 216-candidate search therefore optimised partly for how well each
configuration exploits missingness. The frozen config (`config.yaml:84-110`, validation
LL 0.6147) is not tuned for the honest problem.

The README's leakage table (`README.md:247`, "Real (control) AUC 0.6919") is largely the
same leak.

**Empirical check** (dev window only, same 70/30 split as `_dev_only_split`):

```
all features                          AUC 0.6920 LL 0.6211
  scored on leak-free test rows only  AUC 0.5882 LL 0.6830 (n=860)
reach missingness neutralised (both)  AUC 0.6147 LL 0.6751
```

**Fix.** After fixing finding 1, re-run `models.tune` (dev only), re-freeze the config,
regenerate the model cards and the shuffle/control table, and then do the single eval
re-run.

## Finding 4 — LOW: stacker refit can include same-date labels

**Mechanism.** In `run_walk_forward`, `history` is appended after every event
(`walkforward.py:198-205`). The stacker is refit from `history` on refit events
(`:154`). LightGBM's training filter is `date < event_date`, but `history` has no date
filter. When two scorable events share a date, the second event's refit sees the first
event's outcomes.

**Empirical check.** There are 5 dates with two scored events. Fold 14 (cutoff 2014-08-23,
`first_event_id` 770b9d4813c25902) is a refit on the second event of such a date, so its
stacker includes about 10 same-day labels. This affects the dev window only, and the
effect on a 2-coefficient logistic regression fitted on more than 1,000 rows is
negligible.

**Fix.** `_fit_stacker([h for h in history if h["date"] < event_date], cfg)`, plus a test
that builds two same-date events and asserts the stacker does not change when the first
event's labels are flipped.

## Finding 5 — LOW: cutoff tests trust the engine's self-report

**Mechanism.** `test_training_cutoff_is_respected_for_every_scored_event` compares each
prediction's date to `folds.train_date_max`, which the engine computes from the same
`train_sorted` it claims to fit. The in-engine guard `train["date"].max() >= event_date`
(`walkforward.py:124-133`) cannot fail, given the line above it. Neither would catch a
regression such as fitting on `panel` instead of `train_sorted`, or a model object reused
across a cadence boundary incorrectly.

**Empirical check.** No regression is present: dev predictions reproduce bit-exactly from
HEAD (see finding 6). The weakness is in the test, not in the current code.

**Fix.** Add a poison test. Append sentinel rows dated **on** each predicted event (and
after it) with extreme feature values and flipped labels, run the dev walk-forward, and
assert every prediction is byte-identical to the unpoisoned run.

## Finding 6 — LOW: eval-ledger and parlay provenance

- The only ledger entry (`eval_runs.json`, run 1, 2026-10-10T11:47:52Z = 04:47 PDT) is
  stamped `1c1e1f0`. The walk-forward engine, metrics, report and the
  `min_train_fights` / `min_validation_rows` fix were committed later, at 04:54–04:55
  (`d3ba0ca`, `4ebb622`, `ab82faf`, `496a8e4`). So the recorded SHA does not identify the
  code that produced the numbers.
- **Mitigating evidence:** a dev-only re-run from HEAD
  (`run_walk_forward(cfg, through=dev_end)`) reproduces all 2,154 committed dev
  predictions for every model to within 1.1e-16, and fold 0 early-stops at 183 trees.
  That is consistent with the eval run having used the post-fix config, as the
  `ab82faf` message claims.
- The ledger is a self-maintained JSON file. Deleting it resets the count, and it was
  first committed in the same commit that published the result, so "1 eval run" cannot
  be independently verified.
- `make parlay` scores the eval window repeatedly and records nothing. `max_bet_fraction`,
  `ruin_threshold`, `top_n_per_card` and `bankroll_bootstrap_resamples` were introduced
  in `3de8116`, after the eval result was public. The parlay conclusion is negative and
  robust, so this is a hygiene issue, not a bias in the claimed direction.

**Fix.**

- Refuse to append to the ledger when `git status --porcelain` is non-empty for `src/`
  or `config/`, and record a hash of `config.yaml`.
- Ledger `make parlay` runs too.
- Note in the README that sim settings were chosen after the eval run.

## Findings 7–11 — INFO

- **7.** `lgbm.py:122-160`: the validation tail chooses the tree count *and* fits the
  isotonic calibrator. That double-dips within the training window (mildly optimistic
  calibration) but never crosses the cutoff. A cleaner design would split the tail in
  two.
- **8.** `elo.py:75-80`: with two fights on one date, `ratings[f] = r_snapshot ± delta`
  makes the second update overwrite the first (67 fighter-days, all 1990s). This is a
  correctness issue, not leakage.
- **9.** 57 overturned results are treated as NC from the moment of the fight, although
  the overturn came later. The effect is "no update" rather than injected information, so
  it is negligible.
- **10.** In the demo, `model.joblib` is fitted on all decisive fights. The "they have met"
  branch only matches priced, walk-forward bouts, so a pair that met in an unpriced bout
  is shown as a "hypothetical, never fought" forecast from a model that trained on that
  bout. The fix is to match on the full `fights` table.
- **11.** Stakes are sized at closing odds, and same-card bets compound sequentially, as
  if settled between bouts. Both are slightly flattering; neither changes a −80% result.

---

## What was checked and found clean

- **Per-fight history is strictly prior.** State is snapshotted per date and folded in
  after emission (`pit.py:310-353`). Same-card and same-night (tournament) isolation hold.
  A fight's own round stats never enter its own row. Re-verified by running the suite:
  27/27 pass.
- **Elo** is pre-fight, rated from a per-date snapshot with updates applied after the
  date.
- **Orientation.** Canonical order is by sorted `fighter_id` (`normalize.py:301-305`).
  Label base rate is 0.5006 overall and 0.4929 on priced eval bouts. Synthetic `name:`
  ids always sort into slot 2, but only 27 bouts are affected (label mean 0.52): no
  signal.
- **Odds.**
  - Only 5 whitelisted columns are loaded, and the loader raises on extras (`odds.py:91-99`).
  - De-vig is two-sided.
  - Odds date equals fight date in 100% of joined rows.
  - Orientation is sane: market LL is 0.607 for exact matches and 0.624 for alias matches,
    and the favourite wins 65–67% in both, so there is no flipped-orientation signature.
  - No odds-derived column is in `FEATURE_COLS`, and the market column reaches only
    `MarketModel` and the stacker.
- **Outcome fields** (`method`, `finish_round`, `finish_time`, `winner_id`,
  `is_decisive`) are used only for history and labels, never as same-fight features.
  `title_bout` and `scheduled_rounds` are pre-fight facts.
- **Walk-forward.**
  - LightGBM, Elo and the market are fitted on `date < event_date` only.
  - Early stopping and calibration use the chronological tail of that slice.
  - No scaler or imputer is fitted globally (LightGBM uses NaN natively; Elo-LR is
    one-dimensional and per fold).
  - Retrain staleness biases against the model.
  - All 62 folds early-stopped (159–507 trees) and are calibrated.
- **Tuning confinement.** `tune.py` and `cards.py` filter to `date <= dev_end` and raise
  otherwise. The git history shows the hyperparameters were frozen (`76ffba7`) before the
  walk-forward or eval code existed.
- **Bootstrap.**
  - The resampling unit is the event, and the model-vs-market difference is paired per
    fight (`metrics.py:107-170`).
  - The bankroll bootstrap is event-level, and ruined paths are held rather than dropped.
  - The p-value is a percentile approximation, which is fine for reporting.
- **Parlay sim.**
  - Candidates and ranking use model probability × book price only.
  - `won` / `realized_return` are used solely for scoring after selection.
  - The market control finds 0 positive-EV parlays, as it should.
- **CI** builds data before testing, runs slow and leakage tests, and fails on any
  skipped leakage test (`ci.yml`).
- **Determinism.** Dev predictions reproduce bit-exactly from HEAD.

## Commands run (all read-only, or dev-window-only)

- `uv run pytest -q -m leakage -p no:cacheprovider` gives 27 passed.
- Pandas probes on `data/processed/features.parquet`, `odds.parquet` and `fights.parquet`
  and on `reports/walkforward_predictions.csv` and `reports/parlay_top_*.csv`
  (missingness × outcome by era, subset log losses, parlay leg overlap, same-date events).
- `paired_bootstrap_by_event` on the committed eval predictions, split by leak subset (no
  re-fitting, no ledger write).
- `run_walk_forward(cfg, through=cfg.windows.dev_end)` compared against the committed dev
  rows, giving a max abs diff of 1.1e-16 (no ledger write, no files written).
- Dev-only LightGBM ablation on the 70/30 dev split, with and without reach-missingness
  neutralised.
- `git log -p -- config/` and commit timestamps compared against the ledger timestamp.
