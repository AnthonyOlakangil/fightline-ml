# Adversarial review 2: the leak fix and evaluation run 2

Reviewer scope: whether the fix for review 1's HIGH finding (scrape-time height, reach and
age) is complete, whether evaluation run 2 is honestly provenanced, and whether the new
numbers and the README's claims about them hold up. Reviewed at HEAD `8470af5`. No file
was modified except this one, and nothing was committed. `make backtest` was **not**
run, and `reports/eval_runs.json` was not touched. Every computation below either reads
committed artefacts (`reports/*.csv`, `*.json`, git history) or runs inside the dev
window only. A scratch worktree of `9fbe2b8` was created under `/private/tmp` and has
been removed.

## Summary

| # | Severity | Finding | Location |
|---|---|---|---|
| 1 | **MEDIUM** | The market+model variant's eval "win" (−0.0009, CI upper −0.00004) comes entirely from **recalibrating the market itself**. A walk-forward stacker on `logit(p_market)` alone gets −0.00086 [−0.00157, −0.00015]. Adding LightGBM to it changes log loss by −0.000013 [−0.00036, +0.00037]. Despite this, the README bolds **"beats the market"** and calls the result evidence that "the features carry information the market has not already priced". | `backtest/readme_block.py:95-99, 128-133`; `backtest/metrics.py:170-171`; `README.md:310, 315` |
| 2 | **MEDIUM** | LightGBM outputs **literal certainties**. The isotonic calibrator (`y_min=0, y_max=1`, then clipped to 1e-6) gives p ≤ 1e-5 or ≥ 1−1e-5 on 25 dev and 26 eval fights. It is wrong on 9 and 6 of them, at 13.8 nats each. That alone makes dev LightGBM "worse than a coin flip": 0.7370, or 0.6873 without those rows, or 0.6823 uncalibrated. It also triggers **25 of the variant's 29 staked parlays**. The README blames a different mechanism. The old hyperparameters show the same problem, so the re-tune did not cause it; the leak hid it in run 1. | `models/lgbm.py:143-150`, `models/base.py:23`; `README.md:317, 764-769` |
| 3 | LOW | The parlay tables put "Staked 29" next to EV, ROI and hit rate computed over **55** top-ranked parlays, 26 of which were never staked. ROI on the 29 staked bets is +21.8% per unit, or +11.7% on bankroll turnover; the published +9.6% belongs to the 55. Likewise LightGBM's "Staked 1,261" sits beside a bankroll replay that placed 250 bets. | `backtest/readme_block.py:279-286`; `backtest/readme_sections.py:227-237`; `README.md:377` |
| 4 | LOW | The re-tune made the shuffle test's log-loss check vacuous. The real-outcome control now scores 0.7083, which is also above the 0.68 threshold, so only AUC separates real from shuffled outcomes. The hand-typed README still says the test "asserts on log loss". | `tests/test_leakage.py:503-527`; `features/diagnostics.py:66-74`; `README.md:268-270` |
| 5 | LOW | The two new regression tests have gaps. Test 1 blanks attribute *columns* passed to `build_features`; it does not cover fighters-table *rows*, attributes routed through `build_fights`, or sentinel encodings. Test 2 passes a symmetric-blanking variant (verified). Together they do catch both run 1 and symmetric blanking. | `tests/test_leakage.py:386-447`; `features/pit.py:317-334` |
| 6 | LOW | The README understates run 1's provenance gap. It says only `min_validation_rows` was committed late. In fact `1c1e1f0` contains **no backtest code at all**: the walk-forward, metrics/bootstrap and report modules were first committed 7 minutes after the run. | `README.md:445-447` |
| 7 | LOW | The clean-tree guard ignores derived inputs. `data/processed/*.parquet` is gitignored, so `ufc-predictor backtest` run without `make` would ledger "clean" on stale features. This did **not** happen for run 2. Separately, the parlay pricing at 22:37Z ran with an uncommitted `sim/pipeline.py` change (metadata only), and parlay runs are still not ledgered. | `backtest/report.py:52-66, 316-326`; `sim/pipeline.py:138-144` |
| 8 | LOW | Stale or hand-typed claims: "no evaluation-window number appears anywhere in this repository…"; "Glimpse 2 is the one that matters"; "run 2 differs from run 1 only by the leak fix" (the hyperparameters changed too); the hand-typed "126 → 117", "seven minutes" and "2014-08-23"; a templated tie-rule sentence whose condition doesn't test what it states; and a CI bound rendered as "−0.0000". | `README.md:269, 429, 631-633, 697, 772-773`; `backtest/readme_sections.py:139-145` |
| 9 | INFO | The re-tune's gain on the tuning split (0.6775 vs 0.6825) does not transfer to the dev walk-forward (0.7370 new vs 0.7356 old calibrated; 0.6823 vs 0.6798 uncalibrated). Tuning optimises an uncalibrated single split, but the backtest deploys per-fold isotonic calibration. This is dev only, with no eval contamination. | `models/tune.py:90-102` |
| 10 | INFO | The fighters table still decides *identity* upstream. 56 bouts are dropped because a 2026-roster namesake exists, and 29 bouts (2023–26, all slot 2) carry name-derived ids. Neither reaches a feature value or carries outcome information. | `ingest/normalize.py:194-273` |
| 11 | INFO | Proportional de-vig leaves a favourite–longshot bias. A walk-forward logistic recalibration of the close (slope 1.04–1.11) beats the "vig-free market" by 0.00086 (p≈0.017) with no model at all. | `ingest/odds.py` (de-vig), `backtest/walkforward.py:224-244` |
| 12 | INFO | Review-1 findings 4 and 5 are unchanged. The same-date stacker refit is still confined to dev fold 14 (re-verified on run 2). The JSON→README numbers are test-enforced, but CSV→JSON is not; it was recomputed here and matches. `hyperparameters_frozen_from_dev: True` and `total_priced_decisive: 6790` are literals in `report.py:347, 411`. | `backtest/walkforward.py:154-166`; `backtest/report.py` |

**Bottom line.**

- **The leak fix is complete.** The feature matrix is now byte-identical whether the
  fighters table is intact, has blanked attributes, or is empty. Nothing scrape-time
  reaches the model, the demo or the predict path.
- **Run 2's provenance holds up** against git, the reflogs, CI timing and file mtimes.
  - The settings in `metrics.json` are exactly those committed in `75bab69`.
  - That commit was pushed at 22:27:18Z, and CI was green at 22:32:30Z.
  - Features were rebuilt at 22:32:46Z and the run was ledgered at 22:33:27Z.
  - Market, Elo, odds, folds and cutoffs are bit-identical between runs 1 and 2. Only
    LightGBM and the variant moved.
- **The re-tune reproduces exactly** from leak-free dev data.
- **The headline numbers recompute independently.**
- **The headline (the market beats the model) is robust.**

What needs correcting is the interpretation of the two secondary results:

- **The variant's "beats the market" is the market recalibrating itself**, not the
  features adding information (finding 1).
- **"Worse than a coin flip" and the variant's parlay activity** are both driven by a
  calibrator that emits 0 and 1 (finding 2).

---

## Finding 1 — MEDIUM: the variant's eval improvement is market recalibration, not model information

**Mechanism.** The stacker (`walkforward.py:224-244`) fits
`LogisticRegression([logit p_market, logit p_lightgbm])` with free slopes and an
intercept. Its improvement over the market can therefore come from either of two
sources:

- re-scaling the market's own probabilities; or
- information in LightGBM.

The README attributes it to the second: "it is evidence about whether the features
carry information the market has not already priced" (`readme_block.py:128-133`,
`README.md:315`). The table verdict reads **beats the market** in bold
(`readme_block.py:95-99`, `README.md:310`). `metrics.json` records `beats_market: true`
with the interpretation "negative means the model beats the market; the interval
excludes zero" (`metrics.py:170-171`).

**Empirical check.** From the committed `reports/walkforward_predictions.csv`, no
refitting of any model. I re-implemented the stacker on the same refit schedule (the
first event of each fold, history from all earlier predicted events). The
re-implementation reproduces the committed `p_market_plus_model` to within 1.3e-15. I
then fitted a **market-only** stacker on the same schedule. Paired event bootstrap,
20,000 resamples, my own code and seed:

```
eval (4,636 fights, 421 events)        diff vs market   95% CI                 p
  market + LightGBM  (committed)       -0.000873        [-0.001694, -0.000050]  0.038
  market only (recalibration)          -0.000859        [-0.001565, -0.000149]  0.017
  market + Elo                         -0.000778        [-0.001542, -0.000005]  0.048
  (market+LightGBM) - (market only)    -0.000013        [-0.000363, +0.000371]  0.92
fitted coefficients (2011 -> 2026): b_market 1.04-1.11, b_lightgbm -0.03..+0.027
dev (1,726 fights): variant +0.00107, market-only +0.00088, difference +0.00019 [-0.00018, +0.00068]
```

LightGBM adds nothing measurable beyond a recalibration of the close: its coefficient
is about 0.02 on the logit scale. The whole −0.0009 is the market being slightly
under-confident after proportional de-vigging (see finding 11). For contrast, in run 1
the variant's advantage over market-only was −0.0024 (eval) and −0.0229 (dev,
excluding zero). That was the leak, consistent with review 1.

**Fix.**

1. Add a `market_recalibrated` control: the same stacker on `logit(p_market)` only, with
   the same refit schedule. Report the variant against it, not only against the raw
   market. It is computed from stored out-of-sample predictions, so it needs no third
   scored run, but disclose it as a further eval-window look.
2. Never render a bold **beats the market** for a market-consuming variant. Use
   "improves on the raw close". Replace the "evidence about whether the features carry
   information" sentence with the control's result: "a market-only recalibration
   achieves the same; the features add −0.00001 [−0.0004, +0.0004]".
3. Derive `beats_market` / `interpretation` for the variant differently in `metrics.py`.

## Finding 2 — MEDIUM: isotonic calibration emits probabilities of 0 and 1

**Mechanism.** `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` is
fitted on the fold's chronological tail (`lgbm.py:143-150`). When the extreme score
bin of that tail is all one class, any later fight scoring beyond it gets exactly 0
or 1. `PROB_EPS = 1e-6` (`base.py:23`) turns that into a 13.8-nat penalty when wrong.
Early stopping at 8–122 trees (median 47.5, lr 0.05) gives the trees few distinct
scores, which makes the edge bins coarse.

**Empirical check** (committed predictions):

```
          p<=1e-5 or >=1-1e-5   wrong   share of total loss   LL all   LL excl. them   market excl.
dev                25              9          7.8%            0.7370      0.6873          0.6093
eval               26              6          2.6%            0.6916      0.6775          0.6064
```

Dev-only walk-forward (`run_walk_forward(cfg, through=dev_end)`, no files written):

```
new (frozen) params                LGBM dev LL 0.7370  (25 degenerate)   reproduces committed dev rows to 1.1e-16
old (run-1) params                 LGBM dev LL 0.7356  (23 degenerate)
new params, calibrate=False        LGBM dev LL 0.6823  (0 degenerate)
old params, calibrate=False        LGBM dev LL 0.6798  (0 degenerate)
```

So the dev "worse than a coin flip" is the calibrator, not the re-tune. It was present
in run 1's code too, masked by the leak. A 7-second dev-only dry run would have shown
it before run 2 was spent.

The same outputs drive the variant's parlays. Of the 29 staked variant parlays, **25
contain a leg where LightGBM said p ≥ 0.99999**. A stacker coefficient of about 0.025
times a logit of 13.8 shifts the market by about +0.35 logit, which manufactures the
"edge". Those 25 returned +26.1% per unit and the other 4 returned −4.7%. The sample is
too small to mean anything.

The README's explanation is directionally right ("too confident, not inverted") but
gets the mechanism wrong. Limitations (`README.md:764-769`) blames sharing the tail
between early stopping and calibration (review-1 finding 7). That sharing does not
produce exact 0/1 outputs; the unbounded isotonic ends do. The README also never says
in prose that eval LightGBM (0.6916) is barely better than a coin flip (0.6931) and
worse than one-feature Elo (0.6849). The tables show it, but the prose does not.

The headline is unaffected: excluding those rows, LightGBM scores 0.6775 against the
market's 0.6064.

**Fix.**

- Bound the calibrator, for example `y_min=0.01, y_max=0.99`, or use Platt scaling or a
  shrunk isotonic.
- Add a test that no walk-forward prediction is within 1e-3 of 0 or 1.
- Correct the Limitations text and the generated "too confident" bullet. Add a generated
  sentence comparing eval LightGBM with ln 2 and with Elo.
- Note in the parlay section that the variant's stakes come from degenerate LightGBM
  legs.
- Any fix to the regime is a run-3 decision and must be disclosed as such.

## Finding 3 — LOW: parlay "Staked" column and ROI use different denominators

**Mechanism.** `render_parlay` prints `n_staked` (bets that cleared `min_edge`) beside
EV, ROI and hit rate taken from `predicted_vs_realized["all_top_ranked"]`, which is
every top-3-per-card candidate (`readme_block.py:279-286`). The LEAK table does the
same (`readme_sections.py:227-237`).

**Empirical check** (`reports/parlay_metrics.json`):

```
market_plus_model  n_staked 29 | all_top_ranked n 55  ROI +0.0963  EV +0.0357
                               | cleared n 29         ROI +0.2183  EV +0.0699 | bankroll n_bets 29, ROI on turnover +0.1172
lightgbm           n_staked 1261 | all_top_ranked n 1261 ROI -0.5057 | bankroll n_bets 250
run 1 variant      n_staked 1124 | all_top_ranked n 1196 ROI -0.0732 | cleared ROI -0.0796
```

The error understates the variant, so it is not self-serving. But "29 bets, +9.6% ROI"
is not a true statement about any single set of bets.

**Fix.** Show the `cleared_stake_threshold` block next to `n_staked`, or label the
columns "top-ranked (n)" and "staked (n)" separately. State that the LightGBM bankroll
replay placed 250 of the 1,261 qualifying bets, and why.

## Finding 4 — LOW: the shuffle test's log-loss assertion no longer discriminates

**Mechanism.**

- `diagnostics.fit_predict_dev` uses the frozen `cfg.lightgbm` with a fixed 200 trees.
  After the re-tune (lr 0.05, 31 leaves, depth 6) that overfits: early stopping picks
  8–122 trees.
- The real-outcome control now scores log loss **0.7083**, worse than chance
  (`reports/leakage_diagnostics.json`).
- The shuffle test asserts `min(lls) > 0.68` (`test_leakage.py:521`). A model trained
  on *real* outcomes would therefore pass the "cannot beat chance" check as well.
- Only the AUC band (mean in [0.46, 0.54], against a control of 0.586) still separates
  the two.

The generated DIAGNOSTICS block admits this. The hand-typed `README.md:268-270` still
says the test "averages three independent shuffles and asserts on log loss".

**Fix.** Pin the diagnostic model to fixed, conservative settings independent of the
tuned config (or use early stopping). Assert that shuffled log loss exceeds the
control's by a margin, rather than an absolute 0.68. Update the prose.

## Finding 5 — LOW: regression-test blind spots

**Empirical check.** I ran HEAD's `tests/test_leakage.py` against `9fbe2b8`, the pre-fix
commit. Its `features/` and `ingest/` are identical to run 1's code at `29df404`.

```
pre-fix code:      2 failed
  test_features_ignore_the_scrape_time_fighter_attributes:
    "features read the present-day fighters snapshot: ['height_1', ..., 'age_diff']"
  test_a_blank_feature_never_predicts_who_loses:
    "reach: blank side won 0.057 of 334 bouts"
symmetric-blanking variant (both sides + diff NaN when either is blank):
  test_features_ignore_... FAILED;  test_a_blank_feature_... PASSED
HEAD: both pass; CI ran both with no skips at 75bab69 and 8470af5 (27 leakage tests passed)
```

So the pair catches run 1 and symmetric blanking, but each test alone is narrow:

- **Test 1 checks columns, not rows.** It blanks attribute columns only. A feature built
  from row presence (`fid in fighters`) would pass. Today the features are invariant to
  an *empty* fighters table as well (verified), so this is latent.
- **Test 1 covers only `build_features`.** It never rebuilds `fights`, so an attribute
  joined in `normalize.build_fights` would pass.
- **Test 2 sees only NaN.** It ignores symmetric NaN, sentinel values and booleans.
- **Test 2's band is loose.** At 0.30–0.70 it sits about 5 SE either side of the 0.424
  debut baseline; current rates are 0.42–0.46.

**Fix.**

- Remove the unused `fighters` parameter from `build_features` (`pit.py:317-334`) so the
  signature makes reading the table impossible.
- Add an invariance test with `fighters.iloc[0:0]`.
- Add a pipeline-level test that rebuilds `fights` from a blanked fighters table.
- Tighten the canary band to about [0.35, 0.65].

## Finding 6 — LOW: run-1 provenance is understated

**Empirical check.**

- `git ls-tree -r 1c1e1f0 src/ufc_predictor/backtest/` lists only `__init__.py`.
- `walkforward.py`, `metrics.py` (including the bootstrap) and `report.py` first appear
  in `d3ba0ca`, `4ebb622` and `496a8e4` (04:54–04:55 PDT). Run 1 was ledgered at
  04:47:52 PDT.

`README.md:445-447` says only that `min_validation_rows` was committed seven minutes
later.

**Fix.** Say that the entire backtest and scoring code was uncommitted at the stamped
SHA. Note the mitigating evidence: dev predictions reproduce, and market, Elo and fold
columns are bit-identical between runs.

## Finding 7 — LOW: the guard does not cover derived data or parlay runs

**Mechanism.** `_git_state()` uses `git status --porcelain`, which ignores the gitignored
`data/processed/`. `report.run` reads whatever `features.parquet` exists. `make
backtest` rebuilds it first, but `uv run ufc-predictor backtest` does not.

**Empirical check for run 2.** These all exclude a problem in run 2:

- `features.parquet` mtime is 15:32:46 PDT, 41 s before the ledger write.
- Its 117 columns equal a fresh `build_features` at HEAD (`pit.py` is unchanged since
  `fa76c42`).
- Dev predictions reproduce to 1.1e-16.

**Parlay pricing.** `parlay_metrics.json` (15:37:48 PDT) contains `priced_eval_run`,
which was added in `sim/pipeline.py` and committed only at 15:40:49 in `b12ea61`. So
that pricing ran from a dirty tree; the change is metadata only. Parlay runs still
record no commit or clean flag.

**Fix.** Rebuild or hash the processed tables inside `report.run`, and ledger the hashes.
Apply the same clean-tree and ledger rules to `make parlay`.

## Finding 8 — LOW: stale or hand-typed README text

- **`README.md:631-633`** says "no evaluation-window number appears anywhere in this
  repository until the walk-forward backtest generates it". This contradicts the
  glimpse list (0.6489) and the eval re-scores in `reports/review.md` and in this file.
- **`README.md:697`**: "Glimpse 2 is the one that matters" was written for run 1.
  Before the re-tune and run 2, the authors had seen run 1 and review 1's clean-subset
  eval figures (LightGBM +0.0653, variant +0.0001).
  - Risk is low: I reproduced the re-tune exactly, and the fix choice used pre-2016 data
    only (see "Clean").
  - Still, the paragraph should say this.
- **`README.md:429` and `772-773`**: "Nothing else changed" and "run 2 differs from run 1
  only by the leak fix". The hyperparameters also changed. In practice the re-tune is
  close to neutral in the dev walk-forward (finding 9), but the sentences should name
  both changes.
- **Hand-typed figures** outside the generated blocks: "126 → 117 features"
  (`README.md:426, 178`), "seven minutes" (`:447`), "2014-08-23" (`:775`). They are
  correct today but are not generated or tested.
- **`readme_sections.py:139-145`** prints "the tied candidates had equal capacity"
  whenever `selected == lowest`. That condition does not test capacity. The sentence is
  true here only because the two tied candidates differ just in `learning_rate`, which
  `capacity_rank` ignores.
- **The variant's CI upper bound** (−0.000045) renders as "−0.0000"
  (`README.md:310, 315`). Show one more decimal or "< 0".

## Finding 9 — INFO: the re-tune gain does not replicate in the dev walk-forward

On the tuning split, the selected settings beat the old ones by 0.0050. In the dev
walk-forward they are 0.0014 worse calibrated (0.7370 vs 0.7356) and 0.0025 worse
uncalibrated (0.6823 vs 0.6798). The tuning objective (`calibrate=False`, one split,
`tune.py:90-102`) is not the deployed regime. This is dev only. It is not evidence of
eval contamination, but the selection is probably noise.

## Finding 10 — INFO: the fighters table still decides identity upstream

- **Ambiguous-name drops.** 56 bouts are dropped because a name maps to several ids in
  the 2026 roster. A later namesake therefore removes earlier bouts from opponents'
  histories. That is future information deciding *which* history exists, but it
  carries no outcome. `README.md:229-231` ("no future data is introduced", line 230) is
  slightly loose.
- **Synthetic ids.** 29 bouts (2023-12 → 2026-09) carry name-derived ids, always in slot
  2. Fighter 1 wins 0.519 of the decisive ones. No feature exposes the id.

## Finding 11 — INFO: the "vig-free" baseline keeps a favourite–longshot bias

The market-only recalibration in finding 1 has slope 1.04–1.11 and improves eval log
loss by 0.00086 (p≈0.017). Proportional two-sided normalisation is what CLAUDE.md
specifies, so this is not a defect. It is worth a sentence, though, and a Shin- or
power-de-vigged market would make a stronger robustness baseline.

## Finding 12 — INFO: carried over from review 1, and test coverage

- **Same-date stacker refit.** On run 2's predictions, the only refit whose history
  holds same-date labels is fold 14 (2014-08-23, 11 rows). The 2016-11-19 double-event
  date has no refit, so the README's "touches no evaluation fight" holds.
- **The self-referential cutoff test is unchanged.**
- **CSV→JSON is not test-enforced.** `test_report.py` and `test_readme_sections.py`
  enforce README == render(JSON) for every generated block, and `test_sim.py:423`
  covers the parlay block. No test recomputes `metrics.json` from
  `walkforward_predictions.csv`, or the shuffle figures in `leakage_diagnostics.json`.
- **Tautological literals.** `report.py:411` writes
  `hyperparameters_frozen_from_dev: True`, and `test_report.py:84` asserts it.
  `report.py:347` hard-codes `total_priced_decisive: 6790`.

---

## What was checked and found clean

- **The fix is complete at the feature level.**
  - `build_features` produces byte-identical output whether `fighters` has its five
    attribute columns blanked or is **completely empty** (`fighters.iloc[0:0]`).
  - The on-disk `features.parquet` (117 feature columns) equals a fresh build at HEAD.
  - No `height`, `reach`, `age` or `dob` column exists anywhere in the feature table.
  - The pair-level interaction route is closed, because the attributes are gone rather
    than masked.
- **Demo and predict path.**
  - `app/artifacts/model.joblib` has 117 `feature_cols`, none of them attributes.
  - `app/artifacts/fighters.parquet` has no attribute columns.
  - `export_app.py` reads the fighters table for `display_name` only.
  - `streamlit_app.build_matchup_row` uses the model's `feature_cols`.
  - `predict.py` fits on `date < event_date`.
- **Regression tests.**
  - Both fail on `9fbe2b8`, which has the same `features/` and `ingest/` as run 1's
    `29df404`, and both pass at HEAD.
  - CI runs them with no skips at `75bab69` and `8470af5`. The one skip at `75bab69` was
    `test_report.py:87`, legitimate while `metrics.json` was still run 1.
- **Run-2 provenance.**
  - `metrics.json` `walk_forward.lightgbm` and `lightgbm_fit` equal `config.yaml` at
    `75bab69`, and also at HEAD. The `backtest`, `elo`, `features`, `windows` and `seed`
    blocks are unchanged as well.
  - `75bab69` is an ancestor of HEAD.
  - Timeline from the reflogs, `gh run view` and file mtimes:

    | Time (UTC) | Event |
    |---|---|
    | 22:27:17 | `75bab69` committed (HEAD stays there until 22:40:49) |
    | 22:27:18 | pushed (`origin/main` reflog) |
    | 22:32:30 | CI success on `75bab69` |
    | 22:32:46 | features rebuilt |
    | 22:33:26 | predictions written |
    | 22:33:27 | ledger and metrics written |

  - The clean-tree guard is present at `75bab69` (it came in with `b224f94`).
  - No pushed history was rewritten. The Oct 10 local resets happened before those
    commits were pushed.
- **Nothing but the fix and the re-tune changed the model inputs or scoring between
  runs.**
  - `git diff 29df404 75bab69` leaves `walkforward.py`, `metrics.py`, `models/` (bar a
    comment in `tune.py`), `elo.py`, `ingest/` and `storage/` unchanged.
  - `config.yaml` changed only in the `lightgbm` block and comments. The `sim` additions
    predate run 1's parlay, and `sim` is unchanged since `e53fa55`.
  - `uv.lock` changed only to add a direct `joblib` entry.
  - Comparing run 1 (`721dc16`) with run 2 predictions: same 6,790 fights in the same
    order. `label`, `fold`, `train_cutoff`, `odds_1`, `odds_2`, `p_market` and `p_elo`
    are all **bit-identical**; only `p_lightgbm` and `p_market_plus_model` differ.
- **Re-tune confinement.**
  - `tune.search` re-run into a scratch `reports_dir`, on current leak-free features
    with the run-1 settings as baseline, reproduces `reports/tuning.json` **exactly**
    (80 s).
  - `dev_frame` filters to `date <= dev_end`.
  - `tuning.json` (22:20:29Z) predates the commit and the run.
- **The fix choice did not use eval data.** The commit message's "27.5% vs 47.7%"
  reproduces exactly on all labelled bouts dated ≤ 2015-12-31 (539 / 2,090 unequal-
  experience pairs). The diagnostics' 23.2% / 48.4% are the 2010–2015 subset. On eval
  years the effect would have been 42.7% vs 52.5%; it was not used.
- **Numbers.** My own paired event bootstrap (20,000 resamples, seed 20261012):

  | Comparison | Δ log loss | 95% CI | p |
  |---|---:|---|---:|
  | LightGBM vs market | +0.08528 | [+0.07059, +0.10170] | — |
  | Variant vs market | −0.000873 | [−0.001682, −0.000055] | ≈0.037 |
  | Elo vs market | +0.07864 | [+0.06913, +0.08808] | — |

  These match `metrics.json`: [+0.0703, +0.1020], [−0.00168, −0.000045] with p 0.038,
  and [+0.0694, +0.0879]. The log losses also match (0.6063, 0.6916, 0.6849, 0.6054),
  as does dev (0.6098, 0.7370, 0.6826, 0.6096 on n = 1,726).

  Every figure in the RESULTS, HEADLINE, DIAGNOSTICS, TUNING, DEVCARDS, LEAK and PARLAY
  blocks equals its JSON source. The test suites enforce this:
  `pytest tests/test_leakage.py tests/test_readme_sections.py tests/test_report.py` gives
  57 passed, and the tree was clean afterwards. Bonferroni (0.038 × 3 = 0.114) and
  "0.14% of the market's log loss" check out.
- **The run-1 archive is genuine.**
  - `reports/archive/eval_run_1/metrics.json` is byte-identical to `git show
    721dc16:reports/metrics.json`, which is the only version since `29df404`.
  - The archived `parlay_metrics.json` is byte-identical to `721dc16` (= `e53fa55`).
  - The archived headline equals ledger entry 1.
- **`diagnostics.py` is confined to the dev window.** `dev_only_split` and
  `attribute_leak_evidence` filter to `≤ dev_end` and raise otherwise. Features are built
  over all dates, which is point-in-time-safe, but nothing past `dev_end` is scored.
  `leakage_diagnostics.json` spans end at 2015-12-19 and 2015-12-31.
- **Elo** is unchanged, and so are its eval figures.

## Commands run (read-only, or dev-window-only)

- `git show/diff/log/ls-tree/reflog`, `git merge-base --is-ancestor 75bab69 HEAD`, and
  `gh run list` / `gh run view --log` for CI timing and skips.
- `git worktree add /private/tmp/claude-501/rev2/prefix 9fbe2b8`, with processed tables
  copied in. HEAD's `test_leakage.py` was run with `PYTHONPATH` set to the worktree
  `src`, once as-is and once with a symmetric-blanking patch to the worktree `pit.py`.
  The worktree was then removed.
- A pandas/numpy re-implementation of log loss, the paired event bootstrap and the
  stacker (plus market-only and market+Elo controls) on
  `reports/walkforward_predictions.csv` and on run 1's CSV from `721dc16`.
- `run_walk_forward(cfg', through=dev_end)` for new and old params, each with and
  without calibration. It writes no files and asserts `date <= dev_end`.
- `tune.search(cfg')` with `reports_dir` set to a scratch directory.
- `build_features` with blanked and with empty fighters tables, compared against
  `data/processed/features.parquet`.
- Probes of `parlay_metrics.json` and `parlay_top_*.csv`, matching legs back to
  `p_lightgbm`.
- `pytest -q tests/test_leakage.py tests/test_readme_sections.py tests/test_report.py`
  gives 57 passed.
