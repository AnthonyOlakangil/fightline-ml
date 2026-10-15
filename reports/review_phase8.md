# Phase 8 adversarial review: leakage and overfitting

**Reviewer:** a fresh-context subagent that wrote none of this code. **Date:** 2026-10-14.
**Scope:** `ea60952..7245f81`: the code, configuration, workflow, outputs and README
sections of Phase 8. The brief was data leakage and overfitting in the broad sense, plus
whether the README is honest about them. Style was out of scope.

**Rules followed:** I modified no tracked file. I ran no `experiments`, `confirm`,
`backtest` or anything else that appends to a ledger. My scripts are label-free, or limited
to 2010–2014, or limited to features. They scored nothing in 2015 or 2016+. The
scripts live in the session scratchpad (`rev/`), and the results are quoted below.

## Verdict

**No critical findings.**
- No Phase 8 code path scores, tunes on or selects with a 2016+ outcome.
- No selection decision used a 2015 outcome.
- The confirmation ran once, from the committed and pushed frozen configuration.
- The five new feature families stayed point-in-time under my own cross-family tests.

There is **one important finding**: the README's reading of the closing-line-value (CLV)
analysis draws a conclusion the data do not support. There are also **nine minor
findings**. Fixing I1 is required. M5, M3 and M8(a), (b) and (e) are strongly recommended.
Everything else can be fixed whenever convenient.

---

## Answers to the brief's questions

### 1. Evaluation-window leakage: can any Phase 8 path score, tune on or select with 2016+?

No.

- **Runner.**
  - `experiments/runner.py:252-254` filters scored fights to the requested window.
  - `:270-271` asserts that every fold's training rows end before the predicted date.
  - `:314` runs `assert_scorable` on every returned date. That function
    (`protocol.py:62-84`) raises `EvalWindowClosed` for any date on or after 2016-01-01.
  - Training pools use `date < event_date` (`runner.py:204`), and every event scored is
    dated 2015 or earlier.
- **Stages.**
  - `retune_split` cuts the panel at `selection_end` (`stages.py:271`) and asserts
    `purpose="selection"` (`:279`).
  - Every compare stage calls `run_config(window="selection")` (`:175`).
  - Every entry in `experiments.json` spans 2010-03-21 → 2014-12-20.
- **Confirmation.** `run_config(window="confirmation")` (`confirm.py:84-85`) covers
  2015-01-03 → 2015-12-19 only.
- **CLV.** `assert_scorable(..., purpose="dev")` (`analysis/clv.py:120`) runs on frozen
  development predictions only. `clv.json` spans 2010-03-21 → 2015-12-19.
- **BestFightOdds (BFO) join.**
  - Joined rows dated 2016 or later raise (`ingest/bfo_join.py:298-302`).
  - The snapshot ends 2015-12-19.
- **Forward test.**
  - It trains on 2016+ fights, which is allowed.
  - It scores only cards starting at least 24 hours after `frozen.yaml` was first
    committed (`forward/run.py:521-531`). Those are 2026-10-17 onwards, after `eval_end`
    (2026-10-03).
  - `predict` refuses a card less than 24 hours away (`run.py:278-282`), so no past card
    can acquire predictions to settle.
- **Ledgers.**
  - No Phase 8 module touches `eval_runs.json`.
  - The controller's fresh-clone check re-ran `make backtest` in a scratch clone. It
    reproduced run 3 exactly, so it was a replay rather than a new look, and nothing was
    committed. That is acceptable.

### 2. Selection and confirmation leakage

No selection decision used 2015.

- **Stage 0 holdout:** 2014-04-26 → 2014-12-20, trained on fights up to 2014-04-19
  (`stages.json` stage 0).
- **Final re-tune:** the same split.
- **Compare stages:** the selection window only.
- **Top-k selection:** runs on the fold's training rows only, and was not adopted.

**Git checks:**
- `config/experiments.yaml` was added in `906ac9c` and never changed. Its SHA-256,
  `750e780d…`, matches every ledger entry's `preregistration_hash`.
- All 15 entries in `experiments.json`:
  - come from `d9fa547`, with `worktree_clean: true` and an identical `config_hash`;
  - were recorded between 22:15:01Z and 22:18:14Z;
  - come from a commit descended from `906ac9c` (checked with `git merge-base --is-ancestor`).
- `7fffd20` contains outputs only, so the published code is the code that ran.
- `frozen.yaml` was first committed in `ccd58bf` at 16:01:02 −07:00.
- `confirmation.json` was committed in `9453fd1` at 16:01:55 −07:00, and its
  `git_commit` is `ccd58bf`.
- `frozen.yaml` equals `stages.json`'s `frozen_spec` and `baseline_spec`. So does the
  configuration recorded inside `confirmation.json`.
- The 53-second gap between freeze and confirmation is feasible. I timed the all-family
  feature build at 7.5 s, and both 2015 walk-forwards together take about 15 s.

The single residual 2015-scoring path is a CI test, not a decision (M3).

### 3. Point-in-time features

I wrote my own tests, `rev/pit_checks.py`, covering all 212 `ALL_FEATURE_COLS`:

- **Whole-future scramble that includes the cutoff date.** On or after D, I flipped every
  decisive winner, permuted method and method class, permuted every per-round stat, and
  added a synthetic same-night bout on D plus a synthetic later bout.
  - D = 2015-01-03: features of 3,050 fights dated ≤ D are unchanged, including the 11 on D.
  - D = 2012-08-11: features of 2,043 fights are unchanged, including the 10 on D.
- **Forward-path equivalence.** I rebuilt the last snapshot card (2026-10-03, 13 bouts) as
  `forward/cards.py` does: an unlabelled card with new ids and no stats, appended to the
  history before it. Its 212 features equal the full build's, byte for byte.
- `features_v2.parquet` on disk equals a fresh build.
- In the forward path:
  - ESPN results enter history only when dated strictly before the card
    (`cards.py:273-275`);
  - `sync_results` saves only cards with `snapshot_end < date < card.date`
    (`run.py:154`).

  So a card's own bouts, and other bouts on the same night, cannot reach its features.

### 4. Forward-test integrity

| Question | Answer |
|---|---|
| Late prediction | Prevented. `predict` checks the wall-clock lead (`run.py:278`); the schedule records `missed` and never predicts late (`schedule.py:40-47`). |
| Edited predictions | Detected. `unchanged_since_added` requires exactly one commit and a clean working copy (`ledger.py:65-71`). Unverified cards are left out of the pooled summary (`run.py:492`). |
| Replaced bouts | Never scored against a substitute. They match on competition id plus the normalized pair, fall back to the same pair, and otherwise are void (`settle.py:52-68`). The winner always comes from the result's own pair (`espn.py:167`). |
| Commit-time check | Uses the **committer** date (`ledger.py:56`), which can be forged. The Actions run id and URL are recorded in `meta.json`, `ledger.json` and the commit body, but settlement never checks them (M5). |
| Kalshi close price | Read from candles ending at or before the segment start (`kalshi.py:125-129`), so no in-play price. |

### 5. Overfitting and multiple comparisons

**The rule was applied as pre-registered.**
- Holm within each stage; I recomputed stage 5 as 0.245×2 = 0.490 and max(0.490, 0.286) = 0.490.
- The p-value is the one-sided share of bootstrap resamples ≥ 0.
- The adopt condition is "negative and below 0.10".
- Several candidates passing never occurred.
- All candidates in a stage scored the same 1,693 fights.

**Reporting.**
- Symmetry's CI includes 0 (p = 0.066), and the README does not claim significance. The
  procedure as a whole has a multiplicity caveat (M2).
- The confirmation's interval is described accurately as "too wide to certify a small gain".
- Stage comparisons are tilted toward the incumbent by the stage-0 tuning design (M1).
- The CLV reading is the one real overstatement (I1).

### 6. README honesty

Every P8_* block is generated. `tests/test_readme_sections.py` and
`tests/test_readme_phase8.py` pass; I ran 113 tests across the Phase 8 non-scoring suites,
and all passed. Problems in the hand-written prose are listed under I1 and M8.

### 7. Triage

See the last section.

---

## Findings

### Important

**I1. The CLV "Reading it" credits the model's error, but the measures it cites mostly reflect the favourite's price firming.** (README 733–738)

The README says the market "corrects *away* from the model ... its disagreements are ...
mostly error". Measures (b), the correlation of edge with move, and (c), the directional
mean, cannot support that sentence.

**Why they cannot:**
- The frozen model is strongly compressed toward 0.5. On 2010–2014 BFO fights, its fit
  against the opening price is p_model ≈ 0.383 + 0.243·open.
- Its edge therefore correlates −0.75 with how far the opening price is from 0.5.
- In 88% of the |e| > 5 pp fights, the model is simply siding with the underdog.
- Favourites' prices firm between open and close: mean move toward the opening favourite
  +0.0114; corr(open − 0.5, move) = +0.170.
- A forecaster with no information shows the same pattern more strongly. On the same
  1,548 fights from 2010–2014 (`rev/clv_confound.py`, which reads no labels):

| Forecaster | corr(edge, move) | Directional mean, \|e\| > 5 pp |
|---|---:|---:|
| Frozen model | −0.097 | −0.0087 |
| Constant 0.5 | −0.170 | −0.0126 |
| Open shrunk 75% toward 0.5 | −0.170 | −0.0133 |

- Holding the open fixed reverses the sign. The partial corr(edge, move | logit open) is
  +0.043, with a 95% event-bootstrap CI of [−0.002, +0.091]. This agrees with the
  pre-registered measure (d), +0.175 [−0.012, +0.405].

**Failure scenario.** A hiring reader takes away the claim "the model's disagreements carry
negative information". The data show only that there is no detectable CLV, and the
measure that controls for the open leans slightly toward the model.

**Fix.**
- Rewrite the paragraph: close beats open; (b) and (c) are dominated by favourite-firming
  combined with the model's compression, and a constant-0.5 forecaster scores worse on
  both; (d) is the informative test and its interval includes 0.
- Optionally add the constant-0.5 benchmark and the partial correlation to `clv.json` as
  generated quantities, clearly labelled **post hoc**. Leave the pre-registered (a)–(d)
  as they are.

### Minor

**M1. Stage-0 hyperparameters were tuned on 20.9% of the scored selection fights. This
tilts the compare stages and makes "out-of-sample" too strong.**
- The stage-0 holdout (2014-04-26 → 2014-12-20) contains 353 of the 1,693 scored fights.
  Those parameters were chosen for the **base** set, then held fixed for every candidate,
  as pre-registered (`experiments.yaml:35-37`).
- The comparisons therefore lean toward the incumbent in stages 1–3 and 6. They lean
  toward the re-tuned candidate in the final re-tune, which was rejected anyway.
- The 2010–2013 predictions also use parameters chosen on later 2014 outcomes.
- The bias favours rejection, so no false adoption follows.
- But two README phrases go too far:
  - "more columns for the trees to overfit" (687–689) is a causal guess;
  - "the frozen configuration's out-of-sample predictions" (717) describes 2010–2014
    predictions whose hyperparameters and symmetry were selected on those same outcomes.
- **Fix:** add a one-sentence caveat in each place. In a future phase, use nested tuning,
  or score stages only after the tuning holdout.

**M2. There is no multiplicity control across stages.**
- α = 0.10 one-sided per stage, with Holm only within a stage, over seven sequential
  decisions.
- Symmetry was adopted at p = 0.066. The 2015 result (−0.0125, p = 0.041) points the same
  way.
- **Fix:** one README sentence saying the family-wise error across stages is not controlled.

**M3. A CI test scores real 2015 outcomes with every candidate family.**
- `test_model_does_find_signal_on_real_outcomes` (`tests/test_leakage.py:605-624`) uses
  `diagnostics.dev_only_split`. That split is the last 30% of fights up to 2015-12-31,
  roughly 2013-10 → 2015-12.
- Phase 8 switched the test to `ALL_FEATURE_COLS` and every family. CI runs slow tests on
  every push.
- Only pass or fail is visible, with bounds of 0.55 and 0.80 on the AUC, so it cannot have
  informed a selection.
- But it is a 2015 path outside the protocol, and printing the AUC once would expose 2015
  performance.
- **Fix:** cap the split at `selection_end` when building non-base families, or call
  `assert_scorable(purpose="selection")` in the diagnostic.

**M4. The symmetric model's validation tail contains mirrors of its own training rows.**
- `SymmetricModel.fit` sorts by `["date", "_m"]` (`symmetric.py:97`).
- `LightGBMModel` then splits by position (`lgbm.py:126-128`), which falls inside a date
  block.
- Across the 16 selection folds, 1–10 validation rows (0.1–1.8%) are mirrors of training
  rows (`rev/sym_split.py`).
- Early stopping and calibration therefore see slightly contaminated rows. The effect is
  within the fold only, with no cross-time leakage, and if anything it hurts symmetry.
- **Fix:** split train and validation by date before mirroring, or move the boundary to a
  date edge.

**M5. The forward test's "commitment verified" can be forged, and the server-side
evidence is never checked.**
- `committed_before` reads `%cI` (`ledger.py:56`). A commit made with a backdated
  `GIT_COMMITTER_DATE` would be marked `verified: yes`.
- The README's "a second, server-side timestamp" (761–762) is true only as evidence a
  human could check.
- **Fix:** at settlement:
  - fetch `meta.json`'s `actions_run_id` from the GitHub API, and require
    `run_started_at` ≤ deadline, the expected repository and a scheduled or manually
    dispatched event;
  - require the adding commit's message to contain that run's URL;
  - otherwise set `verified=false`.
- Also check the commit time of `prices_close.json` before using it as the fallback close
  (`run.py:445-447`).

**M6. The forward provenance records a dirty tree for an innocent reason.**
- `meta.json` for 600060773 records `code_tree_clean: false`.
- `sync_results` (`run.py:289`) writes `data/raw/espn_results/600061541.json` before
  `git_state` (`run.py:326`), and that directory is not in `OUTPUT_DIRS`.
- Every card after a new results file will look like it ran uncommitted code.
- **Fix:** take `git_state` before `sync_results`, or ignore that directory for this check.

**M7. "Exactly once" for the confirmation rests on the file existing.**
- `confirm.py:58-59` checks only whether `confirmation.json` exists. A deleted file allows
  a silent local rerun. No ledger records the attempt.
- The preflight also does not check that `frozen.yaml` equals `stages.json`'s
  `frozen_spec`. They are equal here.
- The git evidence for this run is consistent (see question 2).
- **Fix:** append a `confirmation` entry to `experiments.json` before scoring and refuse if
  one exists, and assert the two specs are equal.

**M8. README accuracy.**
- **(a) Contradiction.** Limitations says "No live predictions ... no current odds feed is
  reachable" (1101–1102), but the forward test now commits live predictions with Kalshi
  prices.
- **(b) Overturned results.** "Overturned results ... it adds none" (1097–1098) is wrong.
  Using the final no-contest ruling at earlier dates is a small look-ahead: 57 bouts, 24 of
  them in 2010–2015. Say so.
- **(c) "Joined exactly."** P8_CLV says "Joined exactly (same date, same pair) to 2,049
  fights", but 29 of them were joined through the alias table (`bfo_join.json`).
- **(d) Hand-written facts.** The three upstream odds errors (740–745, 1092–1096) are typed
  by hand, and nothing in `reports/` generates them. I reproduced them: exactly 3 joined
  fights have |BFO close − dataset p1_market| > 0.10. Write that list into
  `bfo_join.json` and render it.
- **(e) Unqualified guard claim.** "Any Phase 8 scoring path that reaches 2016 raises"
  (614) needs "except the forward test, which scores only cards after the freeze".
- **(f) Unpaired comparison.** Each P8_FORWARD row puts the model's log loss over all
  scored bouts next to Kalshi's over only the bouts with a close price
  (`settle.py:95-107`). Show both on the paired subset.
- **(g) Pre-registered count not shown.** The stage-3 floor count is pre-registered as
  "reported", but it appears only in `stages.json` (`last_8_years`: 1 fold).

**M9. The CLV and stage outputs inherit the confirmation's selection.**
- No action is needed beyond M1's wording.
- The CLV model log loss (0.6799) re-reports the same predictions already published, so it
  is no new look at 2015.

---

## Triage of the ledger's deferred items and rulings

**Nothing deferred is must-fix for leakage.** The must-fix items come from this review
(I1), plus wording.

**7a**
- (4) Guard-message wording: acceptable.
- (5) Same-night EWM order by fight id: acceptable. Both bouts are folded after the date, so
  only later dates' weights are affected; this is a definition, not leakage.
- (6) Unbounded `adj` against the opponent's 64-bout deque: acceptable, moot.
- (1)–(3) were carried into 7b and verified done (task-7b-report S3.1/S3.2).

**7b**
- (2) No toy test for a fighter's own bout without control time: acceptable. The reviewer's
  brute force pins the behaviour, and the family was not adopted.
- (4) The overturned-result limitation: acceptable to leave the data alone. **The README
  wording must change** (M8b).

**7c**
- (1) Slack in the worked-example assertion: acceptable. An independent reimplementation
  matches to 1e-12.
- (2) `check_params` finiteness and booleans: acceptable.
- Ruling G2 on rating-deviation (RD) double counting: acceptable. The parameters are a
  priori and Glicko was not adopted.

**Task 8**
- Stale comment, private `_booster` read, pass-3 reading: acceptable.
- No spy test on `forward/run.py`'s `select_top_k`: acceptable while frozen `top_k` is null.
  That path receives `train` cut before the card date (`run.py:186-191`). Add the test if
  `top_k` is ever adopted.
- The K2 amendment was made after looking at gain ranks on training rows before 2012. Those
  were features fitted on labels, not scores, and top-k was rejected, so it is acceptable.

**Task 3**
- The reconciliation that cannot fail, the untested raises, the real-data test asserting
  `joined` rather than `joined_exact`, and the global `eval_start`: all acceptable. The
  eval guard still applies and the snapshot ends 2015-12-19. Fix the README wording in M8c.

**Documented limitations**
- Kalshi close at the segment start: acceptable, documented in README 758–760. It is a
  conservative choice for the market.
- ESPN title-bout rule: acceptable, documented.
- Rows after the snapshot have no statistics: acceptable, documented. It skews the inputs
  the forward model sees relative to its training data, but leaks nothing.

**Rulings**
- **Pre-registered before any run:** "several pass → lowest log loss", the stage-0 training
  set, the stage-3 floor and the `dev` purpose. All are in `experiments.yaml`, `906ac9c`.
- **Made before any selection-window score existed:** the mirror definitions, the 7b column
  counts, the K2 trim, symmetry orientations and Platt. The first ledger entry,
  22:15:01Z, comes after all of them.
- **Forward-test rulings:** freeze-based scope, noreply author, Kalshi matching and the ESPN
  date in US Eastern time. These are sound.
- **Experiments with no subagent active:** the clean ledger confirms this was followed.
- **The INCIDENT** (mixed commit `58330b2`): not a leakage issue. It still needs
  disclosing to the user, as planned.

## Checks I ran

| Script or command | Result |
|---|---|
| `uv run pytest` on the readme, forward, protocol, confirm, stages, CLV and symmetric suites | 113 passed; tree clean afterwards |
| `rev/pit_checks.py` | 0 of 212 columns differ under both scrambles; 0 differ in forward equivalence |
| `rev/time_build.py` | all-family build 7.5 s; `features_v2.parquet` on disk identical |
| `rev/clv_confound.py`, plus a short inline script that ran the constant-0.5 and shrunk-open benchmarks | Numbers in I1; selection window only, no labels |
| `rev/sym_split.py` | Numbers in M4 |
| Ledger, hash, ancestry and git-history inspection | Question 2 |

---

## Response (controller, 2026-10-14)

| Finding | Action |
|---|---|
| **I1** CLV reading overclaimed | **Fixed.** The README reading was rewritten. `clv.json` now carries post-hoc controls, labelled as such: the constant-50% forecaster, which scores more negative than the model on both raw measures, and the edge–move partial correlation given the open. The pre-registered (a)–(d) are unchanged. |
| M1 stage-0 holdout inside the scored window | **Disclosed** in the protocol text and the stage reading. The overfitting explanation is now worded as a guess, and "out-of-sample" is qualified in the CLV text. |
| M2 no control across stages | **Disclosed** in the protocol's rule bullet. |
| M3 leakage-suite AUC test scored 2015 | **Fixed.** The shuffle test and its control use `dev_only_split(..., through=selection_end)`. `diagnose` and the published diagnostics are unchanged. |
| M4 mirrored rows in the symmetric validation tail | **Disclosed** under Phase 8 limitations. Not changed, because it would alter the frozen, confirmed and already forward-tested configuration. |
| M5 commitment check could be forged | **Fixed.** Settlement queries GitHub's API for the Actions run named in `meta.json`. The run must belong to this repository, be scheduled or dispatched, have started before the deadline, and be named in the message of the commit that added the predictions; otherwise the card is `verified: false`. The pre-start price snapshot is used only if it was committed before the card started. The workflow passes its token to avoid API rate limits. |
| M6 `code_tree_clean` false for an innocent reason | **Fixed.** The git state is read before `sync_results`. The first card's `meta.json` (predicted before the fix) still records `false`; it is left as committed. |
| M7 "once" rests on the file | **Accepted.** The confirmation has already run once, from a clean pushed commit (git evidence in question 2). Adding an attempt ledger now could not cover that run, and a backfilled entry would be fabricated. |
| M8 README accuracy | **Fixed:** (a) the stale "no live predictions" limitation; (b) the overturned-results wording, now a stated small look-ahead; (c) the join wording, which now says how many joins went through the alias table; (d) the upstream odds errors, now generated into `clv.json` and rendered (3 of 1,981); (e) the forward-test exception to the 2016 guard; (f) the forward table and pooled line, which now compare model and Kalshi on the same bouts; (g) the stage-3 floor count, now rendered. |
| M9 | No action needed. |

Process note, outside leakage: commit `58330b2` mixes Task 7a's staged feature code with the controller's forward-test files under one subject. It is left as pushed, because pushed history is not rewritten, and is disclosed to the user.

### Re-review of the fix wave (scoped, 2026-10-14)

All findings were addressed or accepted as disclosed, except **M5, which is partly fixed**.
The Actions-run check confirms that the named run is real, belongs to this repository, was
scheduled or dispatched, and started before the deadline. Nothing ties that run to the
commit, though, so a backdated commit naming an earlier real run would still pass. GitHub's
events API cannot supply a push time either: it is incomplete and delayed.

The README now says exactly what "verified" checks and that it would not stop the
repository's owner from forging a commitment. The "server-side timestamp" claim is gone.
A tamper-evident seal is left as a follow-up for the user to decide on: for example, an
Actions artifact, created before the deadline, whose name contains the predictions file's
SHA-256. The first card could not carry it retroactively.

Wording nits from the re-review were also fixed: the tuning tilt now applies to stages
1-6 only, with the final re-tune going the other way, and "out-of-sample" is removed from
`clv.py`.
