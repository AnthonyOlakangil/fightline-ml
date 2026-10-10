# ufc-predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible, provably leak-free UFC fight-outcome model evaluated walk-forward against the vig-free betting market, with a Monte-Carlo parlay pricer, a CLI, and a deployed Streamlit demo.

**Architecture:** Committed dated Parquet snapshots (raw) → DuckDB-queried normalized tables → a single point-in-time feature builder that computes every feature from strictly-prior fights only → three models behind one `predict(X) -> p` interface → a walk-forward backtest that retrains per-event with an asserted training cutoff → a seeded simulator for method/round/parlay pricing. The leakage test suite is written in Phase 2 and gates every later phase.

**Tech Stack:** Python 3.11 (via uv), pandas + pyarrow, DuckDB, scikit-learn, LightGBM, NumPy, matplotlib, Typer, Streamlit, pytest, ruff. NumPyro only in the isolated stretch module.

**Spec:** `CLAUDE.md` (the contract) + the Phase 0–7 kickoff in session history.

---

## Context

This repo is empty apart from `CLAUDE.md`. The deliverable is a portfolio piece whose primary reader is a technical recruiter or hiring engineer, so the binding constraint is not feature count — it is that every number in the README is regenerable from a fresh clone and that the pipeline is demonstrably free of data leakage. A large project with hidden leakage is disqualifying; a small one with a rigorous negative result is strong.

Three things were verified empirically before writing this plan, and they change the design:

1. **UFCStats is behind a JS proof-of-work anti-bot challenge.** `requests` + bs4 returns a 2,998-byte challenge stub, not data. Decision (user-approved): the graded path uses a committed mirror snapshot and never touches the network. CLAUDE.md line 37 sanctions this; line 23 requires it.
2. **The odds→UFCStats join is a name problem, not a date problem.** An exact join on (date, normalized fighter pair) matches 6,597/6,937 = 95.1%. All 340 misses are name variants; zero are fixed by date tolerance. So the join needs an alias table, not fuzzy date logic.
3. **The market is strong.** The vig-free closing line scores **log loss 0.6076, Brier 0.2102, 66.15% favorite accuracy** over 4,744 fights from 2016 onward (coin-flip log loss = 0.6931). This is the bar. Beating it is genuinely hard, and the plan is built to report an honest loss as readily as a win.

## Global Constraints

- Python **3.11** exactly (`requires-python = ">=3.11,<3.12"`). System Python is 3.14.7; uv must fetch 3.11.
- Every feature for a fight is computable from data dated **strictly before** that fight. No exceptions.
- All randomness seeded from `config/config.yaml`. No seed literals in source.
- Fail loudly: unexpected scrape shape, row-dropping join, or unparseable date **raises**. Log row counts at every pipeline stage.
- No real-money framing. Research backtest only; README states it is not betting advice.
- Commits authored by Anthony Olakangil only. **No `Co-Authored-By` trailer, no "Generated with Claude Code" line.**
- Conventional commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`). One logical change per commit. Commit **and push** at every checkpoint, never only at phase ends.
- No hardcoded metrics anywhere. Every README number comes from `reports/metrics.json`.
- Never force-push, never rewrite pushed history, never commit secrets or `.env`.

## Review Focus

Failure modes the spec implies but that no single task's happy path exercises. Each gets a test in the task that owns the code.

1. **Red/Blue corner orientation is label-biased** — Red wins 4,145 vs Blue 3,024 (57.8%). Orienting fights by corner lets a model score well by learning "pick Red" and distorts the market comparison. → Canonical orientation must be label-independent (sorted `fighter_id`), asserted to a ~50% base rate. *Task 2.2.*
2. **Debut fighters have no prior fights** — rolling features must be `NaN`, never backfilled from career totals (which include future fights). → *Task 2.4.*
3. **The 118 precomputed feature columns in the odds CSV are a leakage minefield** — they appear to be career aggregates that may include the fight in question or later ones. → Only 5 columns are ever read from that file (`R_fighter`, `B_fighter`, `date`, `R_odds`, `B_odds`); a test asserts the loader exposes nothing else. *Task 1.3.*
4. **The LightGBM model must not see odds** — otherwise "model beats market" is circular. → A test asserts the feature matrix contains no odds-derived column. *Task 3.3.*
5. **Draws and No Contests** (5 + 3 in the odds file, 217 "OTHER" methods in UFCStats) break a binary label and a method multinomial. → Excluded from training/eval with a logged count, never silently coerced to a loss. *Task 2.1.*

---

## File Structure

```
config/config.yaml              seeds, paths, date windows, hyperparams, kelly fraction
data/raw/ufcstats_snapshot_2026-10-09/   committed Parquet + SOURCE.md
data/raw/odds_snapshot_2026-10-09/       committed Parquet + SOURCE.md
data/processed/                 normalized tables + features.parquet
src/ufc_predictor/
  config.py                     typed config loader, single source of seeds
  ingest/snapshot.py            fetch+parse mirror CSVs -> raw Parquet (network, manual)
  ingest/normalize.py           raw -> events/fighters/fights/fight_stats
  ingest/odds.py                odds CSV -> odds table; vig removal; alias join
  ingest/aliases.csv            reviewed name-variant map (hand-checked)
  storage/db.py                 DuckDB connection + table registration helpers
  features/pit.py               THE point-in-time feature builder
  features/elo.py               sequential Elo (pre-fight ratings only)
  models/base.py                Model protocol: fit(X,y) / predict_proba(X) -> p
  models/market.py              vig-free market baseline
  models/elo_model.py           Elo -> probability
  models/lgbm.py                LightGBM
  backtest/walkforward.py       per-event train/predict loop with cutoff assertion
  backtest/metrics.py           log loss, Brier, ECE, paired bootstrap CIs
  backtest/report.py            charts + metrics.json
  sim/method_model.py           P(method, round | winner)
  sim/simulator.py              seeded Monte Carlo card simulator
  sim/parlay.py                 parlay EV ranking, fractional Kelly, bankroll paths
  cli.py                        Typer: data, features, backtest, report, predict
app/streamlit_app.py            demo
tests/                          test_leakage.py (mandatory) + per-module tests
reports/                        generated charts, metrics.json, review.md
```

---

## Phase 0 — Scaffold + configure

**Deliverables:** uv project on Python 3.11; repo layout above; `Makefile` (`setup`, `data`, `features`, `backtest`, `report`, `test`, `lint`, `app`); `ruff` + `pytest` configured; `.github/workflows/ci.yml` running ruff + pytest on push; `config/config.yaml` with all seeds; `.gitignore` excluding `data/processed/`, `.venv`, `__pycache__`, `.env`; `docs/` containing this plan.

**Already done this session:** `.claude/settings.json` written; `web-design-guidelines` skill installed and symlinked for Claude Code.

**Tests:** `pytest` collects and passes a trivial `test_config.py` asserting `load_config().seed == 42` and that the seed is read from YAML, not a literal. `ruff check` clean.

**Artifacts/numbers:** CI green badge; `uv.lock` committed; `python --version` == 3.11.x inside `.venv`.

**Checkpoint commits:** `chore: scaffold uv project and repo layout` → `chore: add ruff, pytest, CI workflow` → `feat: add seeded config loader`. Set origin, push.

> **Git/GitHub gate: CLEARED (verified 2026-10-10).**
> - `gh api user` → `AnthonyOlakangil`, token scopes `gist, read:org, repo`. Account confirmed as the user's.
> - Git identity already correct: Anthony Olakangil / a.olakangil26@gmail.com.
> - **The public repo `AnthonyOlakangil/ufc-predictor` already exists and is empty** (`isEmpty: true`, `diskUsage: 0`, zero commits, no default branch). Nothing to overwrite.
>
> Phase 0 therefore **skips `gh repo create`** and instead runs:
> `git remote add origin https://github.com/AnthonyOlakangil/ufc-predictor.git`
> then `git push -u origin main`. The scaffold is still the first commit, as specified.
> Local branch is already `main`, matching GitHub's default — no branch rename needed.

---

## Phase 1 — Ingest + store

**Deliverables:** Dated raw snapshots committed as Parquet with a `SOURCE.md` recording URL, retrieval date, row count, and license for each. Normalized tables written to `data/processed/`. DuckDB helpers registering each Parquet as a view.

**Sources (reachability verified 2026-10-09):**

| Source | Status | Use | License |
|---|---|---|---|
| `ufcstats.com` | **Blocked** — JS proof-of-work challenge | none (graded path) | — |
| `Greco1899/scrape_ufc_stats` mirror | 200 OK, current to **Oct 3 2026** | canonical fights/stats/fighters | GPL-3.0 (code); data is factual record. Own parsers written; no code copied. |
| `shortlikeafox/ultimate_ufc_dataset` | 200 OK | **odds columns only** | Apache-2.0 |
| `bestfightodds.com` | 200 OK | unused; documented fallback | — |

**Expected row counts (verified):** events 791 · fight_results 8,923 joinable bouts · fight_stats 42,062 per-round rows · fighters 4,629 · odds 6,937 priced fights (2010-03-21 → 2026-03-28).

**Tests:**
- `test_snapshot_row_counts` — each raw Parquet matches the count recorded in `SOURCE.md`; drift raises.
- `test_normalize_no_silent_drops` — input rows == output rows + explicitly logged exclusions; asserts the two reconcile exactly.
- `test_odds_loader_column_whitelist` — the odds loader's output has exactly the 5 permitted source columns plus derived vig-free probabilities (Review Focus #3).
- `test_vig_removal` — two-sided probabilities sum to 1.0 within 1e-9; a known pair (-130/+102) maps to hand-computed values.
- `test_dates_parse_strictly` — a malformed date raises rather than becoming `NaT`.
- `test_bout_string_whitespace` — `"UFC 332: Silva vs. Wang "` (trailing space, present in real data) joins correctly.

**Artifacts/numbers:** `reports/ingest_log.json` with per-stage row counts and the final join rate. Target: ≥95.1% baseline, ≥98% after the alias table, with every unresolved pair listed in `reports/unjoined_odds.csv` and dropped loudly — never silently.

**Checkpoint commits:** snapshot fetch+commit → normalizer → odds loader + vig removal → alias table.

---

## Phase 2 — Point-in-time features ⟵ **the crux; run at xhigh**

**Deliverables:** `features/pit.py` producing one row per fight with features for both fighters, plus `features/elo.py`. **The leakage suite is written in this phase, before any modeling.** Nothing proceeds until it is green.

**Method:** build a long fighter-fight table sorted by `(fighter_id, date)`. Every rolling/cumulative statistic is computed as `groupby(fighter_id).shift(1)` **before** aggregation, so the current fight can never enter its own features. Elo is a single forward pass over events in date order that records the pre-fight rating, then updates.

**Features:** reach/height/age diffs (age from static DOB + fight date); rolling (last 3, last 5, career-to-date) significant strikes landed & absorbed per minute, takedowns landed per 15, takedown defense, submission attempts per 15, control-time share, knockdowns per 15; days since last fight; prior UFC fight count; win/loss streak; pre-fight Elo and Elo diff. All diffs computed in canonical orientation.

**Tests (`tests/test_leakage.py` — mandatory):**

```python
def test_features_recomputed_from_truncated_history_are_identical(fights, stats):
    """Gold standard: rebuilding features from only pre-fight data must not change them."""
    full = build_features(fights, stats)
    rng = np.random.default_rng(load_config().seed)
    for fid in rng.choice(full.index, size=50, replace=False):
        d = full.loc[fid, "date"]
        truncated = build_features(fights[fights.date < d], stats_before(stats, d))
        if fid in truncated.index:
            pd.testing.assert_series_equal(
                full.loc[fid, FEATURE_COLS],
                truncated.loc[fid, FEATURE_COLS],
                check_names=False,
            )


def test_no_feature_row_uses_event_on_or_after_its_fight(feature_provenance):
    """Every source event contributing to a feature row predates the fight."""
    for fight_id, fight_date, source_dates in feature_provenance:
        assert max(source_dates) < fight_date, f"{fight_id} used event on/after fight date"


def test_elo_rating_is_pre_fight(fights):
    elo = build_elo(fights)
    for fid in fights.sample(30, random_state=load_config().seed).index:
        d = fights.loc[fid, "date"]
        assert elo.loc[fid, "elo_1"] == build_elo(fights[fights.date < d]).rating(
            fights.loc[fid, "fighter_1"]
        )


def test_debut_fighter_rolling_features_are_null(features):
    """Not zero, not a career average — NaN. Backfilling would import the future."""
    debuts = features[features.n_prior_fights_1 == 0]
    assert len(debuts) > 0
    assert debuts[ROLLING_COLS_1].isna().all().all()


def test_canonical_orientation_has_no_label_bias(features):
    """Red corner wins 57.8%; sorted-id orientation must be ~50% or the eval is distorted."""
    assert 0.48 <= features.label.mean() <= 0.52


def test_draws_and_no_contests_excluded_with_logged_count(features, ingest_log):
    assert features.label.isin([0, 1]).all()
    assert ingest_log["excluded_non_decisive"] == 8
```

**Artifacts/numbers:** `data/processed/features.parquet`; `reports/feature_summary.json` with row count, per-feature null rate, and the asserted label base rate. Expected ≈8,900 fights with features, ≈6,800 of them also priced.

**Checkpoint commits:** `test: add leakage suite` (failing, first) → `feat: point-in-time feature builder` → `feat: sequential Elo` → `test: leakage suite green`.

---

## Phase 3 — Models

**Deliverables:** `models/base.py` defining the shared protocol; three implementations.

**Interfaces:**
- Consumes: `features.parquet`, `odds` table.
- Produces: `Model.fit(X: pd.DataFrame, y: pd.Series) -> None`, `Model.predict_proba(X: pd.DataFrame) -> np.ndarray` (shape `(n,)`, P(fighter_1 wins)). All three conform, so the backtest treats them identically.

**(a) Market baseline** — vig-free favorite from closing odds; `fit` is a no-op. **(b) Elo** — logistic on pre-fight rating gap, K and scale from config. **(c) LightGBM** — hyperparams fixed from the 2010–2015 dev window only, then frozen.

**Tests:**
- `test_all_models_satisfy_protocol` — parametrized over all three; output shape, dtype, and range `[0,1]`.
- `test_market_recovers_known_probability` — hand-computed vig-free value for a known pair.
- `test_lgbm_feature_matrix_has_no_odds_columns` — Review Focus #4; fails if any column name matches an odds/market pattern.
- `test_determinism` — same seed, same config → bitwise-identical predictions across two runs.

**Artifacts/numbers:** `reports/model_cards.json` — feature count per model, hyperparams, dev-window fit metrics.

---

## Phase 4 — Walk-forward backtest + evaluation ⟵ **the core result; run at xhigh**

**Deliverables:** `backtest/walkforward.py`. Events sorted by date; for each event in the eval window, train on all fights strictly before that event's date, predict that event, advance. LightGBM retrains on a configurable cadence (default: every 10 events) but **always** with a cutoff strictly before the event being predicted — asserted per fold regardless of cadence.

**Windows:** dev/tuning **2010–2015** (2,172 priced fights) — the only window hyperparameters may touch. Headline walk-forward evaluation **2016-01-01 → 2026-10-03** (4,765 priced fights, 4,744 decisive), untouched during development.

**Tests:**
- `test_training_cutoff_is_respected_every_fold` — for every fold, `train.date.max() < test_event_date`. This is the test CLAUDE.md mandates.
- `test_no_fight_predicted_twice` — fold test sets partition the eval window exactly.
- `test_retrain_cadence_does_not_leak` — with cadence 10, the model predicting event *k* was fit only on data before event *k*, not before event *k−k%10*.
- `test_bootstrap_ci_is_seeded` — two runs, identical bounds.
- `test_metrics_match_hand_computation` — log loss and Brier on a 5-row fixture vs. values computed by hand.

**Artifacts/numbers:** `reports/metrics.json` — per model: log loss, Brier, accuracy, AUC, ECE, each with a seeded **paired bootstrap 95% CI (10,000 resamples)** over fights, plus skill score vs. market and the paired difference CI (the honest "did we beat the market" test). `reports/calibration.png`, `reports/log_loss_by_year.png`, `reports/reliability_market_vs_model.png`.

**The bar, already measured:** market = log loss **0.6076**, Brier **0.2102**, favorite accuracy **66.15%** on n=4,744. The README reports the model's numbers against these whatever the direction. If the paired CI for (model − market) log loss includes zero or is positive, the README says plainly that the model does not beat the closing line, and shows it.

---

## Phase 5 — Simulator + parlay pricer

**Deliverables:** Method/round model factored as P(winner) × P(method | winner) × P(round | method, winner) — avoids a sparse 30-class multinomial. Observed method base rates for calibration: DEC 4,149 / KO-TKO 2,853 / SUB 1,731 / OTHER 217 (excluded). Seeded Monte Carlo, 10,000 sims per card. Parlay EV ranking against book odds, fractional Kelly (default 0.25, from config), bankroll paths with a bootstrap CI band.

**Documented assumption:** fights on a card are modeled independent conditional on features, so a straight-parlay probability equals the product of legs; the simulator is what makes method/round legs and the bankroll CI band tractable. Stated explicitly in the README rather than left implicit.

**Tests:**
- `test_simulator_is_seeded` — same seed → identical sims.
- `test_method_probabilities_sum_to_one` — per fighter, across all method×round cells.
- `test_independent_parlay_matches_analytic_product` — 3-leg parlay, sim within Monte-Carlo error of the exact product.
- `test_kelly_is_zero_at_nonpositive_ev` — no stake when EV ≤ 0.
- `test_bankroll_never_negative` — fractional Kelly cannot bankrupt a path.
- `test_no_real_money_language` — greps `app/` and `README.md` for betting-advice phrasing.

**Artifacts/numbers:** `reports/bankroll.png` (median + 95% band), `reports/parlay_ev_table.csv`, `reports/sim_metrics.json` with method-model log loss vs. a base-rate baseline.

---

## Phase 6 — Interface

**Deliverables:** Typer CLI — `data`, `features`, `backtest`, `report`, and `predict --event <name>` writing a prediction-log CSV (fight, model p, market p, stake, result once known). Streamlit demo: pick or enter a matchup, show model vs. market probability, calibration, and a sample parlay. Load `frontend-design` **and** the installed `web-design-guidelines` skill before building the UI.

**Tests:** `test_cli_predict_writes_expected_csv_schema` (Typer `CliRunner`); `test_predict_unknown_event_exits_nonzero_with_message`; `test_predict_unknown_fighter_is_handled` (debut/unseen fighter must not crash); `test_streamlit_app_imports` (smoke, no network).

**Artifacts/numbers:** `reports/predictions_<event>.csv`.

---

## Phase 7 — Deploy + document

**Deliverables:** Streamlit Community Cloud deploy (fallback: Hugging Face Spaces). README to CLAUDE.md's Definition of Done: the question, data + sources with licenses, the point-in-time method, models, results table with bootstrap CIs vs. market, calibration plot, bankroll plot with CI band, live-demo link, and an honest limitations section. Then the mandated adversarial review.

**Tests:** `make setup && make data && make features && make backtest && make report` clean from a fresh clone; full `pytest` green; `ruff check` clean; CI green. `test_readme_numbers_match_metrics_json` — parses the README results table and asserts every figure equals `reports/metrics.json`, so the README cannot drift or hold a stale hand-typed number.

**Artifacts:** `reports/review.md` from the adversarial subagent; fixes committed.

---

## Subagents

Per the kickoff: `model: sonnet` for parallel/read-heavy/boilerplate work; leakage and backtest reasoning stay on the main Opus thread.

| Phase | Agent | Model | Returns |
|---|---|---|---|
| 1 | Snapshot schema profiler | sonnet | Per-column dtype, null rate, cardinality, and anomalies for all 4 mirror CSVs + the odds CSV; a written schema report. No code. |
| 1 | Name-alias resolver | sonnet | The 340 unjoined pairs resolved into `ingest/aliases.csv` (odds name → UFCStats name), plus a list of pairs that are genuinely absent from UFCStats with the reason. Hand-reviewed by me before commit. |
| 3 | Model boilerplate | sonnet | `market.py` and `elo_model.py` conforming to the Phase-3 protocol, with their unit tests. Interface is fixed by me first. |
| 6 | Streamlit UI builder | sonnet | `app/streamlit_app.py` against an interface I define, after loading the design skills. |
| 7 | **Adversarial leakage/overfit reviewer** | **opus** | `reports/review.md`. Fresh context, hunts only for leakage and overfitting. CLAUDE.md line 74 mandates a subagent here; this is the one place sonnet would under-deliver, so it gets Opus. |

Phases 2 and 4 use **no** subagents — that reasoning stays on the main thread.

---

## STRETCH — only after Phases 0–7 ship green

Gated explicitly: neither of these may touch the graded path, and neither is started
until `make setup && make data && make backtest && make report` is clean from a fresh
clone and CI is green.

**S1 — Bayesian skill model (NumPyro).** Per-fighter latent striking and grappling skill
drifting over time (a Gaussian random walk across each fighter's fight sequence), with
P(A beats B) = sigmoid(skill gap). Isolated in `src/ufc_predictor/models/bayes.py` behind
an optional `[bayes]` extra so the whole module can be skipped and CI stays fast; a test
asserts the core suite passes with NumPyro uninstalled. Added as a **4th row** in the
backtest table, subject to the same per-fold cutoff assertion as every other model — the
random walk must only ever condition on fights before the predicted event, which is the
same leakage trap as Elo and gets the same test.

**S2 — LLM news flags.** Cache-backed module flagging injury / short-notice / weight-cut
signals per fighter via the Claude API, with responses cached on disk so runs are
reproducible and the API is hit once per fighter-event. Surfaced as an **annotation in the
Streamlit app only**, never as a model feature on the graded path — point-in-time
correctness cannot be guaranteed for news text, so it stays off the result entirely. A
test asserts no news-derived column appears in the feature matrix.

---

## Top 3 risks

**1. Data leakage (highest).** The project's credibility rests entirely on this, and it is the first thing a reviewer will check. Four distinct vectors: (a) rolling features including the current fight; (b) Elo updated before predicting; (c) the odds CSV's 118 precomputed career-aggregate columns; (d) corner-based orientation leaking label information. *Neutralized by:* writing the leakage suite in Phase 2 before any model exists; the recompute-from-truncated-history test, which catches leakage no column audit would; a 5-column whitelist on the odds loader; sorted-id canonical orientation asserted to a ~50% base rate; a per-fold training-cutoff assertion in the backtest; and a fresh-context adversarial Opus review in Phase 7.

**2. Overfitting to the eval window / implicit hyperparameter leakage.** With ~4,700 evaluation fights and a market baseline at 0.6076, small log-loss gains are easy to manufacture by tuning against the test period. *Neutralized by:* hyperparameters fixed on 2010–2015 and frozen before the 2016+ window is ever scored; walk-forward (every prediction out-of-sample by construction); **paired** bootstrap CIs on the model−market difference rather than two separately-eyeballed point estimates; and a stated willingness to publish a negative result. The honest expectation is that closing lines are hard to beat.

**3. The join and the single odds dependency.** Odds come from one community dataset covering 2010-03-21 → 2026-03-28 (so the most recent ~6 months of fights are unpriced), joined by fighter name at 95.1% exact. Bad alias matching could silently mispair fights and quietly corrupt the headline result. *Neutralized by:* exact (date, fighter-pair) matching as the only automatic rule — no fuzzy-score auto-accept; a hand-reviewed alias table; every unresolved pair written to `reports/unjoined_odds.csv` and dropped with a logged count, never silently; row-count reconciliation tests; and `SOURCE.md` documenting coverage limits so the README states them plainly.

---

## Verification (end to end, from a fresh clone)

CLAUDE.md's Definition of Done names exactly four commands, so `make backtest` must
depend on `features` and `make features` on `data`. The four-command path is the
contract; the finer targets exist for iteration.

```bash
git clone <repo> && cd ufc-predictor
make setup                 # uv sync, Python 3.11
make data                  # snapshot -> normalized Parquet; prints row counts; no network
make backtest              # (depends on: features) walk-forward 2016+; writes metrics.json
make report                # charts + README figure refresh
# finer targets for iteration:
make features              # features.parquet; prints label base rate (expect ~0.50)
make test                  # pytest incl. tests/test_leakage.py  -> ALL GREEN REQUIRED
make lint                  # ruff check
uv run ufc-predictor predict --event "UFC 332: Silva vs. Wang"
make app                   # Streamlit demo on localhost
```

Acceptance: leakage suite green; `metrics.json` contains model and market log loss with a paired bootstrap CI on the difference; every README number traceable to `metrics.json` (enforced by `test_readme_numbers_match_metrics_json`); CI green on GitHub.

---

## Open items for you

1. ~~`gh auth login` is required~~ — **done.** Authenticated as `AnthonyOlakangil`; repo exists and is empty. No blockers remain.
2. **Effort level is user-controlled.** I cannot raise or lower it myself; `/effort` is yours. It is now at **high**, which suits Phases 0, 1, 3, 5, 6, 7. **Raise to xhigh before Phase 2 (features/leakage) and Phase 4 (backtest)** — those are the two phases where the reasoning depth actually pays for itself.
3. **Python 3.11 vs. system 3.14.7** — uv will fetch 3.11 per CLAUDE.md. 3.11 is also the safer target for the NumPyro stretch module.
4. **Plan still awaiting your approval.** Nothing in the repo has been built; only `.claude/settings.json` and this plan document exist.
