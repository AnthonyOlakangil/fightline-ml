# Phase 8 — Model Improvement (dev only), CLV, and a Forward Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the fight model using the development window only, measure closing-line value on historical opening lines, freeze the best development configuration, and forward-test it on future cards whose predictions are committed before each event.

**Architecture:** Five independently testable parts, built in order: (A) data sources — a BestFightOdds historical opening/closing-line snapshot and live clients for ESPN cards, Kalshi prices and BestFightOdds odds; (B) a development-window protocol — a selection window (2010–2014), a once-only confirmation year (2015), a pre-registered experiment list and an append-only experiment ledger, with code guards that refuse to score anything dated 2016 or later; (C) new point-in-time feature families plus model, calibration and symmetry options, compared in pre-registered stages; (D) a closing-line-value analysis on the development window; (E) a forward-test pipeline that commits predictions before each event and scores them after.

**Tech Stack:** Python 3.11 (uv), pandas, NumPy, scikit-learn, LightGBM, requests + BeautifulSoup (already dependencies), pytest, ruff. No new runtime dependency: Glicko-2 is implemented in-repo (about 80 lines) and checked against Glickman's published worked example.

**Spec:** the user's Phase 8 request (session of 2026-10-13), the three decisions recorded under **Decisions** below, and `CLAUDE.md`.

## Decisions already made (user, 2026-10-13)

1. **Opening lines:** scrape BestFightOdds fighter pages (about 1,200 pages at one request per 5 s, cached, committed as a dated snapshot). Use BestFightOdds for both the opening and the closing price in the CLV analysis, so the two share one definition.
2. **Forward-test sources:** ESPN's public JSON for cards; Kalshi's public market data as the primary price (mid of bid and ask, normalized across the two fighters); BestFightOdds' multi-book odds as a cross-check. No accounts or API keys.
3. **Development protocol:** select on 2010–2014 (walk-forward inside it), confirm on 2015 exactly once, with a pre-registered experiment list and Holm-corrected comparisons.

## Decisions at approval (user, 2026-10-13)

4. **Execution is hybrid.** A fresh implementer subagent plus a leakage reviewer for Tasks 7a, 7b, 7c, 8 and 3 (the join that feeds CLV); every other task is implemented directly, with one review at the end.
5. **BestFightOdds:** check `robots.txt`, rate-limit and cache before scraping, and stop if anything disallows it. Checked 2026-10-13: `User-agent: * / Allow: /`, no crawl delay; `/terms` holds only an accuracy disclaimer.
6. **Forward-test prices:** prefer the Kalshi API.
7. **Forward-test automation:** a GitHub Actions schedule commits predictions at least 24 hours before each card and alerts the user on failure (new Task 15).
8. The repository is renamed to `AnthonyOlakangil/fightline-ml`; the remote and README follow.

Execution order (Task 6 consumes Task 9's models; Task 3 waits for the crawl): 1, 2, 5, 7a, 7b, 7c, 8, 9, 6, 10, 10b, 3, 11, 4, 12, 13, 15, 14.

## Global Constraints

- **The 2016+ evaluation window is closed.** No code path in this phase may score a prediction dated on or after `2016-01-01` against an outcome, except the forward test, which only scores fights dated after the freeze commit. Training on 2016+ fights is allowed only for the frozen forward-test model.
- **Selection window:** `2010-01-01`–`2014-12-31`. **Confirmation window:** `2015-01-01`–`2015-12-31`, scored once for the frozen configuration and ledgered; a second confirmation run raises.
- Every new feature is computable from data dated **strictly before** the fight; the existing mutation and shuffle leakage tests must cover every new column.
- Nothing from the present-day fighters table (height, reach, date of birth, stance, weight) may become a feature (existing tests enforce this).
- All randomness seeded from `config/config.yaml`; no literals.
- Fail loudly on an unexpected scrape shape, a row-dropping join, or an unmatched fighter name.
- Network access only in scraper/client modules, behind an on-disk cache; tests use committed fixtures and never touch the network.
- Scraping: one request per 5 s to BestFightOdds, an identifying User-Agent, cache every response, never re-request a cached URL.
- Every README number is generated from a JSON file in `reports/` and covered by a README-section test.
- Conventional commits, authored by the user only, no Claude attribution, push after every commit, never rewrite pushed history.
- Research only; nothing here is betting advice. The forward test records a market comparison, never a stake.

## Review Focus

1. **Name matching for upcoming fighters.** A real veteran whose ESPN name differs from UFCStats ("Alex Pereira" vs "Alex Pereira Jr.") must not be treated as a debutant; that would silently give him empty history. → `forward/cards.py` raises when an unmatched name has a near match (similarity ≥ 0.85) and treats a name as a debutant only when no candidate exists. *Task 12.*
2. **A cancelled or replaced bout after predictions are committed.** Late replacements are common. → Settlement scores only fights whose committed pair matches the result pair; replaced bouts are reported as void, never scored against a different opponent. *Task 13.*
3. **Same-date events in the walk-forward** (a known review finding). → The new experiment runner filters stacker and calibration history with `date < event_date`, and a test injects two events on one date. *Task 6.*
4. **Feature selection seeing the fold it scores.** → Selection runs on the fold's training rows only; a test plants a feature that predicts the label only in the scored rows and asserts it is never selected. *Task 8.*
5. **An opening or closing price missing on one side, or a BFO row on a different date from our fight.** → The join drops (with a logged count) any fight missing either side's open or close and never matches across dates. *Task 3.*

---

## File Structure

```
config/
  config.yaml                     # + windows.selection_end, windows.confirmation_*,
                                  #   features.recency_half_life_fights, glicko2.*, forward.*
  experiments.yaml                # NEW: pre-registered stages, candidates, decision rule
src/ufc_predictor/
  ingest/
    bfo.py                        # NEW: BestFightOdds scraper + fighter-page parser (historical)
    bfo_join.py                   # NEW: join BFO open/close rows to fights
  features/
    glicko2.py                    # NEW: Glicko-2 rating system, point-in-time per event date
    adjusted.py                   # NEW: opponent-adjusted and recency-weighted rates
    styles.py                     # NEW: style-matchup interactions
    decline.py                    # NEW: decline signals
    pit.py                        # MODIFY: FEATURE_FAMILIES registry; build_features adds families
  models/
    logistic.py                   # NEW: L2 logistic regression benchmark
    calibration.py                # NEW: Platt and bounded-isotonic calibrators behind one interface
    symmetric.py                  # NEW: orientation-symmetric wrapper
    lgbm.py                       # MODIFY: calibrator chosen by config; optional feature subset
  experiments/
    __init__.py
    protocol.py                   # NEW: windows, eval-closed guard, experiment ledger
    selection.py                  # NEW: per-fold top-K feature selection
    runner.py                     # NEW: dev walk-forward for one configuration
    stages.py                     # NEW: runs pre-registered stages, applies the decision rule
    confirm.py                    # NEW: the once-only 2015 confirmation
  analysis/
    clv.py                        # NEW: closing-line value on the development window
  forward/
    espn.py                       # NEW: upcoming cards and results (cached client)
    kalshi.py                     # NEW: market prices (cached client)
    bfo_live.py                   # NEW: current multi-book odds (cached client)
    cards.py                      # NEW: map card names to fighter ids; build prediction frames
    ledger.py                     # NEW: append-only forward ledger; commit-before-event check
    run.py                        # NEW: predict / close / settle commands
  cli.py                          # MODIFY: bfo, experiments, confirm, clv, forward commands
tests/
  fixtures/bfo/, fixtures/espn/, fixtures/kalshi/   # small recorded responses
  test_bfo.py  test_glicko2.py  test_features_v2.py  test_protocol.py  test_selection.py
  test_calibration.py  test_logistic.py  test_symmetric.py  test_stages.py
  test_clv.py  test_forward.py
reports/
  dev/experiments.json            # append-only experiment ledger
  dev/stages.json                 # per-stage results incl. rejected candidates
  dev/confirmation.json           # the single 2015 confirmation
  clv.json
  forward/ledger.json, forward/<event>/predictions.csv, prices_*.json, settlement.json
data/raw/bfo_snapshot_<date>/     # committed Parquet snapshot of BFO historical lines
```

---

## Part A — Data sources

### Task 1: BestFightOdds discovery spike (no code committed)

**Purpose:** confirm the crawl path before writing the scraper. Fighter pages carry `Open` and `Closing range` (verified on `/fighters/jon-jones-819`, back to 2008); the open question is how to enumerate every UFC fighter of 2010–2015.

- [ ] **Step 1:** Fetch `https://www.bestfightodds.com/archive` (cached) and one 2012 UFC event page linked from it. Record whether event pages list fighter links (`/fighters/<slug>-<id>`) for every bout.
- [ ] **Step 2:** If event pages are not reachable from the archive for 2010–2015, fetch BFO's search (`/search?query=<fighter name>`) for 3 known fighters and record the result shape.
- [ ] **Step 3:** Write the chosen path (archive → events → fighters, or names from our fights table → search → fighters) and the request count into the Task 2 docstring. Stop and report to the user if neither path works.

### Task 2: BFO historical scraper, parser, snapshot

**Files:** Create `src/ufc_predictor/ingest/bfo.py`, `tests/test_bfo.py`, `tests/fixtures/bfo/fighter_page.html` (one real page trimmed to 6 fights), `data/raw/bfo_snapshot_<date>/lines.parquet`, `data/raw/bfo_snapshot_<date>/SOURCE.md`.

**Interfaces:**
- Produces: `parse_fighter_page(html: str) -> pd.DataFrame` with columns `fighter, opponent, event, date, open_fighter, open_opponent, close_lo_fighter, close_hi_fighter, close_lo_opponent, close_hi_opponent` (American odds as int); `fetch(url: str, cache_dir: Path, delay_s: float) -> str`.

- [ ] **Step 1: Failing parser test** against the fixture:

```python
def test_parses_open_and_closing_range_for_both_sides():
    html = (FIXTURES / "fighter_page.html").read_text(encoding="utf-8")
    rows = parse_fighter_page(html)
    shogun = rows.loc[rows["opponent"] == "Mauricio Rua"].iloc[0]
    assert shogun["date"] == dt.date(2011, 3, 19)
    assert shogun["open_opponent"] == 175
    assert (shogun["close_lo_opponent"], shogun["close_hi_opponent"]) == (195, 195)
    assert rows["date"].notna().all() and len(rows) == 6


def test_unexpected_page_shape_raises():
    with pytest.raises(ValueError, match="BestFightOdds page shape"):
        parse_fighter_page("<html><body>maintenance</body></html>")
```

- [ ] **Step 2:** Run `uv run pytest tests/test_bfo.py -v`; expect ImportError/FAIL.
- [ ] **Step 3:** Implement `parse_fighter_page` (BeautifulSoup over the table whose header row is `Matchup | Open | Closing range | Movement | | Event`; dates like `Mar 19th 2011` parsed with `dt.datetime.strptime(re.sub(r"(\d)(st|nd|rd|th)", r"\1", s), "%b %d %Y")`; raise `ValueError("BestFightOdds page shape changed: ...")` when the header is absent) and `fetch` (cache key = sha256 of URL; sleep `delay_s` only on a cache miss; identifying User-Agent from config).
- [ ] **Step 4:** Tests pass. Run the crawl from Task 1's path with `delay_s=5`; write `lines.parquet` and `SOURCE.md` (URL pattern, crawl dates, row counts, robots.txt text, the fact that the terms of service are silent on automated access).
- [ ] **Step 5:** Commit `feat: snapshot BestFightOdds historical opening and closing lines` and push.

### Task 3: Join BFO lines to fights

**Files:** Create `src/ufc_predictor/ingest/bfo_join.py`; test in `tests/test_bfo.py`. Writes `data/processed/bfo_lines.parquet` and `reports/bfo_join.json`.

**Interfaces:** Consumes `lines.parquet`, `fights` table, the alias machinery in `ingest/names.py`. Produces `join_bfo(lines, fights, aliases) -> (joined: DataFrame[fight_id, open_p1, close_p1, open_odds_1, open_odds_2, close_mid_odds_1, close_mid_odds_2], audit: dict)` where `open_p1`/`close_p1` are proportionally de-vigged probabilities for `fighter_1`, and the closing price is the midpoint of the closing range in implied-probability space.

- [ ] **Step 1: Failing tests**: exact (date, normalized pair) matches orient to `fighter_1`; a pair present twice on one date raises; a fight missing either side's open is dropped and counted in `audit["dropped_incomplete"]`; the join never matches across dates.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement with the same conservative rule as the existing odds join (exact match, alias table, no fuzzy auto-accept); write unmatched rows to `reports/bfo_unjoined.csv`.
- [ ] **Step 4:** Tests pass; run on real data; assert the dev-window join rate is reported in `bfo_join.json` (no hard threshold; the README states it).
- [ ] **Step 5:** Commit `feat: join BestFightOdds opening and closing lines to fights` and push.

### Task 4: Live clients for ESPN, Kalshi and BFO (fixtures only in tests)

**Files:** Create `forward/espn.py`, `forward/kalshi.py`, `forward/bfo_live.py`, fixtures under `tests/fixtures/{espn,kalshi,bfo}/`, `tests/test_forward.py`.

**Interfaces:**
- `espn.upcoming_cards(cache) -> list[Card]` where `Card = {event_id: str, name: str, date: dt.date, start_utc: dt.datetime, bouts: list[tuple[str, str]]}`; `espn.results(event_id, cache) -> list[{"pair": (str, str), "winner": str | None, "method": str}]`.
- `kalshi.fight_prices(card_date, cache) -> dict[frozenset[str], dict[str, float]]` — mid of `yes_bid_dollars`/`yes_ask_dollars` per fighter, normalized to sum to 1, plus raw bid, ask and volume.
- `bfo_live.current_odds(cache) -> DataFrame[event, fighter, opponent, best_odds, median_odds]`.

- [ ] **Step 1:** Record fixtures from today's live responses (ESPN scoreboard for 2026-10-18, one Kalshi `KXUFCFIGHT` events response, the BFO homepage trimmed to one event).
- [ ] **Step 2: Failing tests** parse each fixture: ESPN returns Buckley vs. Malott with its bouts; Kalshi mids for one fight sum to 1 and lie inside the bid-ask; a Kalshi market with no bid or ask is returned as missing, never as 0.5.
- [ ] **Step 3:** Implement; every client raises on an unexpected shape.
- [ ] **Step 4:** Tests pass offline (`pytest --disable-socket` is not available, so tests monkeypatch `requests.get` to raise).
- [ ] **Step 5:** Commit `feat: add cached ESPN, Kalshi and BestFightOdds clients for the forward test` and push. **Report to user: Part A row counts, join rate, and anything unreachable.**

---

## Part B — Development protocol

### Task 5: Windows, eval-closed guard, experiment ledger

**Files:** Modify `config/config.yaml`, `src/ufc_predictor/config.py`; create `experiments/protocol.py`, `config/experiments.yaml`, `tests/test_protocol.py`.

**Interfaces:**
- Config adds `windows.selection_end: "2014-12-31"`, `windows.confirmation_start: "2015-01-01"`, `windows.confirmation_end: "2015-12-31"`; `Windows.__post_init__` asserts `dev_start <= selection_end < confirmation_start <= confirmation_end == dev_end < eval_start`.
- `protocol.assert_scorable(dates, cfg, *, purpose: Literal["selection", "confirmation"])` raises `EvalWindowClosed` for any date ≥ `eval_start`, and raises for a selection score outside the selection window or a confirmation score outside 2015.
- `protocol.append_experiment(entry: dict, cfg) -> int` appends to `reports/dev/experiments.json` with `git_commit`, `worktree_clean` (via `provenance.git_state`), `config_hash`, `stage`, `candidate`, `metrics`.

- [ ] **Step 1: Failing tests:**

```python
def test_scoring_any_eval_date_raises(cfg):
    with pytest.raises(EvalWindowClosed):
        assert_scorable([dt.date(2016, 1, 2)], cfg, purpose="selection")

def test_selection_cannot_touch_the_confirmation_year(cfg):
    with pytest.raises(ValueError, match="selection"):
        assert_scorable([dt.date(2015, 3, 1)], cfg, purpose="selection")

def test_windows_are_ordered_and_disjoint(cfg):
    w = cfg.windows
    assert w.dev_start <= w.selection_end < w.confirmation_start <= w.confirmation_end == w.dev_end < w.eval_start
```

- [ ] **Step 2:** Run; FAIL.
- [ ] **Step 3:** Implement. Write `config/experiments.yaml` exactly as in **Pre-registration** below.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit `feat: add the development protocol, eval-closed guard and pre-registered experiments` and push. **This commit is the pre-registration: no experiment may run from an earlier commit.**

### Task 6: Dev walk-forward runner

**Files:** Create `experiments/runner.py`, tests in `tests/test_protocol.py`.

**Interfaces:** `run_config(cfg, spec: ExperimentSpec, *, window: Literal["selection","confirmation"]) -> pd.DataFrame[fight_id, event_id, date, label, p_market, p_model]`. `ExperimentSpec` is a frozen dataclass: `features: tuple[str, ...]`, `top_k: int | None`, `train_from: dt.date | None`, `train_years: int | None`, `symmetric: bool`, `calibration: Literal["isotonic_bounded","platt","none"]`, `model: Literal["lightgbm","logistic"]`, `lightgbm_params: dict`.

It reuses the existing walk-forward cadence and `min_train_fights`, trains on every decisive fight strictly before each predicted event (including pre-2010), predicts only events inside `window`, and calls `assert_scorable` before returning.

- [ ] **Step 1: Failing tests:** the training cutoff for every fold is strictly before the fold's first event (checked by a sentinel: a fight inserted on the predicted date with a deliberately extreme label pattern must not change any prediction — the poison test from review 1, finding 5); two events on one date never see each other's labels; requesting `window="selection"` returns no fight after 2014-12-31.
- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement. **Step 4:** Pass. **Step 5:** Commit `feat: add the development walk-forward runner with a poison-test cutoff check` and push.

---

## Part C — Features, models, stages

All feature families plug into `pit.build_features` through a registry so the existing mutation, shuffle and fighters-table tests cover them automatically:

```python
# pit.py
FEATURE_FAMILIES: dict[str, list[str]] = {
    "base": FEATURE_COLS,                 # the 117 run-3 features, unchanged
    "adjusted": ADJUSTED_COLS,            # Task 7a
    "recency": RECENCY_COLS,              # Task 7a
    "style": STYLE_COLS,                  # Task 7b
    "decline": DECLINE_COLS,              # Task 7b
    "glicko": GLICKO_COLS,                # Task 7c
}
ALL_FEATURE_COLS = [c for cols in FEATURE_FAMILIES.values() for c in cols]
```

`FEATURE_COLS` stays the 117 run-3 columns, so the published run-3 pipeline, the demo and every existing test are untouched. The leakage suite's mutation and fighters-table tests switch from `FEATURE_COLS` to `ALL_FEATURE_COLS`.

### Task 7a: Opponent-adjusted and recency-weighted rates

**Files:** Create `features/adjusted.py`; modify `pit.py`; tests in `tests/test_features_v2.py`.

**Definitions (a priori, not searched):**
- For each of the 10 `RATE_METRICS` and each past bout *b* of fighter *F* against opponent *O*: `resid_b = F's rate in b − O's pre-bout career average of the mirrored metric` (landed ↔ absorbed, accuracy ↔ defence, takedowns landed ↔ takedown defence). *O's pre-bout average comes from O's snapshot as of b's date*, so it is point-in-time by construction.
- `adj_<metric>_career_{1,2}` = mean of `resid_b` over F's prior bouts; NaN with no prior stats.
- `ewm_<metric>_{1,2}` = exponentially weighted mean of F's raw per-bout rate over prior bouts with half-life `features.recency_half_life_fights: 3` (config), most recent bout weight 1.
- Diffs `_diff` for both. 40 columns + 20 diffs = 60.

- [ ] **Step 1: Failing tests (hand-computed):** a three-fight toy history where F lands 4, 6, 2 sig. strikes per minute against opponents whose pre-bout absorbed averages were 3, 5, NaN: `adj_sig_str_landed_pm_career` before the fourth bout equals mean(1, 1) = 1.0 (the NaN-opponent bout excluded); `ewm` with half-life 3 equals `(2·1 + 6·2^(-1/3) + 4·2^(-2/3)) / (1 + 2^(-1/3) + 2^(-2/3))`. Mutation test: changing O's *later* fight leaves F's adjusted feature for an earlier date byte-identical.
- [ ] **Step 2:** FAIL. **Step 3:** Implement inside the per-date snapshot loop (state stores, per prior bout, F's rate and O's pre-bout mirrored average). **Step 4:** Pass, plus the full leakage suite on `ALL_FEATURE_COLS`. **Step 5:** Commit `feat: add opponent-adjusted and recency-weighted rate features` and push.

### Task 7b: Style-matchup interactions and decline signals

**Files:** Create `features/styles.py`, `features/decline.py`; tests in `tests/test_features_v2.py`.

**Definitions:**
- Style (computed from the two fighters' pre-fight career rates, 8 columns): `td_threat_1 = td_landed_p15_career_1 × (1 − td_def_career_2)`, `td_threat_2` mirrored, `strike_exchange_1 = sig_str_landed_pm_career_1 × (1 − sig_str_def_career_2)`, mirrored, `grappler_vs_striker = (td_landed_p15_career_1 − td_landed_p15_career_2) × (sig_str_landed_pm_career_2 − sig_str_landed_pm_career_1)`, `ko_power_vs_chin_{1,2} = kd_p15_career_1 × kd_absorbed_p15_career_2` (requires adding `kd_absorbed` to the history table), plus `_diff` where paired.
- Decline (per fighter, 12 columns): `years_since_ufc_debut` (from the fights table, point-in-time), `finished_losses_last3` (KO/TKO or submission losses in last 3 bouts), `kd_absorbed_last3`, `absorbed_trend = sig_str_absorbed_pm_w3 − sig_str_absorbed_pm_career`, `landed_trend = sig_str_landed_pm_w3 − sig_str_landed_pm_career`, `days_since_last_finish_loss`; `_diff` each.

- [ ] **Step 1: Failing tests:** hand-computed values on a toy history for every column; `years_since_ufc_debut` for a debutant is 0.0 with `is_debut` true; nothing reads the fighters table (existing test, now over `ALL_FEATURE_COLS`).
- [ ] **Steps 2–5:** FAIL → implement → pass (full leakage suite) → commit `feat: add style-matchup interactions and decline signals` and push.

### Task 7c: Glicko-2

**Files:** Create `features/glicko2.py`, `tests/test_glicko2.py`; register in `pit.py`.

**Interfaces:** `Glicko2(tau: float, initial_rating: float, initial_rd: float, initial_vol: float, period_days: int)`; `.pre_fight(fighter_id, date) -> (mu, phi, sigma)` with RD inflated for each elapsed `period_days` since the fighter's last bout; `.update_date(date, results: list[(a, b, score_a)])` updates every participant of that date simultaneously from their pre-date state. Features (7): `glicko_mu_{1,2}`, `glicko_rd_{1,2}`, `glicko_mu_diff`, `glicko_expected_1 = 1 / (1 + exp(−g(√(φ1²+φ2²))·(μ1−μ2)))`, `glicko_rd_sum`. Config `glicko2: {tau: 0.5, initial_rating: 1500, initial_rd: 350, initial_vol: 0.06, period_days: 30}` — Glickman's recommended defaults, not searched.

- [ ] **Step 1: Failing test — Glickman's worked example** (Glicko-2 paper, §3):

```python
def test_matches_glickmans_worked_example():
    g = Glicko2(tau=0.5, initial_rating=1500, initial_rd=200, initial_vol=0.06, period_days=30)
    g.set_state("p", rating=1500, rd=200, vol=0.06)
    g.set_state("a", rating=1400, rd=30, vol=0.06)
    g.set_state("b", rating=1550, rd=100, vol=0.06)
    g.set_state("c", rating=1700, rd=300, vol=0.06)
    g.update_period("p", [("a", 1.0), ("b", 0.0), ("c", 0.0)])
    r, rd, vol = g.rating("p")
    assert r == pytest.approx(1464.06, abs=0.01)
    assert rd == pytest.approx(151.52, abs=0.01)
    assert vol == pytest.approx(0.05999, abs=1e-5)
```

  Plus: a fighter's pre-fight rating never reflects the fight itself (mutation test); two fights on one date update from the same pre-date state.
- [ ] **Step 2:** FAIL. **Step 3:** Implement Glickman's algorithm (scale 173.7178; volatility by the Illinois iteration, ε = 1e-6). **Step 4:** Pass. **Step 5:** Commit `feat: add a point-in-time Glicko-2 rating` and push.

### Task 8: Per-fold feature selection

**Files:** Create `experiments/selection.py`, `tests/test_selection.py`.

**Interfaces:** `select_top_k(train: pd.DataFrame, features: list[str], k: int, cfg) -> list[str]` — fits a LightGBM on `train` with the frozen hyperparameters and early stopping on the chronological tail of `train`, ranks by total gain, returns the top `k` (ties broken by name), always as `_1/_2/_diff` groups trimmed to exactly `k`.

- [ ] **Step 1: Failing tests:** deterministic under the seed; a planted column equal to the label on the *scored* rows and noise on `train` is never selected (selection cannot see scored rows because it is never given them); `k` respected.
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: add per-fold top-k feature selection` and push.

### Task 9: Calibrators, symmetry wrapper, logistic benchmark

**Files:** Create `models/calibration.py`, `models/symmetric.py`, `models/logistic.py`; modify `models/lgbm.py` (calibrator from config: `lightgbm_fit.calibration: isotonic_bounded | platt | none`; default `isotonic_bounded`, so run 3 is unchanged); tests `test_calibration.py`, `test_symmetric.py`, `test_logistic.py`.

**Interfaces:**
- `Calibrator.fit(raw: np.ndarray, y: np.ndarray) -> Calibrator`, `.predict(raw) -> np.ndarray`. `Platt` fits `LogisticRegression(C=1e6)` on `logit(raw)`; `BoundedIsotonic(lo, hi)` is the run-3 calibrator.
- `SymmetricModel(inner: Model)`: `fit` trains `inner` on the training rows plus their mirror (swap every `_1`/`_2` column, negate every `_diff`, `label → 1 − label`); `predict_proba(X) = (inner.p(X) + 1 − inner.p(mirror(X))) / 2`.
- `LogisticModel(C_grid=(0.01, 0.1, 1.0))`: median imputation plus missing-indicator columns, standardization, L2 logistic; `C` chosen on the chronological tail of the fold's training rows; all preprocessing fitted on the fold's training rows only.

- [ ] **Step 1: Failing tests:**

```python
def test_symmetric_model_is_exactly_symmetric(dev_frame):
    m = SymmetricModel(build_model("lightgbm", cfg)).fit(train, train["label"])
    p = m.predict_proba(test)
    p_swapped = m.predict_proba(mirror(test))
    np.testing.assert_allclose(p + p_swapped, 1.0, atol=1e-12)

def test_platt_never_outputs_a_certainty():
    raw = np.r_[np.full(50, 0.001), np.full(50, 0.999)]
    y = np.r_[np.zeros(50), np.ones(50)]
    p = Platt().fit(raw, y).predict(raw)
    assert ((p > 1e-5) & (p < 1 - 1e-5)).all()

def test_logistic_preprocessing_never_sees_scored_rows():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "date": DATES[:3], "label": [0, 1, 0]})
    scored = pd.DataFrame({"x": [100.0, np.nan], "date": DATES[3:5]})
    m = LogisticModel(features=["x"], C_grid=(1.0,)).fit(train, train["label"])
    assert m.imputer_.statistics_[0] == 2.0          # median of the training rows only
    assert m.scaler_.mean_[0] == pytest.approx(2.0)  # 100.0 never enters the fit
    m.predict_proba(scored)
    assert m.imputer_.statistics_[0] == 2.0          # prediction does not refit
```

- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: add Platt calibration, an orientation-symmetric wrapper and a logistic benchmark` and push.

### Task 10: Pre-registered stages, decision rule, re-tune, freeze

**Files:** Create `experiments/stages.py`, `tests/test_stages.py`; writes `reports/dev/stages.json`, `reports/dev/frozen.yaml`.

**Pre-registration** (`config/experiments.yaml`, committed in Task 5 before any experiment runs):

```yaml
selection_window: [2010-01-01, 2014-12-31]
metric: log_loss                      # paired per fight vs the stage incumbent, event bootstrap
decision_rule: >
  Adopt a candidate over the stage incumbent only if its paired difference in
  selection-window log loss is negative and its Holm-adjusted one-sided bootstrap
  p-value within the stage is below 0.10. Otherwise keep the incumbent (the simpler
  option on a tie). Every candidate's result is reported, adopted or not.
stage_0_retune:   # re-tune the run-3 baseline inside 2010-2014 only (same grid, same tie rule,
                  # holdout = last 20% of 2010-2014), because the frozen params saw 2015.
stage_1_features: [base+adjusted, base+recency, base+style, base+decline, base+glicko, all]
stage_2_selection: [top_k_25]          # vs the stage-1 winner's full set
stage_3_train_window: [from_2001, last_8_years]
stage_4_symmetry: [symmetric]
stage_5_calibration: [platt, none]     # vs isotonic_bounded
stage_6_model: [logistic]              # vs lightgbm, same features
final_retune: true                     # re-tune the frozen configuration inside 2010-2014
```

- [ ] **Step 1: Failing tests:** `decide(results)` applies Holm across a stage's candidates and adopts only by the rule (table-driven test with synthetic bootstrap results); every candidate is written to `stages.json` with `adopted: bool`; running a stage from a commit earlier than the pre-registration commit raises.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass.
- [ ] **Step 5:** Run stages 0–6 and the final re-tune; each run appends to `reports/dev/experiments.json`. Commit `feat: run the pre-registered development stages` with `stages.json` and push.
- [ ] **Step 6:** Write `reports/dev/frozen.yaml` (the winning `ExperimentSpec` and hyperparameters). Commit `chore: freeze the development configuration` and push **before** Task 10b.

### Task 10b: The 2015 confirmation (once)

**Files:** Create `experiments/confirm.py`; test in `tests/test_stages.py`; writes `reports/dev/confirmation.json`.

- [ ] **Step 1: Failing tests:** `confirm()` refuses to run if `confirmation.json` exists, if the tree is dirty, or if `frozen.yaml` is not committed; it scores only 2015 fights; it reports the frozen configuration *and* the run-3 baseline (re-tuned in stage 0) on the same 2015 fights, with a paired event bootstrap against each other and against the market.
- [ ] **Steps 2–4:** FAIL → implement → pass. Commit and push the code.
- [ ] **Step 5:** Run once from a clean tree. Commit `docs: publish the 2015 confirmation of the frozen development configuration` and push. **Report to user: every stage result (adopted or not), the confirmation, and its interval width.**

---

## Part D — Closing-line value (development window)

### Task 11: CLV analysis

**Files:** Create `analysis/clv.py`, `tests/test_clv.py`; writes `reports/clv.json`.

**Definitions:** For each dev-window fight with BFO open and close: `move = close_p1 − open_p1`; the model's edge at the open `e = p_model − open_p1` using the frozen configuration's out-of-sample dev predictions (selection-window walk-forward plus the confirmation year, available only after Task 10b). Report (a) the market's own improvement, open vs close log loss (sanity: close should beat open); (b) the correlation of `e` with `move` and its event-bootstrap interval; (c) mean `move` in the direction of `sign(e)` for fights with |e| > 0.05, with interval; (d) does the model add information beyond the open about the close — logistic regression of the close's direction on `logit(open)` and `logit(p_model)`, coefficient interval.

- [ ] **Step 1: Failing tests** on a synthetic frame where `move` is constructed as `0.5·e + noise`: the correlation and the directional mean recover the construction; the analysis raises if given any fight dated 2016 or later.
- [ ] **Steps 2–5:** FAIL → implement → pass → run → commit `feat: add a closing-line-value analysis on the development window` and push.

---

## Part E — Forward test

### Task 12: Cards → prediction frames

**Files:** Create `forward/cards.py`; tests in `tests/test_forward.py`.

**Interfaces:** `match_names(names: list[str], fighters, fights, aliases) -> dict[str, str | None]` — exact on normalized name, then the alias table; `None` only when no candidate has similarity ≥ 0.85 (debutant); raises listing every name with a near match but no exact one. `prediction_frame(card, history_fights, cfg) -> pd.DataFrame` appends the card's bouts to the historical fights (label NaN, no round stats) and calls `build_features`, so every feature is computed from strictly earlier fights by the same builder.

**Known limitation, documented in the README:** UFCStats is unreachable, so fights after the 2026-10-09 snapshot enter history from ESPN results only (winner and method, no round statistics). Elo, Glicko-2, records and streaks stay current; per-round rate features use the last available statistics.

- [ ] **Step 1: Failing tests:** a fixture card with one veteran under an alias, one debutant, and one near-miss name; the alias maps, the debutant is `None` with `is_debut` true in the frame, the near-miss raises.
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: build forward-test prediction frames from ESPN cards` and push.

### Task 13: Forward ledger, predict / close / settle

**Files:** Create `forward/ledger.py`, `forward/run.py`; modify `cli.py`; tests in `tests/test_forward.py`.

**Interfaces:** CLI `forward predict --event <espn_id>` writes `reports/forward/<event>/predictions.csv` (model p from the frozen configuration trained on every decisive fight through the snapshot plus ESPN results; Kalshi mid and BFO median at prediction time) and appends to `reports/forward/ledger.json`; it refuses if the event has started. `forward close --event` records Kalshi and BFO prices within 2 hours of the scheduled start. `forward settle --event` pulls ESPN results, voids replaced bouts, scores model vs Kalshi-close vs BFO-close, and writes `settlement.json`.

- [ ] **Step 1: Failing tests:** `predict` on a started event raises; a replaced bout is void, not scored; the ledger check `committed_before_start(event)` reads `git log --format=%cI -- reports/forward/<event>/predictions.csv` and fails when the first commit time is after `start_utc` (tested in a temporary git repo).
- [ ] **Steps 2–5:** FAIL → implement → pass → commit `feat: add the forward-test predict, close and settle commands` and push.

### Task 15: Scheduled forward test on GitHub Actions

**Files:** Create `.github/workflows/forward.yml`, `src/ufc_predictor/forward/schedule.py`; tests in `tests/test_forward.py`.

**Interfaces:** `schedule.due_actions(cards, ledger, now) -> list[(action, event_id)]` decides, from ESPN's calendar and the forward ledger, which events need `predict` (start between 24 h and `forward.predict_window_hours` away, no committed predictions), `close` (start within 2 h) or `settle` (finished, unsettled), and reports `missed` for an event that reached 24 h before its start with no committed predictions. CLI `forward run-due` executes them.

- [ ] **Step 1: Failing tests:** a card 30 h away with no predictions is `predict`; one 23 h away with none is `missed` (never predicted late); one 90 min away is `close`; a finished unsettled card is `settle`; every decision is a pure function of `(cards, ledger, now)`.
- [ ] **Step 2:** FAIL → implement → pass.
- [ ] **Step 3:** Workflow: cron every 3 hours plus `workflow_dispatch`; `uv sync`, `make data`, `ufc-predictor forward run-due`; commits `reports/forward/` with the user's name and GitHub noreply address and pushes; records the run id and URL in the ledger. On any failure, or a `missed` event, it opens or updates one issue labelled `forward-test` that mentions the user (GitHub also emails the failed run).
- [ ] **Step 4:** Commit `ci: schedule the forward test on GitHub Actions` and push; trigger it once with `workflow_dispatch` and confirm a green run.

### Task 14: First forward card and README

- [ ] **Step 1:** For the next card on ESPN's calendar: `forward predict`, commit `docs: commit forward predictions for <event> before it starts`, push — at least 24 hours before the scheduled start.
- [ ] **Step 2:** Within 2 hours of the start: `forward close`; commit and push.
- [ ] **Step 3:** After the event: `forward settle`; commit and push.
- [ ] **Step 4:** README Phase 8 sections, all generated from JSON with section tests: data sources and reachability; the protocol and pre-registration; every stage's candidates with adopted/rejected and intervals (including what did not help); the 2015 confirmation; CLV; the forward ledger with the stated limitation that a handful of cards cannot show an edge.
- [ ] **Step 5:** Fresh-context adversarial review subagent (leakage and overfitting only) writes `reports/review_phase8.md`; fix findings; fresh-clone run of `make setup && make data && make backtest && make report` reproduces every published figure.

---

## Expectations, stated in advance

- The selection window holds about 1,690 priced fights; a paired difference of 0.003–0.005 log loss is near the noise floor, so several candidates are expected to be rejected. That is a result, and it is reported.
- The 2015 confirmation has about 460 fights; its interval on a difference will be roughly ±0.01, so it can catch a large overfit but not certify a small gain.
- A forward test of a few cards (about 12 fights each) cannot distinguish a model from the market; it demonstrates an honest, timestamped pipeline.

## Disclosure carried into the README

The run-3 hyperparameters were tuned on a holdout that included 2015, and earlier dev diagnostics and model cards scored 2015 fights. 2015 is therefore not pristine for the *old* configuration. It is unseen by every *new* selection decision in this phase, which is what the confirmation measures; stage 0 and the final re-tune keep 2015 out of tuning from now on.
