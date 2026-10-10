# ufc-predictor

A reproducible UFC fight-outcome model, evaluated walk-forward against the vig-free
betting market, with a Monte-Carlo parlay pricer.

> **Research backtest only. This is not betting advice, and nothing here should be used
> to wager.** The project exists to answer a measurement question honestly, not to
> produce picks.

> **Status: under construction.** Phase 0 (scaffold) is complete. This README will carry
> the results table, calibration plot, and bankroll chart once Phase 4 generates them.
> **No metrics appear below yet, by design** — every number in this README is produced by
> `make report` from `reports/metrics.json` and is regenerable from a fresh clone. None
> are typed by hand.

---

## The question

Closing betting lines are a strong, hard-to-beat forecast: the market aggregates
information from people with money at risk. So the question is not "can a model predict
UFC fights" — it is:

**Can a model built only from historical fight statistics beat the vig-free closing line,
out of sample, after honest accounting for uncertainty?**

The answer may well be no. A rigorous negative result is reported as readily as a
positive one, with confidence intervals on the difference rather than two point estimates
compared by eye.

---

## Non-negotiables

This project is built around four rules, because they are what make the result mean
anything:

1. **Point-in-time correctness.** Every feature for a fight is computable using only data
   dated strictly before that fight. Features are built with as-of joins over each
   fighter's prior bouts.
2. **Leakage tests are mandatory.** `tests/test_leakage.py` asserts no feature row uses an
   event dated on or after the fight it describes, and that the walk-forward training
   cutoff is respected on every fold. These were written *before* any model existed.
3. **Determinism.** All randomness is seeded from `config/config.yaml`. Same seed, same
   numbers. No seed literals in source.
4. **Fail loudly.** An unexpected data shape, a row-dropping join, or an unparseable date
   raises. Row counts are logged at every pipeline stage.

---

## Data sources

### Fight data — UFCStats, via a committed mirror snapshot

The canonical source for fights, per-round statistics, and fighter attributes is
[UFCStats](http://ufcstats.com). **It cannot be scraped directly from a clean
environment as of October 2026:** the site is behind a JavaScript proof-of-work
anti-bot challenge. A plain `requests` + BeautifulSoup fetch returns a ~3 KB
"Checking your browser…" stub containing an inline SHA-256 implementation, not fight
data — so a naive scraper appears to run successfully while producing nothing. The site
also serves HTTP only; port 443 refuses connections.

Rather than work around that, this project takes the route its own engineering rules
already require — the graded path must not depend on a live site being up — and reads a
**dated, committed snapshot** of a public UFCStats mirror:

| | |
|---|---|
| **Source** | [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats) |
| **License** | GPL-3.0 (applies to that project's scraper code) |
| **Snapshot date** | 2026-10-09 (most recent event: UFC 332, 2026-10-03) |
| **Committed at** | `data/raw/ufcstats_snapshot_2026-10-09/` |

**No code from that repository is vendored or copied.** Its license covers its scraper
implementation; this project reads only the published CSV data — factual records of fight
results — and all parsing, normalization, and schema logic in `src/ufc_predictor/ingest/`
is written from scratch for this repo. Provenance, retrieval date, row counts, and
license are recorded in the snapshot's `SOURCE.md`.

### Closing odds

| | |
|---|---|
| **Source** | [`shortlikeafox/ultimate_ufc_dataset`](https://github.com/shortlikeafox/ultimate_ufc_dataset) |
| **License** | Apache-2.0 |
| **Coverage** | 2010-03-21 → 2026-03-28 |
| **Committed at** | `data/raw/odds_snapshot_2026-10-09/` |
| **Line type** | **Closing** — verified independently, see below |

#### Verifying these are closing lines

The upstream dataset documents neither the sportsbook nor when the price was captured,
and "the market baseline" means nothing if the prices are opening lines or, worse,
assigned after the fact. So the timing was established empirically rather than assumed:

- **Not post-fight.** Structurally, the upstream project publishes odds for bouts that
  have not happened yet. Empirically, the favourite wins about 67% of the time — if the
  prices encoded known results it would be ~100%.
- **Closing, not opening.** Matched against an independent source of per-sportsbook
  *at-close* moneylines ([`iankotliar/UFC_Final`](https://github.com/iankotliar/UFC_Final),
  `moneyline_data_at_close.zip`, derived from BestFightOdds across 12 books) over
  **3,812 jointly-covered fights**, these prices agree with the closing consensus to a
  **median absolute difference of 5 American odds points** — 71.5% within 10 points,
  88.2% within 25, and 97.4% agreement on which fighter was favoured.
- **Contrast with mid-week prices.** The same upstream project's `upcoming.csv`, captured
  during fight week, diverges much further from the committed values (Oliveira vs
  Chandler at UFC 309: −258 mid-week against −130 final), confirming the committed file
  is not a mid-week capture.

That dataset ships 118 columns, most of them precomputed career aggregates. Those are
**not used** — they are a leakage hazard, since it cannot be established that they exclude
the fight they describe or later ones. The loader whitelists exactly five columns
(both fighter names, the date, and both American moneylines) and a test enforces that
nothing else reaches the feature table. Every feature in this project is computed from
UFCStats with an explicit as-of join.

American odds are converted to implied probabilities and de-vigged by two-sided
normalization before any comparison. The market baseline is the vig-free favorite.

### Joining the two sources

Both sources key fights on fighter *names*, so the join is a correctness problem rather
than a formatting one: attaching one fighter's closing line to another fighter's result
would corrupt the market baseline without changing any row count. The policy is that a
**dropped row always beats a guessed pairing**, implemented as three tiers:

1. **Exact** match on (event date, normalized fighter pair) — 6,527 fights.
2. **Alias table** (`src/ufc_predictor/ingest/aliases.csv`) — 296 fights. The two sources
   disagree on names for three reasons: legal and married name changes (UFCStats carries
   a fighter's *current* name while the odds data uses the name used at the time, so
   "Tecia Torres" is "Tecia Pennington", "Cris Cyborg" is "Cristiane Justino", "Katlyn
   Chookagian" is "Katlyn Cerminara"), differing word order for many Chinese and
   Mongolian names, and ring names ("Rampage Jackson" / "Quinton Jackson").
   Each of the 64 aliases is **corroborated by at least two independent fights against
   different opponents** — Bobby Green → King Green is implied by 19 — and any name
   implying two different targets is rejected outright. Proposals are generated with
   evidence by `python -m ufc_predictor.ingest.propose_aliases` and reviewed by hand;
   single-evidence proposals were **not** accepted.
3. **Fuzzy**, a near-inert backstop requiring 0.92 similarity on *both* fighters plus a
   margin over the runner-up candidate. It resolves **1** fight in the whole dataset. A
   one-character surname typo scores ≈0.74 and is deliberately rejected, because a
   corroborated alias is far safer than a similarity score.

Every fuzzy match is logged to `reports/fuzzy_matched_odds.csv` and every failure to
`reports/unjoined_odds.csv`, both for audit. **80 priced fights (1.16%) remain unjoined**
and are dropped with a logged count; the pipeline raises if that share exceeds 5%.

### Known coverage limits

Reported honestly because they bound what the results can claim:

- **Odds end 2026-03-28** (the upstream project's last update was 2026-04-01), so fights
  after that date are unpriced and excluded from any market comparison.
- **56 bouts** are dropped because a fighter's name maps to several UFCStats ids (nine
  names are shared by multiple fighters, e.g. two different "Mike Davis"). Merging two
  careers would corrupt every point-in-time feature for both, so these are dropped rather
  than guessed.
- **27 bouts** are dropped because the upstream event file omits three events that its own
  fight file references (`Road to UFC 4.6`, `Fight Night: Grasso vs. Shevchenko 2`,
  `Fight Night: Lopes vs. Silva`). Without an event date a fight cannot be placed in time,
  and no date is invented.
- **20 priced rows** are dropped for an arithmetically impossible overround — both sides
  quoted plus-money (e.g. +176 / +275, a notional 37% arbitrage). A real two-sided book
  must price above 100%, so these are corrupted quotes.
- **30 fighter slots** use a name-derived id because the fighter is absent from the
  tale-of-the-tape table. At worst this splits one career into two, understating history
  rather than inventing it.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is fetched automatically.

```bash
git clone https://github.com/AnthonyOlakangil/ufc-predictor.git
cd ufc-predictor
make setup      # venv + pinned dependencies
make data       # snapshot -> normalized Parquet (no network)
make backtest   # walk-forward evaluation -> reports/metrics.json
make report     # charts + README figures
```

Other targets: `make test` (full suite), `make test-leakage` (point-in-time suite only),
`make lint`, `make app` (Streamlit demo). `make help` lists everything.

---

## Repository layout

```
config/            seeds, paths, date windows, hyperparameters
data/raw/          committed dated snapshots (the repo runs without re-scraping)
data/processed/    derived Parquet (generated; not committed)
src/ufc_predictor/
  ingest/          loaders, normalization, odds + vig removal, alias join
  storage/         DuckDB helpers
  features/        point-in-time feature builder, Elo
  models/          market baseline, Elo, LightGBM
  backtest/        walk-forward engine, metrics, bootstrap CIs
  sim/             method/round model, Monte Carlo, parlay pricing, Kelly
  cli.py           Typer entrypoint
app/               Streamlit demo
tests/             includes the mandatory test_leakage.py
reports/           generated charts and metrics.json
docs/              implementation plan
```

---

## Methodology, results, and limitations

To be written in Phases 2–7, generated from code. Planned contents: the point-in-time
feature construction, the three models and their shared interface, the walk-forward
protocol and its held-out window, a results table with bootstrap confidence intervals
against the market baseline, a calibration plot, a simulated-bankroll plot with a CI band,
and an honest limitations section.

The full implementation plan is in
[`docs/superpowers/plans/2026-10-09-ufc-predictor.md`](docs/superpowers/plans/2026-10-09-ufc-predictor.md).

---

## License

MIT (this project's own code). Third-party data retains the licenses listed above.
