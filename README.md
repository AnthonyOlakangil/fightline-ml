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

That dataset ships 118 columns, most of them precomputed career aggregates. Those are
**not used** — they are a leakage hazard, since it cannot be established that they exclude
the fight they describe or later ones. The loader whitelists exactly five columns
(both fighter names, the date, and both American moneylines) and a test enforces that
nothing else reaches the feature table. Every feature in this project is computed from
UFCStats with an explicit as-of join.

American odds are converted to implied probabilities and de-vigged by two-sided
normalization before any comparison. The market baseline is the vig-free favorite.

### Known coverage limits

- Odds end 2026-03-28, so fights after that date are unpriced and excluded from market
  comparison.
- Odds join to UFCStats on (date, fighter pair) at 95.1% exactly. All unmatched rows are
  name variants, not date mismatches; they are resolved via a hand-reviewed alias table,
  and anything still unresolved is written to `reports/unjoined_odds.csv` and dropped with
  a logged count — never silently fuzzy-matched.

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
