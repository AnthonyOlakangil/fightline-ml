# Closing moneyline odds snapshot

- **Retrieved (UTC):** 2026-10-10T06:44:47+00:00
- **Upstream repository:** https://github.com/shortlikeafox/ultimate_ufc_dataset
- **Upstream license:** Apache-2.0

Closing moneylines. The upstream dataset documents neither the sportsbook nor the
capture timing, so the timing was verified independently: matched against
BestFightOdds per-book at-close data (iankotliar/UFC_Final,
`moneyline_data_at_close.zip`) over 3,812 fights, these prices agree with the closing
consensus to a median absolute difference of 5 American odds points -- 71.5% within 10
points, 88.2% within 25, and 97.4% agreement on which fighter was favoured. Mid-week
captures from the same upstream project's `upcoming.csv` differ substantially more
(e.g. Oliveira vs Chandler at UFC 309: -258 mid-week vs -130 final), so these are
closing rather than opening lines. Post-fight assignment is ruled out structurally
(the upstream project publishes odds for bouts that have not happened yet) and
empirically (the favourite wins ~66% of the time, not ~100%).

Coverage runs 2010-03-21 to 2026-03-28. The upstream project's last update was
2026-04-01, so fights after that date are unpriced and are excluded from any
market comparison.

Only five columns are read: R_fighter, B_fighter, date, R_odds, B_odds. The remaining
113 upstream columns are precomputed career aggregates whose point-in-time correctness
cannot be established, and are dropped here at the snapshot boundary.


No upstream code is vendored in this project. The CSV data published by the
upstream repository is a factual record of fight outcomes; all parsing,
normalization, and schema logic here is written for this repository.

Row counts below are asserted by `tests/test_ingest.py`, so a silent change in
a refreshed snapshot fails the suite instead of shifting results unnoticed.

## Tables

### `odds_master.parquet`

- Historical fights with closing moneylines (only 5 columns are ever used)
- Source: `https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv`
- Rows: **7,177**
- Columns: 5
- SHA-256 of source CSV: `b8aa6604539d5981bba561cb65da26133567732145ca48efbec2d9423b081937`
- Parquet size: 77 KiB
