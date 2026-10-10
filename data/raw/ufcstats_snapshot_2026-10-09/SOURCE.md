# UFCStats snapshot (via public mirror)

- **Retrieved (UTC):** 2026-10-10T06:44:43+00:00
- **Upstream repository:** https://github.com/Greco1899/scrape_ufc_stats
- **Upstream license:** GPL-3.0 (upstream scraper code; no upstream code is vendored here)

UFCStats (ufcstats.com) is the canonical source for fights, per-round statistics, and
fighter attributes, but it cannot be read directly from a clean environment as of
October 2026: the site answers with a JavaScript proof-of-work challenge (a ~3 KB
"Checking your browser" stub containing an inline SHA-256 implementation) rather than
page content, and it serves HTTP only -- port 443 refuses connections. A plain
requests + BeautifulSoup scraper therefore appears to succeed while returning no data.

This project reads a committed snapshot of a public mirror instead, which also
satisfies the requirement that the graded path not depend on a live site being up.


No upstream code is vendored in this project. The CSV data published by the
upstream repository is a factual record of fight outcomes; all parsing,
normalization, and schema logic here is written for this repository.

Row counts below are asserted by `tests/test_ingest.py`, so a silent change in
a refreshed snapshot fails the suite instead of shifting results unnoticed.

## Tables

### `events.parquet`

- Event name, date, location
- Source: `https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/ufc_event_details.csv`
- Rows: **791**
- Columns: 4
- SHA-256 of source CSV: `7ff5ff0155d6a1dc55a39b4ec61ef569c6ce0f43b1f22edd730744759e7ee66a`
- Parquet size: 26 KiB

### `fight_results.parquet`

- One row per bout: outcome, method, round, time, referee
- Source: `https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/ufc_fight_results.csv`
- Rows: **8,950**
- Columns: 11
- SHA-256 of source CSV: `e7253107447bf2eaab780776401b2543ac5283d0272771b67fde3d1983be139b`
- Parquet size: 297 KiB

### `fight_stats.parquet`

- One row per fighter per round: strikes, takedowns, control time
- Source: `https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/ufc_fight_stats.csv`
- Rows: **42,062**
- Columns: 19
- SHA-256 of source CSV: `58f6f0d7ac45f299c877e16135458c33cc3c50454c948abdeec5dc8ea54d2c09`
- Parquet size: 778 KiB

### `fighter_tott.parquet`

- Fighter tale-of-the-tape: height, weight, reach, stance, date of birth
- Source: `https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/ufc_fighter_tott.csv`
- Rows: **4,629**
- Columns: 7
- SHA-256 of source CSV: `3296e67941f984cd5484488226e5a5d94de3249c63fe289fbc66d98546311ea1`
- Parquet size: 125 KiB
