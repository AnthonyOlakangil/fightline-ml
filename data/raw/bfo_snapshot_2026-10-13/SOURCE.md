# BestFightOdds historical opening and closing lines

- **Source:** https://www.bestfightodds.com (fighter pages: `Open` and `Closing range`).
- **Crawled:** 2026-10-13 by `uv run ufc-predictor bfo` (`src/ufc_predictor/ingest/bfo.py`).
- **Window:** UFC events dated 2010-01-01 to 2015-12-31 (the development window only).
- **Path:** `/sitemap-events.xml` (event dates) -> 212 UFC event pages -> 1076
  fighter pages. One request per 5 s, identifying User-Agent, every response
  cached and never re-requested.
- **Rows:** 2,269 bouts (2,269 seen on both fighters' pages and identical);
  2,269 have both sides' opening price and closing range.
- **robots.txt** (fetched 2026-10-13):

```
User-agent: *
Allow: /

Sitemap: https://www.bestfightodds.com/sitemap.xml
Sitemap: https://www.bestfightodds.com/news-sitemap.xml
```

- **Terms of service** (`/terms`, fetched 2026-10-13): an "as is" accuracy disclaimer only;
  silent on automated access.

Columns: `fighter`, `fighter_url`, `opponent`, `opponent_url`, `event`, `event_id`, `event_url`, `date`, `open_fighter`, `open_opponent`, `close_lo_fighter`, `close_hi_fighter`, `close_lo_opponent`, `close_hi_opponent`. Prices are American odds; a blank cell on the site is missing here,
never zero. `close_lo_*`/`close_hi_*` are the lowest and highest closing price across
the books BestFightOdds tracked for that bout.

The full audit is in `audit.json`.
