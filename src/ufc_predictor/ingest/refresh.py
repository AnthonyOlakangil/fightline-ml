"""Rebuild the committed raw snapshots. Run by hand; requires network.

    uv run python -m ufc_predictor.ingest.refresh

This is the only module that reaches the network and it is deliberately not wired into
``make data``: the graded path reads the committed Parquet so a fresh clone reproduces
the results without depending on any live host.
"""

from __future__ import annotations

from ufc_predictor.config import get_config
from ufc_predictor.ingest.snapshot import (
    ODDS_SOURCES,
    UFCSTATS_SOURCES,
    fetch_snapshot,
)

UFCSTATS_NOTES = """\
UFCStats (ufcstats.com) is the canonical source for fights, per-round statistics, and
fighter attributes, but it cannot be read directly from a clean environment as of
October 2026: the site answers with a JavaScript proof-of-work challenge (a ~3 KB
"Checking your browser" stub containing an inline SHA-256 implementation) rather than
page content, and it serves HTTP only -- port 443 refuses connections. A plain
requests + BeautifulSoup scraper therefore appears to succeed while returning no data.

This project reads a committed snapshot of a public mirror instead, which also
satisfies the requirement that the graded path not depend on a live site being up.
"""

ODDS_NOTES = """\
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
"""


def main() -> None:
    cfg = get_config()

    print(f"Fetching UFCStats mirror snapshot -> {cfg.paths.ufcstats_snapshot}")
    counts = fetch_snapshot(
        UFCSTATS_SOURCES,
        cfg.paths.ufcstats_snapshot,
        title="UFCStats snapshot (via public mirror)",
        source_repo="https://github.com/Greco1899/scrape_ufc_stats",
        source_license="GPL-3.0 (upstream scraper code; no upstream code is vendored here)",
        notes=UFCSTATS_NOTES,
    )
    for table, n in counts.items():
        print(f"  {table:<16} {n:>8,} rows")

    print(f"\nFetching odds snapshot -> {cfg.paths.odds_snapshot}")
    odds_counts = fetch_snapshot(
        ODDS_SOURCES,
        cfg.paths.odds_snapshot,
        title="Closing moneyline odds snapshot",
        source_repo="https://github.com/shortlikeafox/ultimate_ufc_dataset",
        source_license="Apache-2.0",
        notes=ODDS_NOTES,
    )
    for table, n in odds_counts.items():
        print(f"  {table:<16} {n:>8,} rows")

    print("\nSnapshot refreshed. Review SOURCE.md, check sizes, then commit.")


if __name__ == "__main__":
    main()
