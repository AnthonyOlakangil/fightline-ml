"""The forward test: live clients (ESPN cards and results, Kalshi prices, BestFightOdds).

Offline: every client reads through a fetcher, and these tests hand it recorded
fixtures from ``tests/fixtures/{espn,kalshi,bfo}/`` (2026-10-13). ``requests.get`` is
replaced with a function that raises, so nothing here can reach the network.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from ufc_predictor.forward import bfo_live, espn, http, kalshi

FIXTURES = Path(__file__).parent / "fixtures"
UTC = dt.UTC


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(http.requests, "get", refuse)


class FixtureFetcher:
    """Serves recorded bodies by URL; an unknown URL fails the test."""

    def __init__(self, routes: dict[str, str]):
        self.routes = routes
        self.requested: list[str] = []

    def text(self, url: str) -> str:
        self.requested.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected URL {url}")
        return (FIXTURES / self.routes[url]).read_text(encoding="utf-8")

    def json(self, url: str):
        return json.loads(self.text(url))


ESPN_ROUTES = {
    espn.SCOREBOARD_URL: "espn/scoreboard_calendar.json",
    espn.SCOREBOARD_URL + "?dates=20261017-20261018": "espn/scoreboard_20261017-20261018.json",
    espn.SCOREBOARD_URL + "?dates=20261010-20261011": "espn/scoreboard_20261010-20261011.json",
}


# ---------------------------------------------------------------------------
# ESPN
# ---------------------------------------------------------------------------


def test_calendar_lists_ufc_events_and_skips_the_contender_series():
    entries = espn.calendar(FixtureFetcher(ESPN_ROUTES))
    names = [e.name for e in entries]
    assert "UFC Fight Night: Buckley vs. Malott" in names
    assert not any("Contender Series" in n for n in names)
    buckley = next(e for e in entries if e.event_id == "600060773")
    assert buckley.start_utc == dt.datetime(2026, 10, 18, 0, 0, tzinfo=UTC)


def test_card_lists_every_bout_with_its_segment_start():
    entry = next(e for e in espn.calendar(FixtureFetcher(ESPN_ROUTES)) if e.event_id == "600060773")
    card = espn.card(FixtureFetcher(ESPN_ROUTES), entry)
    assert card.name == "UFC Fight Night: Buckley vs. Malott"
    assert len(card.bouts) == 13
    assert card.start_utc == dt.datetime(2026, 10, 17, 21, 0, tzinfo=UTC)
    assert card.date == dt.date(2026, 10, 17)  # US Eastern calendar date of the first bout
    main = next(
        b for b in card.bouts if {b.fighter_a, b.fighter_b} == {"Mike Malott", "Joaquin Buckley"}
    )
    assert main.start_utc == dt.datetime(2026, 10, 18, 0, 0, tzinfo=UTC)
    assert main.scheduled_rounds == 5
    assert all(b.status == "STATUS_SCHEDULED" for b in card.bouts)


def test_results_read_winner_method_and_draws():
    entry = espn.CalendarEntry(
        "600061541", "UFC Fight Night: Allen vs. Duncan", dt.datetime(2026, 10, 11, tzinfo=UTC)
    )
    results = {frozenset(r.pair): r for r in espn.results(FixtureFetcher(ESPN_ROUTES), entry)}
    main = results[frozenset({"Brendan Allen", "Christian Leroy Duncan"})]
    assert (main.winner, main.method_class, main.finish_round) == ("Brendan Allen", "SUB", 5)
    ko = results[frozenset({"Allen Frye Jr.", "RJ Harris"})]
    assert ko.method_class == "KO/TKO"
    draw = results[frozenset({"Melissa Gatto", "Ernesta Kareckaite"})]
    assert draw.completed and draw.winner is None
    assert all(r.completed for r in results.values())


def test_an_unexpected_espn_shape_raises():
    class Broken(FixtureFetcher):
        def json(self, url):
            return {"leagues": []}

    with pytest.raises(ValueError, match="ESPN"):
        espn.calendar(Broken({}))


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

KALSHI_ROUTES = {
    kalshi.events_url(cursor=None): "kalshi/events_open.json",
}


def test_date_code_matches_kalshi_tickers():
    assert kalshi.date_code(dt.date(2026, 10, 17)) == "26OCT17"


def test_fight_prices_are_mids_normalized_across_the_two_fighters():
    fights = kalshi.fight_markets(FixtureFetcher(KALSHI_ROUTES), dt.date(2026, 10, 17))
    assert len(fights) == 13
    main = next(f for f in fights if set(f.names) == {"Joaquin Buckley", "Mike Malott"})
    p = kalshi.mid_probabilities(main)
    assert p is not None
    assert sum(p.values()) == pytest.approx(1.0)
    for name, side in zip(main.names, main.sides, strict=True):
        mid = (side.bid + side.ask) / 2
        assert side.bid <= mid <= side.ask
        assert p[name] == pytest.approx(mid / sum((s.bid + s.ask) / 2 for s in main.sides))


def test_a_market_without_a_bid_or_ask_is_missing_not_even():
    side_ok = kalshi.Side(ticker="A", bid=0.55, ask=0.57)
    no_bid = kalshi.Side(ticker="B", bid=0.0, ask=0.46)
    fight = kalshi.KalshiFight("EV", ("Alpha", "Beta"), (side_ok, no_bid), volume=10.0)
    assert kalshi.mid_probabilities(fight) is None
    no_ask = kalshi.Side(ticker="B", bid=0.43, ask=1.0)
    assert kalshi.mid_probabilities(fight._replace(sides=(side_ok, no_ask))) is None


def test_close_mid_reads_the_last_candle_at_or_before_the_cutoff():
    ticker = "KXUFCFIGHT-26OCT10ALLDUN-ALL"
    at = dt.datetime(2026, 10, 11, 1, 0, tzinfo=UTC)
    url = kalshi.candlesticks_url(ticker, at - dt.timedelta(hours=7), at, period_minutes=60)
    fetcher = FixtureFetcher({url: "kalshi/candlesticks_ALLDUN-ALL.json"})
    bid, ask = kalshi.quote_at(
        fetcher, ticker, at, lookback=dt.timedelta(hours=7), period_minutes=60
    )
    assert (bid, ask) == (0.51, 0.52)


# ---------------------------------------------------------------------------
# BestFightOdds (cross-check)
# ---------------------------------------------------------------------------


def test_bfo_current_odds_take_the_median_over_sportsbooks_only():
    fetcher = FixtureFetcher({bfo_live.HOME_URL: "bfo/homepage.html"})
    odds = bfo_live.current_odds(fetcher)
    row = odds.loc[
        (odds["fighter"] == "Joaquin Buckley") & (odds["opponent"] == "Mike Malott")
    ].iloc[0]
    # FanDuel -102, Caesars +105, BetRivers +108, BetWay +100, Unibet +108 (Kalshi and
    # Polymarket are exchanges and are excluded from the sportsbook consensus).
    assert row["n_books"] == 5
    assert row["median_odds"] == 105
    assert row["best_odds"] == 108
    assert row["event_date_text"] == "October 18th"
    assert set(odds["fighter"]) >= {"Mike Malott", "Cristian Perez", "Kyle Nelson"}
