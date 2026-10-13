"""Kalshi's public market data: the forward test's primary market price.

Kalshi lists one event per UFC bout (series ``KXUFCFIGHT``), holding one yes/no market
per fighter. The price used here is the **mid** of each fighter's best bid and ask,
normalized across the two fighters so the pair sums to one -- the exchange analogue of
removing the vig. A side with no bid or no ask has no mid, and the fight is reported
missing rather than filled with 0.5.

No account or key is needed: events, markets and candlesticks are public endpoints.
Candlesticks keep the bid/ask history after a market settles, so the closing price at a
bout's start can be read when the result is settled instead of by a job that must run
at the right minute.
"""

from __future__ import annotations

import datetime as dt
from typing import NamedTuple

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXUFCFIGHT"


class Side(NamedTuple):
    ticker: str
    bid: float
    ask: float


class KalshiFight(NamedTuple):
    event_ticker: str
    names: tuple[str, str]
    sides: tuple[Side, Side]
    volume: float


def _shape(cond: bool, what: str) -> None:
    if not cond:
        raise ValueError(f"Kalshi response shape changed: {what}")


def date_code(day: dt.date) -> str:
    """Kalshi's ticker date, e.g. ``26OCT17`` for 2026-10-17."""
    return day.strftime("%y%b%d").upper()


def events_url(*, cursor: str | None, status: str = "open") -> str:
    url = (
        f"{BASE_URL}/events?series_ticker={SERIES}&status={status}"
        "&with_nested_markets=true&limit=200"
    )
    return url + (f"&cursor={cursor}" if cursor else "")


def candlesticks_url(
    ticker: str, start: dt.datetime, end: dt.datetime, *, period_minutes: int
) -> str:
    return (
        f"{BASE_URL}/series/{SERIES}/markets/{ticker}/candlesticks"
        f"?start_ts={int(start.timestamp())}&end_ts={int(end.timestamp())}"
        f"&period_interval={period_minutes}"
    )


def _dollars(value) -> float:
    return float(value) if value not in (None, "") else 0.0


def fight_markets(fetcher, card_date: dt.date, *, status: str = "open") -> list[KalshiFight]:
    """Every two-sided bout market whose ticker carries ``card_date``."""
    code = f"-{date_code(card_date)}"
    fights: list[KalshiFight] = []
    cursor: str | None = None
    while True:
        data = fetcher.json(events_url(cursor=cursor, status=status))
        _shape(isinstance(data, dict) and "events" in data, "no events list")
        for event in data["events"]:
            if code not in event.get("event_ticker", ""):
                continue
            markets = event.get("markets") or []
            _shape(len(markets) == 2, f"{event.get('event_ticker')} has {len(markets)} markets")
            sides, names, volume = [], [], 0.0
            for m in markets:
                _shape(bool(m.get("yes_sub_title")) and bool(m.get("ticker")), "unnamed market")
                names.append(m["yes_sub_title"])
                sides.append(
                    Side(
                        m["ticker"],
                        _dollars(m.get("yes_bid_dollars")),
                        _dollars(m.get("yes_ask_dollars")),
                    )
                )
                volume += _dollars(m.get("volume_fp"))
            fights.append(KalshiFight(event["event_ticker"], tuple(names), tuple(sides), volume))
        cursor = data.get("cursor") or None
        if cursor is None:
            break
    return fights


def _quoted(side: Side) -> bool:
    return 0.0 < side.bid <= side.ask < 1.0


def mid_probabilities(fight: KalshiFight) -> dict[str, float] | None:
    """Each fighter's normalized mid, or None if either side lacks a bid or an ask."""
    if not all(_quoted(s) for s in fight.sides):
        return None
    mids = [(s.bid + s.ask) / 2 for s in fight.sides]
    total = sum(mids)
    return {name: mid / total for name, mid in zip(fight.names, mids, strict=True)}


def quote_at(
    fetcher,
    ticker: str,
    at: dt.datetime,
    *,
    lookback: dt.timedelta = dt.timedelta(hours=2),
    period_minutes: int = 1,
) -> tuple[float, float] | None:
    """The best bid and ask at the close of the last candle ending at or before ``at``."""
    data = fetcher.json(candlesticks_url(ticker, at - lookback, at, period_minutes=period_minutes))
    _shape(isinstance(data, dict) and "candlesticks" in data, "no candlesticks list")
    cutoff = int(at.timestamp())
    usable = [c for c in data["candlesticks"] if int(c["end_period_ts"]) <= cutoff]
    if not usable:
        return None
    last = max(usable, key=lambda c: int(c["end_period_ts"]))
    bid = _dollars(last.get("yes_bid", {}).get("close_dollars"))
    ask = _dollars(last.get("yes_ask", {}).get("close_dollars"))
    return (bid, ask) if 0.0 < bid <= ask < 1.0 else None
