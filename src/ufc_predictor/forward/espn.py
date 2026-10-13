"""ESPN's public scoreboard JSON: the UFC calendar, card line-ups and results.

Unofficial and undocumented, so every read checks the shape it relies on and raises on
anything else. The Dana White's Contender Series shares the UFC scoreboard; it is not a
UFC event in the UFCStats history this model is trained on, so it is skipped.

Times are UTC. A card's ``date`` is the US Eastern calendar date of its first bout, the
convention Kalshi's tickers and UFC's own listings use.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
EASTERN = ZoneInfo("America/New_York")

#: ESPN's result detail text -> the method classes used in the fights table.
METHODS = {
    "Unofficial Winner Kotko": "KO/TKO",
    "Unofficial Winner Submission": "SUB",
    "Unofficial Winner Decision": "DEC",
}


@dataclass(frozen=True)
class CalendarEntry:
    event_id: str
    name: str
    start_utc: dt.datetime


@dataclass(frozen=True)
class Bout:
    competition_id: str
    fighter_a: str
    fighter_b: str
    start_utc: dt.datetime  # the start of the bout's card segment, not the bout itself
    weight_class: str
    scheduled_rounds: int | None
    status: str


@dataclass(frozen=True)
class Card:
    event_id: str
    name: str
    date: dt.date
    start_utc: dt.datetime
    bouts: tuple[Bout, ...]


@dataclass(frozen=True)
class Result:
    competition_id: str
    pair: tuple[str, str]
    completed: bool
    winner: str | None
    method_class: str | None
    finish_round: int | None


def _shape(cond: bool, what: str) -> None:
    if not cond:
        raise ValueError(f"ESPN scoreboard shape changed: {what}")


def parse_utc(text: str) -> dt.datetime:
    """ESPN writes ``2026-10-18T00:00Z``."""
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(dt.UTC)


def is_ufc_event(name: str) -> bool:
    return name.startswith("UFC") and "Contender Series" not in name


def calendar(fetcher) -> list[CalendarEntry]:
    """Every UFC event on ESPN's season calendar, in start order."""
    data = fetcher.json(SCOREBOARD_URL)
    leagues = data.get("leagues") if isinstance(data, dict) else None
    _shape(bool(leagues) and "calendar" in leagues[0], "no leagues[0].calendar")
    out = []
    for item in leagues[0]["calendar"]:
        _shape(
            isinstance(item, dict) and {"label", "startDate", "event"} <= item.keys(),
            "a calendar entry lacks label/startDate/event",
        )
        ref = item["event"].get("$ref", "")
        _shape("/events/" in ref, f"calendar entry {item['label']!r} has no event reference")
        event_id = ref.split("/events/")[1].split("?")[0]
        if is_ufc_event(item["label"]):
            out.append(CalendarEntry(event_id, item["label"], parse_utc(item["startDate"])))
    return sorted(out, key=lambda e: e.start_utc)


def _range_url(entry: CalendarEntry) -> str:
    # ESPN files an event under its US date, which can be the day before its UTC start.
    end = entry.start_utc.date()
    start = end - dt.timedelta(days=1)
    return f"{SCOREBOARD_URL}?dates={start:%Y%m%d}-{end:%Y%m%d}"


def _event(fetcher, entry: CalendarEntry) -> dict[str, Any]:
    data = fetcher.json(_range_url(entry))
    _shape(isinstance(data, dict) and "events" in data, "no events list")
    matches = [e for e in data["events"] if str(e.get("id")) == entry.event_id]
    _shape(len(matches) == 1, f"event {entry.event_id} not found exactly once")
    return matches[0]


def _names(comp: dict[str, Any]) -> list[dict[str, Any]]:
    competitors = comp.get("competitors") or []
    _shape(len(competitors) == 2, f"competition {comp.get('id')} does not have two competitors")
    competitors = sorted(competitors, key=lambda c: c.get("order", 0))
    for c in competitors:
        _shape(bool(c.get("athlete", {}).get("displayName")), "a competitor has no name")
    return competitors


def card(fetcher, entry: CalendarEntry) -> Card:
    """The card's bouts, each with its segment start, weight class and scheduled rounds."""
    event = _event(fetcher, entry)
    bouts = []
    for comp in event.get("competitions") or []:
        a, b = _names(comp)
        periods = comp.get("format", {}).get("regulation", {}).get("periods")
        bouts.append(
            Bout(
                competition_id=str(comp["id"]),
                fighter_a=a["athlete"]["displayName"],
                fighter_b=b["athlete"]["displayName"],
                start_utc=parse_utc(comp["date"]),
                weight_class=comp.get("type", {}).get("abbreviation", ""),
                scheduled_rounds=int(periods) if periods else None,
                status=comp.get("status", {}).get("type", {}).get("name", ""),
            )
        )
    _shape(bool(bouts), f"event {entry.event_id} lists no bouts")
    start = min(b.start_utc for b in bouts)
    return Card(
        event_id=entry.event_id,
        name=event.get("name", entry.name),
        date=start.astimezone(EASTERN).date(),
        start_utc=start,
        bouts=tuple(bouts),
    )


def upcoming_cards(fetcher, now: dt.datetime, *, horizon_days: int) -> list[Card]:
    """Cards whose first bout starts within ``horizon_days`` after ``now`` (or has started
    in the last two days, so a card under way is still visible)."""
    lo, hi = now - dt.timedelta(days=2), now + dt.timedelta(days=horizon_days)
    return [card(fetcher, e) for e in calendar(fetcher) if lo <= e.start_utc <= hi]


def results(fetcher, entry: CalendarEntry) -> list[Result]:
    """Every bout's outcome. A completed bout with no winner is a draw or no contest."""
    event = _event(fetcher, entry)
    out = []
    for comp in event.get("competitions") or []:
        a, b = _names(comp)
        status = comp.get("status", {})
        completed = bool(status.get("type", {}).get("completed"))
        winners = [c["athlete"]["displayName"] for c in (a, b) if c.get("winner")]
        _shape(len(winners) <= 1, f"competition {comp['id']} has two winners")
        method = None
        if winners:
            texts = [d.get("type", {}).get("text", "") for d in comp.get("details") or []]
            found = [METHODS[t] for t in texts if t in METHODS]
            _shape(len(found) == 1, f"competition {comp['id']} has no single method ({texts})")
            method = found[0]
        out.append(
            Result(
                competition_id=str(comp["id"]),
                pair=(a["athlete"]["displayName"], b["athlete"]["displayName"]),
                completed=completed,
                winner=winners[0] if winners else None,
                method_class=method,
                finish_round=int(status["period"]) if completed and status.get("period") else None,
            )
        )
    return out
