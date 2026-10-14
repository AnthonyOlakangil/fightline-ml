"""BestFightOdds: parsing recorded pages, the response cache, and the snapshot crawl.

Every test runs offline against fixtures in ``tests/fixtures/bfo/``, recorded from the
live site on 2026-10-13 and trimmed. ``requests.get`` is replaced with a function that
raises, so a test that reached the network would fail rather than pass by luck.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from ufc_predictor.ingest import bfo
from ufc_predictor.ingest.bfo import (
    fetch,
    parse_event_page,
    parse_events_sitemap,
    parse_fighter_page,
)

FIXTURES = Path(__file__).parent / "fixtures" / "bfo"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(bfo.requests, "get", refuse)


def fighter_html() -> str:
    return (FIXTURES / "fighter_page.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fighter pages
# ---------------------------------------------------------------------------


def test_parses_open_and_closing_range_for_both_sides():
    rows = parse_fighter_page(fighter_html())
    shogun = rows.loc[rows["opponent"] == "Mauricio Rua"].iloc[0]
    assert shogun["date"] == dt.date(2011, 3, 19)
    assert shogun["open_opponent"] == 175
    assert (shogun["close_lo_opponent"], shogun["close_hi_opponent"]) == (195, 195)
    assert rows["date"].notna().all() and len(rows) == 6


def test_fighter_side_and_event_identity_are_parsed():
    rows = parse_fighter_page(fighter_html())
    shogun = rows.loc[rows["opponent"] == "Mauricio Rua"].iloc[0]
    assert shogun["fighter"] == "Jon Jones"
    assert shogun["fighter_url"] == "/fighters/jon-jones-819"
    assert shogun["opponent_url"] == "/fighters/mauricio-rua-110"
    assert shogun["event"] == "UFC 128: Shogun vs. Jones"
    assert shogun["event_id"] == 346
    assert (shogun["open_fighter"], shogun["close_lo_fighter"], shogun["close_hi_fighter"]) == (
        -225,
        -215,
        -215,
    )


def test_a_missing_price_is_missing_not_zero():
    html = fighter_html().replace('<span id="oID117">+175</span>', '<span id="oID117"></span>')
    rows = parse_fighter_page(html)
    shogun = rows.loc[rows["opponent"] == "Mauricio Rua"].iloc[0]
    assert pd.isna(shogun["open_opponent"])
    assert shogun["close_lo_opponent"] == 195


def test_the_rumored_matchups_section_label_is_skipped():
    """Pages with unbooked matchups carry a section label row before them."""
    label = (
        '<tr class="event-header rumored-header"><td colspan="8" scope="row">Rumored &amp; '
        "future matchups <span>· not yet officially booked</span></td></tr>"
    )
    html = fighter_html()
    first_block = html.index('<tr class="event-header')
    rows = parse_fighter_page(html[:first_block] + label + html[first_block:])
    assert len(rows) == 6


def test_the_undated_future_events_block_is_kept_undated():
    """BestFightOdds files unbooked bouts under a "Future Events" pseudo-event with no date."""
    html = fighter_html()
    start = html.index('<tr class="event-header')
    end = html.index('<tr class="event-header', start + 1)
    block = html[start:end].replace(
        '<a href="/events/ufc-140-jones-vs-machida-451">UFC 140: Jones vs. Machida</a> Dec 10th 2011',
        '<a href="/events/future-events-197">Future Events</a>',
    )
    rows = parse_fighter_page(html[:start] + block + html[start:])
    future = rows.loc[rows["event_id"] == 197]
    assert len(future) == 1 and future["date"].isna().all()
    assert rows.loc[rows["event_id"] != 197, "date"].notna().all()


def test_unexpected_page_shape_raises():
    with pytest.raises(ValueError, match="BestFightOdds page shape"):
        parse_fighter_page("<html><body>maintenance</body></html>")


def test_an_unparseable_date_raises():
    html = fighter_html().replace("Mar 19th 2011</td>", "sometime in 2011</td>")
    with pytest.raises(ValueError, match="date"):
        parse_fighter_page(html)


def test_a_bout_block_missing_its_opponent_row_raises():
    html = fighter_html()
    start = html.index('<a href="/fighters/mauricio-rua-110">')
    row_start = html.rindex("<tr>", 0, start)
    row_end = html.index("</tr>", start) + len("</tr>")
    with pytest.raises(ValueError, match="BestFightOdds page shape"):
        parse_fighter_page(html[:row_start] + html[row_end:])


# ---------------------------------------------------------------------------
# Event pages and the sitemap
# ---------------------------------------------------------------------------


def test_event_page_pairs_fighters_by_matchup():
    bouts = parse_event_page((FIXTURES / "event_page.html").read_text(encoding="utf-8"))
    assert list(bouts["matchup_id"]) == [2192, 2294, 2324]
    first = bouts.iloc[0]
    assert (first["fighter_a"], first["fighter_b"]) == ("Dan Hardy", "Georges St-Pierre")
    assert first["fighter_b_url"] == "/fighters/georges-st-pierre-80"


def test_event_page_skips_the_event_props_row():
    """Event pages carry an "Event props" row (id ``mu-e<n>``) that is not a bout."""
    html = (FIXTURES / "event_page.html").read_text(encoding="utf-8")
    props = (
        '<tr class="eventprop" id="mu-e1013"><th scope="row"><a data-mu="1013" href="#">'
        'Event props</a></th></tr><tr style="display: none;"></tr>'
    )
    bouts = parse_event_page(html.replace("</table>", props + "</table>"))
    assert list(bouts["matchup_id"]) == [2192, 2294, 2324]


def test_event_page_without_an_odds_table_raises():
    with pytest.raises(ValueError, match="BestFightOdds page shape"):
        parse_event_page("<html><body><p>no table</p></body></html>")


def test_sitemap_lists_ufc_events_with_their_dates():
    events = parse_events_sitemap((FIXTURES / "sitemap_events.xml").read_text(encoding="utf-8"))
    assert len(events) == 8
    ufc = events.loc[events["is_ufc"]]
    assert set(ufc["slug"]) >= {"ufc-111-st-pierre-vs-hardy-236", "ufc-194-aldo-vs-mcgregor-1005"}
    assert "bellator-3468" not in set(ufc["slug"])
    row = events.loc[events["slug"] == "ufc-111-st-pierre-vs-hardy-236"].iloc[0]
    assert row["event_id"] == 236
    assert row["date"] == dt.date(2010, 3, 27)


def test_events_in_window_keeps_only_ufc_events_inside_it():
    events = parse_events_sitemap((FIXTURES / "sitemap_events.xml").read_text(encoding="utf-8"))
    picked = bfo.events_in_window(events, dt.date(2010, 1, 1), dt.date(2015, 12, 31))
    assert list(picked["slug"]) == [
        "ufc-111-st-pierre-vs-hardy-236",
        "ufc-fight-night-80-vanzant-vs-namajunas-1023",
        "ufc-194-aldo-vs-mcgregor-1005",
    ]


# ---------------------------------------------------------------------------
# The response cache
# ---------------------------------------------------------------------------


def test_fetch_serves_a_cached_url_without_a_request_or_a_delay(tmp_path, monkeypatch):
    url = "https://www.bestfightodds.com/fighters/jon-jones-819"
    bfo.cache_path(url, tmp_path).write_text("<html>cached</html>", encoding="utf-8")
    slept: list[float] = []
    monkeypatch.setattr(bfo.time, "sleep", slept.append)
    assert fetch(url, tmp_path, delay_s=5.0) == "<html>cached</html>"
    assert slept == []


def test_fetch_sleeps_then_caches_on_a_miss(tmp_path, monkeypatch):
    url = "https://www.bestfightodds.com/fighters/mauricio-rua-110"
    calls: list[tuple[str, dict]] = []
    slept: list[float] = []

    class Response:
        status_code = 200
        text = "<html>fresh</html>"

        def raise_for_status(self):
            return None

    def fake_get(u, headers, timeout):
        calls.append((u, headers))
        return Response()

    monkeypatch.setattr(bfo.requests, "get", fake_get)
    monkeypatch.setattr(bfo.time, "sleep", slept.append)

    assert fetch(url, tmp_path, delay_s=5.0) == "<html>fresh</html>"
    assert fetch(url, tmp_path, delay_s=5.0) == "<html>fresh</html>"
    assert len(calls) == 1, "a cached URL must never be requested again"
    assert slept == [5.0]
    assert "fightline-ml" in calls[0][1]["User-Agent"]
