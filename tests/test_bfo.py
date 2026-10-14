"""BestFightOdds: parsing recorded pages, the response cache, and the snapshot crawl.

Every test runs offline against fixtures in ``tests/fixtures/bfo/``, recorded from the
live site on 2026-10-13 and trimmed. ``requests.get`` is replaced with a function that
raises, so a test that reached the network would fail rather than pass by luck.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments.protocol import EvalWindowClosed
from ufc_predictor.ingest import bfo, bfo_join
from ufc_predictor.ingest.bfo import (
    LINE_COLUMNS,
    fetch,
    parse_event_page,
    parse_events_sitemap,
    parse_fighter_page,
)
from ufc_predictor.ingest.bfo_join import join_bfo

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


# ---------------------------------------------------------------------------
# Joining lines to fights (ingest/bfo_join.py)
#
# Constructed frames. Every bout below is priced the same way unless a test says
# otherwise, so the expected probabilities can be computed by hand:
#
#   open:  -150 -> 150/250 = 3/5;   +130 -> 100/230 = 10/23
#          de-vigged first side = (3/5) / (3/5 + 10/23) = 69/119
#   close: first side  -200..-150 -> (2/3 + 3/5) / 2   = 19/30
#          second side +120..+170 -> (5/11 + 10/27) / 2 = 245/594
#          de-vigged first side = (19/30) / (19/30 + 245/594) = 1881/3106
# ---------------------------------------------------------------------------

DAY = dt.date(2012, 3, 3)
PRICES = {
    "open_fighter": -150,
    "open_opponent": 130,
    "close_lo_fighter": -200,
    "close_hi_fighter": -150,
    "close_lo_opponent": 120,
    "close_hi_opponent": 170,
}


def line(fighter: str, opponent: str, date: dt.date = DAY, **prices) -> dict:
    return {
        "fighter": fighter,
        "fighter_url": "/fighters/" + fighter.lower().replace(" ", "-"),
        "opponent": opponent,
        "opponent_url": "/fighters/" + opponent.lower().replace(" ", "-"),
        "event": "UFC on FX 2",
        "event_id": 500,
        "event_url": "/events/ufc-on-fx-2-500",
        "date": date,
        **{**PRICES, **prices},
    }


def lines_frame(*rows: dict) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=LINE_COLUMNS)
    for col in PRICES:
        frame[col] = frame[col].astype("Int64")
    return frame


def fights_frame(*rows: tuple[str, dt.date, str, str]) -> pd.DataFrame:
    """``(fight_id, date, name_1, name_2)``; names already normalized, as in the table."""
    return pd.DataFrame(
        [
            {
                "fight_id": fid,
                "date": date,
                "fighter_1": "id-" + n1.replace(" ", "-"),
                "fighter_2": "id-" + n2.replace(" ", "-"),
                "name_1": n1,
                "name_2": n2,
            }
            for fid, date, n1, n2 in rows
        ]
    )


def test_join_orients_prices_to_fighter_1_whatever_order_bfo_lists_the_pair():
    fights = fights_frame(
        ("f-same", DAY, "jon jones", "mauricio rua"),
        ("f-swapped", DAY, "anderson silva", "chael sonnen"),
    )
    lines = lines_frame(
        line("Jon Jones", "Mauricio Rua"),
        line("Chael Sonnen", "Anderson Silva"),  # BFO's first side is fighter_2
    )
    joined, audit = join_bfo(lines, fights, {})
    assert audit["joined"] == 2
    same = joined.set_index("fight_id").loc["f-same"]
    swapped = joined.set_index("fight_id").loc["f-swapped"]

    assert same["fighter_1"] == "id-jon-jones"
    assert (same["open_odds_1"], same["open_odds_2"]) == (-150, 130)
    assert same["open_p1"] == pytest.approx(69 / 119)
    assert same["close_p1"] == pytest.approx(1881 / 3106)

    assert swapped["fighter_1"] == "id-anderson-silva"
    assert (swapped["open_odds_1"], swapped["open_odds_2"]) == (130, -150)
    assert swapped["open_p1"] == pytest.approx(50 / 119)
    assert swapped["close_p1"] == pytest.approx(1225 / 3106)
    assert swapped["close_mid_p1_raw"] == pytest.approx(245 / 594)
    assert swapped["close_mid_p2_raw"] == pytest.approx(19 / 30)


def test_devig_and_closing_midpoint_arithmetic():
    fights = fights_frame(("f1", DAY, "jon jones", "mauricio rua"))
    joined, _ = join_bfo(lines_frame(line("Jon Jones", "Mauricio Rua")), fights, {})
    row = joined.iloc[0]
    assert row["open_p1"] == pytest.approx(69 / 119, abs=1e-12)
    # The closing midpoint is taken in implied-probability space, per side ...
    assert row["close_mid_p1_raw"] == pytest.approx(19 / 30, abs=1e-12)
    assert row["close_mid_p2_raw"] == pytest.approx(245 / 594, abs=1e-12)
    # ... then the two sides are proportionally de-vigged.
    assert row["close_p1"] == pytest.approx(1881 / 3106, abs=1e-12)
    # Not the midpoint of the American prices (-175 -> 175/275 for the first side).
    assert row["close_mid_p1_raw"] != pytest.approx(175 / 275)
    # The midpoint re-expressed as an American price: decimal 30/19 -> -100 / (11/19).
    assert row["close_mid_odds_1"] == pytest.approx(-1900 / 11)
    assert row["close_mid_odds_2"] == pytest.approx(100 * (594 / 245 - 1))


def test_a_pair_present_only_on_another_date_is_not_joined():
    fights = fights_frame(("f1", DAY, "jon jones", "mauricio rua"))
    lines = lines_frame(line("Jon Jones", "Mauricio Rua", date=DAY - dt.timedelta(days=1)))
    joined, audit = join_bfo(lines, fights, {})
    assert joined.empty
    assert audit["dropped_date_mismatch"] == 1
    assert audit["joined"] == 0 and audit["unmatched_no_name_match"] == 0


def test_a_pair_unknown_on_every_date_is_a_name_mismatch_and_counts_partition_the_input():
    fights = fights_frame(
        ("f1", DAY, "jon jones", "mauricio rua"),
        ("f2", DAY, "anderson silva", "chael sonnen"),
    )
    lines = lines_frame(
        line("Jon Jones", "Mauricio Rua"),
        line("Anderson Silva", "Chael Sonnen", date=DAY + dt.timedelta(days=1)),
        line("Jon Jonez", "Mauricio Rua"),  # a spelling the alias table does not know
        line("Someone Else", "Another Fighter"),
    )
    joined, audit = join_bfo(lines, fights, {})
    assert audit["lines_in"] == 4
    assert audit["joined"] == len(joined) == 1
    assert audit["dropped_date_mismatch"] == 1
    assert audit["unmatched_no_name_match"] == 2
    assert audit["dropped_incomplete"] == 0


def test_the_alias_table_resolves_a_name_variant_exactly():
    fights = fights_frame(("f1", DAY, "aleksandra albu", "kailin curran"))
    lines = lines_frame(line("Kailin Curran", "Alexandra Albu"))
    joined, audit = join_bfo(lines, fights, {"alexandra albu": "aleksandra albu"})
    assert audit["joined"] == 1 and audit["joined_alias"] == 1
    assert joined.iloc[0]["open_p1"] == pytest.approx(50 / 119)  # Albu is fighter_1


def test_a_pair_listed_twice_on_one_date_raises():
    fights = fights_frame(("f1", DAY, "jon jones", "mauricio rua"))
    lines = lines_frame(line("Jon Jones", "Mauricio Rua"), line("Mauricio Rua", "Jon Jones"))
    with pytest.raises(ValueError, match="twice"):
        join_bfo(lines, fights, {})


def test_two_bfo_rows_matching_one_fight_raises():
    fights = fights_frame(("f1", DAY, "rafael dos anjos", "donald cerrone"))
    lines = lines_frame(
        line("Rafael dos Anjos", "Donald Cerrone"),
        line("R dos Anjos", "Donald Cerrone"),  # a different spelling, aliased to the same man
    )
    with pytest.raises(ValueError, match="same fight"):
        join_bfo(lines, fights, {"r dos anjos": "rafael dos anjos"})


@pytest.mark.parametrize("missing", list(PRICES))
def test_a_bout_missing_any_price_is_dropped_and_counted(missing):
    fights = fights_frame(("f1", DAY, "jon jones", "mauricio rua"))
    lines = lines_frame(line("Jon Jones", "Mauricio Rua", **{missing: pd.NA}))
    joined, audit = join_bfo(lines, fights, {})
    assert joined.empty
    assert audit["dropped_incomplete"] == 1
    assert audit["joined"] == 0


def test_a_fight_in_the_evaluation_window_raises():
    first_eval_day = get_config().windows.eval_start
    fights = fights_frame(("f1", first_eval_day, "jon jones", "mauricio rua"))
    lines = lines_frame(line("Jon Jones", "Mauricio Rua", date=first_eval_day))
    with pytest.raises(EvalWindowClosed):
        join_bfo(lines, fights, {})


def test_run_writes_the_table_the_audit_and_the_unjoined_rows(tmp_path):
    """``run`` end to end on constructed inputs, never touching the real outputs."""
    cfg = get_config()
    processed, reports, snapshot = tmp_path / "processed", tmp_path / "reports", tmp_path / "bfo"
    for d in (processed, reports, snapshot):
        d.mkdir()
    later = dt.date(2017, 5, 6)
    fights = fights_frame(
        ("f-joined", DAY, "jon jones", "mauricio rua"),
        ("f-unjoined", DAY, "anderson silva", "chael sonnen"),
        ("f-draw", DAY, "frankie edgar", "gray maynard"),
        ("f-unpriced", DAY, "dan hardy", "carlos condit"),
        ("f-eval", later, "max holloway", "jose aldo"),
    )
    fights["is_decisive"] = [True, True, False, True, True]
    fights["winner_id"] = [fights.loc[i, "fighter_1"] for i in range(4)] + ["id-max-holloway"]
    fights.loc[2, "winner_id"] = None
    fights.to_parquet(processed / "fights.parquet", index=False)
    pd.DataFrame({"fight_id": ["f-joined", "f-unjoined", "f-draw", "f-eval"]}).to_parquet(
        processed / "odds.parquet", index=False
    )
    lines_frame(
        line("Jon Jones", "Mauricio Rua"),
        line("Anderson Silva", "Chael Sonnen", date=DAY - dt.timedelta(days=1)),
        line("Nobody Known", "Also Unknown"),
    ).to_parquet(snapshot / "lines.parquet", index=False)
    test_cfg = dataclasses.replace(
        cfg,
        paths=dataclasses.replace(cfg.paths, processed_dir=processed, reports_dir=reports),
        bfo={**cfg.bfo, "snapshot_dir": str(snapshot)},
    )

    audit = bfo_join.run(test_cfg)

    written = json.loads((reports / "bfo_join.json").read_text(encoding="utf-8"))
    assert written == audit
    assert (audit["lines_in"], audit["joined"]) == (3, 1)
    assert (audit["dropped_date_mismatch"], audit["unmatched_no_name_match"]) == (1, 1)
    # Priced, decisive, dev-window fights: f-joined and f-unjoined (f-draw has no label,
    # f-unpriced is not in the odds table, f-eval is outside the window).
    assert (audit["dev_priced_fights"], audit["dev_priced_joined"]) == (2, 1)
    assert audit["dev_priced_join_rate"] == pytest.approx(0.5)
    table = pd.read_parquet(processed / "bfo_lines.parquet")
    assert list(table["fight_id"]) == ["f-joined"]
    unjoined = pd.read_csv(reports / "bfo_unjoined.csv")
    assert sorted(unjoined["reason"]) == ["date_mismatch", "no_name_match"]


#: The audit keys the README renderer reads (controller ruling J4).
J4_INT_KEYS = (
    "lines_in",
    "joined",
    "dropped_incomplete",
    "dropped_date_mismatch",
    "unmatched_no_name_match",
    "dev_priced_fights",
    "dev_priced_joined",
)


def test_real_snapshot_joins_at_least_2000_bouts_exactly():
    cfg = get_config()
    try:
        joined, _, audit = bfo_join.build(cfg)
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data`")
    for key in J4_INT_KEYS:
        assert isinstance(audit[key], int), key
    assert isinstance(audit["dev_priced_join_rate"], float)
    assert 0.0 <= audit["dev_priced_join_rate"] <= 1.0
    assert audit["joined"] == len(joined) >= 2000
    assert joined["fight_id"].is_unique
    assert max(joined["date"]) < cfg.windows.eval_start
    assert joined[["open_p1", "close_p1"]].notna().all().all()
    assert ((joined["open_p1"] > 0) & (joined["open_p1"] < 1)).all()
