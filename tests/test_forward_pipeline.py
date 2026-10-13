"""The forward pipeline's pure parts: names, history after the snapshot, the ledger, scheduling."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from ufc_predictor.forward import cards, ledger, schedule
from ufc_predictor.forward.espn import Bout, Card, Result

UTC = dt.UTC
D0 = dt.date(2026, 10, 3)


def history() -> pd.DataFrame:
    """A three-bout snapshot: four known fighters, last event 2026-10-03."""
    rows = [
        (
            "f1",
            "e1",
            "UFC 330",
            dt.date(2026, 8, 1),
            "aaa",
            "bbb",
            "zhang weili",
            "joaquin buckley",
            "aaa",
        ),
        (
            "f2",
            "e2",
            "UFC 331",
            dt.date(2026, 9, 1),
            "ccc",
            "ddd",
            "mike malott",
            "kyle nelson",
            "ddd",
        ),
        ("f3", "e3", "UFC 332", D0, "aaa", "ccc", "zhang weili", "mike malott", "ccc"),
    ]
    out = []
    for fid, eid, ename, date, f1, f2, n1, n2, w in rows:
        out.append(
            {
                "fight_id": fid,
                "event_id": eid,
                "event_name": ename,
                "date": date,
                "pair_key": "|".join(sorted([n1, n2])),
                "fighter_1": f1,
                "fighter_2": f2,
                "name_1": n1,
                "name_2": n2,
                "winner_id": w,
                "method": "DEC",
                "method_class": "DEC",
                "finish_round": 3,
                "finish_time_sec": 300,
                "weight_class": "Welterweight Bout",
                "title_bout": False,
                "scheduled_rounds": 3,
                "is_decisive": True,
            }
        )
    df = pd.DataFrame(out)
    for col in ("finish_round", "finish_time_sec", "scheduled_rounds"):
        df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_names_match_exactly_by_alias_by_reordering_or_are_debutants():
    index = cards.fighter_index(history())
    aliases = {"joaquim buckley": "bbb"}
    got = cards.match_names(
        ["Mike Malott", "Joaquim Buckley", "Weili Zhang", "Totally Newperson"], index, aliases
    )
    assert got == {
        "Mike Malott": "ccc",
        "Joaquim Buckley": "bbb",  # alias
        "Weili Zhang": "aaa",  # same tokens, reordered
        "Totally Newperson": None,  # debutant
    }


def test_a_near_miss_name_raises_and_lists_every_offender():
    index = cards.fighter_index(history())
    with pytest.raises(cards.NameMatchError) as err:
        cards.match_names(["Mike Mallott", "Kyle Nelsen", "Totally Newperson"], index, {})
    assert "Mike Mallott" in str(err.value) and "Kyle Nelsen" in str(err.value)


def test_a_confirmed_debutant_near_a_known_name_is_a_debutant():
    index = cards.fighter_index(history())
    got = cards.match_names(["Mike Mallott"], index, {"mike mallott": cards.DEBUT})
    assert got == {"Mike Mallott": None}


def test_one_name_for_two_fighter_ids_raises():
    h = history()
    h.loc[1, "fighter_1"] = "zzz"  # "mike malott" now names both ccc and zzz
    with pytest.raises(ValueError, match="several fighter ids"):
        cards.fighter_index(h)


# ---------------------------------------------------------------------------
# Card rows and post-snapshot history
# ---------------------------------------------------------------------------


def card(*bouts: tuple[str, str], name="UFC Fight Night: Buckley vs. Malott", rounds=3) -> Card:
    start = dt.datetime(2026, 10, 17, 21, tzinfo=UTC)
    return Card(
        "600060773",
        name,
        dt.date(2026, 10, 17),
        start,
        tuple(
            Bout(f"c{i}", a, b, start, "Welterweight", rounds, "STATUS_SCHEDULED")
            for i, (a, b) in enumerate(bouts)
        ),
    )


def test_card_rows_are_unlabelled_canonical_and_give_debutants_synthetic_ids():
    index = cards.fighter_index(history())
    rows = cards.card_rows(
        card(("Mike Malott", "Joaquin Buckley"), ("Totally Newperson", "Kyle Nelson")), index, {}
    )
    assert list(rows.columns) == cards.FIGHT_COLUMNS
    first = rows.iloc[0]
    assert (first["fighter_1"], first["fighter_2"]) == ("bbb", "ccc")  # sorted ids
    assert not first["is_decisive"] and pd.isna(first["winner_id"])
    second = rows.iloc[1]
    assert "espn:totally newperson" in (second["fighter_1"], second["fighter_2"])
    assert rows["date"].eq(dt.date(2026, 10, 17)).all()


def test_title_bout_rule_marks_five_rounders_on_numbered_events_only():
    index = cards.fighter_index(history())
    numbered = cards.card_rows(
        card(("Mike Malott", "Joaquin Buckley"), name="UFC 333: X vs. Y", rounds=5), index, {}
    )
    fight_night = cards.card_rows(card(("Mike Malott", "Joaquin Buckley"), rounds=5), index, {})
    assert numbered["title_bout"].iloc[0] and not fight_night["title_bout"].iloc[0]


def test_saved_results_extend_history_and_keep_draws_as_non_decisive(tmp_path):
    c = card(
        ("Mike Malott", "Joaquin Buckley"),
        ("Totally Newperson", "Kyle Nelson"),
        ("Zhang Weili", "Kyle Nelson"),
    )
    c = Card(c.event_id, c.name, dt.date(2026, 10, 10), c.start_utc, c.bouts)
    results = [
        Result("c0", ("Mike Malott", "Joaquin Buckley"), True, "Joaquin Buckley", "KO/TKO", 2),
        Result("c1", ("Totally Newperson", "Kyle Nelson"), True, None, None, 3),
        Result("c2", ("Zhang Weili", "Kyle Nelson"), False, None, None, None),  # cancelled
    ]
    cards.save_results(tmp_path / "600060773.json", c, results)
    combined, index = cards.forward_history(history(), tmp_path, {}, before=dt.date(2026, 10, 17))
    new = combined.loc[combined["fight_id"].str.startswith("espn-")]
    assert len(new) == 2  # the cancelled bout is not history
    ko = new.loc[new["fight_id"] == "espn-c0"].iloc[0]
    assert ko["winner_id"] == "bbb" and ko["is_decisive"] and ko["method_class"] == "KO/TKO"
    draw = new.loc[new["fight_id"] == "espn-c1"].iloc[0]
    assert not draw["is_decisive"] and pd.isna(draw["winner_id"])
    assert index["totally newperson"] == "espn:totally newperson"
    assert combined["date"].is_monotonic_increasing


def test_results_on_or_after_the_card_date_are_not_history(tmp_path):
    c = card(("Mike Malott", "Joaquin Buckley"))
    cards.save_results(
        tmp_path / "x.json",
        c,
        [Result("c0", ("Mike Malott", "Joaquin Buckley"), True, "Mike Malott", "DEC", 3)],
    )
    combined, _ = cards.forward_history(history(), tmp_path, {}, before=dt.date(2026, 10, 17))
    assert not combined["fight_id"].str.startswith("espn-").any()


def test_results_overlapping_the_snapshot_raise(tmp_path):
    c = card(("Mike Malott", "Joaquin Buckley"))
    c = Card(c.event_id, c.name, D0, c.start_utc, c.bouts)
    cards.save_results(
        tmp_path / "x.json",
        c,
        [Result("c0", ("Mike Malott", "Joaquin Buckley"), True, "Mike Malott", "DEC", 3)],
    )
    with pytest.raises(ValueError, match="overlap the snapshot"):
        cards.forward_history(history(), tmp_path, {}, before=dt.date(2026, 10, 17))


# ---------------------------------------------------------------------------
# Ledger and the commit-before-start check
# ---------------------------------------------------------------------------


def test_ledger_records_each_action_once(tmp_path):
    path = tmp_path / "ledger.json"
    ledger.append(path, {"event_id": "e", "action": "predict"})
    with pytest.raises(ValueError, match="already recorded"):
        ledger.append(path, {"event_id": "e", "action": "predict"})
    with pytest.raises(ValueError, match="unknown"):
        ledger.append(path, {"event_id": "e", "action": "bet"})
    assert ledger.has(ledger.load(path), "e", "predict")


def _git(repo, *args, env=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    )


def _commit_at(repo: Path, when: dt.datetime, message: str):
    import os

    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": when.isoformat(),
        "GIT_COMMITTER_DATE": when.isoformat(),
    }
    _git(repo, "add", ".", env=env)
    _git(repo, "commit", "-q", "-m", message, env=env)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    return r


def test_committed_before_reads_the_commit_that_added_the_predictions(repo):
    start = dt.datetime(2026, 10, 24, 17, tzinfo=UTC)
    rel = "reports/forward/600061267/predictions.csv"
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_text("p\n0.6\n")
    _commit_at(repo, start - dt.timedelta(hours=30), "predict")
    assert ledger.committed_before(repo, rel, start - dt.timedelta(hours=24))
    assert ledger.unchanged_since_added(repo, rel)

    (repo / rel).write_text("p\n0.7\n")  # an edit after the fact
    _commit_at(repo, start - dt.timedelta(hours=1), "tweak")
    assert not ledger.unchanged_since_added(repo, rel)


def test_predictions_added_too_late_fail_the_check(repo):
    start = dt.datetime(2026, 10, 24, 17, tzinfo=UTC)
    rel = "reports/forward/600061267/predictions.csv"
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_text("p\n0.6\n")
    _commit_at(repo, start - dt.timedelta(hours=10), "predict")
    assert not ledger.committed_before(repo, rel, start - dt.timedelta(hours=24))
    assert not ledger.committed_before(repo, "reports/forward/none/predictions.csv", start)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

KW = {
    "min_lead_hours": 24,
    "predict_window_hours": 72,
    "close_window_hours": 3,
    "settle_after_hours": 36,
}
START = dt.datetime(2026, 10, 24, 17, tzinfo=UTC)


def sched_card(status="STATUS_SCHEDULED"):
    return Card("ev", "UFC 333", START.date(), START, (Bout("c", "A", "B", START, "", 3, status),))


@pytest.mark.parametrize(
    ("hours_before", "entries", "status", "expected"),
    [
        (30, [], "STATUS_SCHEDULED", [("predict", "ev")]),
        (80, [], "STATUS_SCHEDULED", []),  # not yet inside the window
        (23, [], "STATUS_SCHEDULED", [("missed", "ev")]),  # never predicted late
        (-2, [], "STATUS_SCHEDULED", [("missed", "ev")]),
        (23, [{"event_id": "ev", "action": "missed"}], "STATUS_SCHEDULED", []),
        (30, [{"event_id": "ev", "action": "predict"}], "STATUS_SCHEDULED", []),
        (1.5, [{"event_id": "ev", "action": "predict"}], "STATUS_SCHEDULED", [("close", "ev")]),
        (-3, [{"event_id": "ev", "action": "predict"}], "STATUS_IN_PROGRESS", []),
        (-6, [{"event_id": "ev", "action": "predict"}], "STATUS_FINAL", [("settle", "ev")]),
        (-40, [{"event_id": "ev", "action": "predict"}], "STATUS_SCHEDULED", [("settle", "ev")]),
        (
            -6,
            [{"event_id": "ev", "action": "predict"}, {"event_id": "ev", "action": "settle"}],
            "STATUS_FINAL",
            [],
        ),
    ],
)
def test_due_actions(hours_before, entries, status, expected):
    now = START - dt.timedelta(hours=hours_before)
    assert schedule.due_actions([sched_card(status)], entries, now, **KW) == expected
