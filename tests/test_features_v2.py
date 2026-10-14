"""Phase 8 feature families: hand-computed values, the registry, and point-in-time checks.

``tests/test_leakage.py`` runs every new column through the real-data mutation, same-card
and fighters-table tests. This file proves the families are *right*: each expected value
below is worked out by hand on a toy history small enough to check on paper, and the toy
mutation tests pin the property the definitions depend on -- an opponent's baseline is
frozen at the bout's date, so nothing the opponent does later can move it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.features import adjusted
from ufc_predictor.features.pit import (
    ADJUSTED_COLS,
    ALL_FEATURE_COLS,
    DECLINE_COLS,
    FEATURE_COLS,
    FEATURE_FAMILIES,
    RATE_METRICS,
    RECENCY_COLS,
    ROUND_SECONDS,
    STYLE_COLS,
    build_features,
    feature_columns,
    fighter_fight_history,
)
from ufc_predictor.models.symmetric import mirror
from ufc_predictor.storage import db

META_COLS = ["fight_id", "date", "fighter_1", "fighter_2", "label", "is_decisive"]
NEW_FAMILIES = ("adjusted", "recency", "style", "decline")
NEW_COLS = [c for family in NEW_FAMILIES for c in FEATURE_FAMILIES[family]]
# Toy builds use a single 3-fight window, so their base columns are this subset.
TOY_BASE = feature_columns((3,))

D0 = dt.date(2019, 1, 1)
D1 = dt.date(2019, 3, 1)
D2 = dt.date(2019, 6, 1)
D3 = dt.date(2019, 9, 1)
D4 = dt.date(2020, 1, 1)
D5 = dt.date(2020, 3, 1)

# 2^(-k/3): the recency weight of the k-th most recent bout at the configured half-life.
W1, W2 = 2.0 ** (-1 / 3), 2.0 ** (-2 / 3)


def _fight(fid, date, f1, f2, winner="", *, rnd=1, secs=ROUND_SECONDS):
    """One five-minute bout in canonical orientation (fighter_1 < fighter_2 by id)."""
    a, b = sorted([f1, f2])
    return {
        "fight_id": fid,
        "event_id": f"e_{fid}",
        "event_name": f"Event {fid}",
        "date": date,
        "pair_key": f"n_{a}|n_{b}",
        "fighter_1": a,
        "fighter_2": b,
        "name_1": f"n_{a}",
        "name_2": f"n_{b}",
        "winner_id": winner or f1,
        "method": "DEC",
        "method_class": "DEC",
        "finish_round": rnd,
        "finish_time_sec": secs,
        "weight_class": "Lightweight",
        "title_bout": False,
        "scheduled_rounds": 3,
        "is_decisive": True,
    }


def _stat(fid, fighter, **kw):
    """One fighter's single-round totals for a bout. Unset counts are zero."""
    row = {"fight_id": fid, "fighter_norm": f"n_{fighter}", "round": 1}
    for col in (
        "kd",
        "sub_att",
        "ctrl_sec",
        "sig_str_landed",
        "sig_str_att",
        "td_landed",
        "td_att",
    ):
        row[col] = 0
    row.update(kw)
    return row


def _fighters(ids):
    return pd.DataFrame(
        [
            {
                "fighter_id": i,
                "name": f"n_{i}",
                "name_norm": f"n_{i}",
                "height_cm": 180.0,
                "reach_cm": 183.0,
                "weight_lbs": 155.0,
                "stance": "Orthodox",
                "dob": dt.date(1990, 1, 1),
            }
            for i in ids
        ]
    )


def _ids(fights: pd.DataFrame) -> list[str]:
    return sorted(set(fights["fighter_1"]) | set(fights["fighter_2"]))


def _build(fights, stats, **kw):
    kw.setdefault("families", NEW_FAMILIES)
    return build_features(fights, stats, _fighters(_ids(fights)), rolling_windows=(3,), **kw)


def _side(feats: pd.DataFrame, fight_id: str, fighter: str, col: str) -> float:
    """``col`` for ``fighter`` in ``fight_id``, read from whichever slot they occupy."""
    row = feats.loc[feats["fight_id"] == fight_id].iloc[0]
    slot = "1" if row["fighter_1"] == fighter else "2"
    return row[f"{col}_{slot}"]


# ---------------------------------------------------------------------------
# The brief's toy history
#
# o1 and o2 each have one bout before facing f, absorbing 3.0 and 5.0 significant strikes
# a minute. o3 debuts against f, so has no baseline. f lands 4, 6 and 2 a minute in those
# three bouts, then meets o4 in f4.
# ---------------------------------------------------------------------------


def _toy():
    fights = pd.DataFrame(
        [
            _fight("p1", D0, "o1", "x1"),
            _fight("p2", D0, "o2", "x2"),
            _fight("f1", D1, "f", "o1"),
            _fight("f2", D2, "f", "o2"),
            _fight("f3", D3, "f", "o3"),
            _fight("f4", D4, "f", "o4"),
        ]
    )
    stats = pd.DataFrame(
        [
            # o1 absorbs 15 of 30 in 5:00 (3.0/min, sig. str. defence 0.5) and is taken
            # down once (3.0 per 15 min). x1 also lands a knockdown, two submission
            # attempts and a minute of control.
            _stat("p1", "o1", sig_str_landed=5, sig_str_att=10),
            _stat(
                "p1",
                "x1",
                sig_str_landed=15,
                sig_str_att=30,
                td_landed=1,
                td_att=2,
                kd=1,
                sub_att=2,
                ctrl_sec=60,
            ),
            # o2 absorbs 25 of 100 in 5:00 (5.0/min, defence 0.75) and no takedowns.
            _stat("p2", "o2", sig_str_landed=5, sig_str_att=10),
            _stat("p2", "x2", sig_str_landed=25, sig_str_att=100),
            # f: 20 of 25 (4.0/min, acc 0.8) with 2 takedowns (6.0 per 15 min).
            _stat("f1", "f", sig_str_landed=20, sig_str_att=25, td_landed=2, td_att=4),
            _stat("f1", "o1", sig_str_landed=10, sig_str_att=20),
            # f: 30 of 60 (6.0/min, acc 0.5), no takedowns.
            _stat("f2", "f", sig_str_landed=30, sig_str_att=60),
            _stat("f2", "o2", sig_str_landed=10, sig_str_att=20),
            # f: 10 of 20 (2.0/min) against the debutant o3.
            _stat("f3", "f", sig_str_landed=10, sig_str_att=20),
            _stat("f3", "o3", sig_str_landed=10, sig_str_att=20),
        ]
    )
    return fights, stats


def test_adjusted_strike_rate_matches_the_hand_computed_mean_residual():
    """mean(4 - 3, 6 - 5) = 1.0; the bout against debutant o3 has no baseline and is
    excluded rather than counted as a residual against zero."""
    feats = _build(*_toy())
    assert _side(feats, "f4", "f", "adj_sig_str_landed_pm_career") == pytest.approx(1.0)
    # The same quantity at earlier dates, built up one bout at a time.
    assert _side(feats, "f2", "f", "adj_sig_str_landed_pm_career") == pytest.approx(1.0)
    assert _side(feats, "f3", "f", "adj_sig_str_landed_pm_career") == pytest.approx(1.0)


def test_ewm_strike_rate_matches_the_hand_computed_weights():
    """Half-life 3, most recent bout weight 1: (2*1 + 6*2^-1/3 + 4*2^-2/3) / (1 + ...)."""
    feats = _build(*_toy(), recency_half_life_fights=3.0)
    expected = (2 * 1 + 6 * W1 + 4 * W2) / (1 + W1 + W2)
    assert _side(feats, "f4", "f", "ewm_sig_str_landed_pm") == pytest.approx(expected)
    assert _side(feats, "f3", "f", "ewm_sig_str_landed_pm") == pytest.approx(
        (6 + 4 * W1) / (1 + W1)
    )
    assert _side(feats, "f2", "f", "ewm_sig_str_landed_pm") == pytest.approx(4.0)


def test_the_half_life_keyword_sets_the_decay():
    """Half-life 1 halves each step back: (2 + 6/2 + 4/4) / (1 + 1/2 + 1/4)."""
    feats = _build(*_toy(), recency_half_life_fights=1.0)
    expected = (2 + 6 * 0.5 + 4 * 0.25) / 1.75
    assert _side(feats, "f4", "f", "ewm_sig_str_landed_pm") == pytest.approx(expected)


def test_a_ratio_metric_is_adjusted_by_what_the_opponents_opponents_achieved():
    """sig_str_acc is mirrored by 1 - the opponent's defence: the accuracy the opponent
    had allowed before. (0.8 - 0.5) and (0.5 - 0.25), mean 0.275."""
    feats = _build(*_toy())
    assert _side(feats, "f4", "f", "adj_sig_str_acc_career") == pytest.approx(0.275)


def test_takedown_rate_is_adjusted_by_the_opponents_takedowns_absorbed():
    """td_landed_p15 is mirrored by a new aggregate: takedowns the opponent had conceded
    per 15 minutes. o1 had conceded 3.0, o2 none: (6.0 - 3.0) and (0.0 - 0.0), mean 1.5."""
    feats = _build(*_toy())
    assert _side(feats, "f4", "f", "adj_td_landed_p15_career") == pytest.approx(1.5)


def test_absorbed_aggregates_are_career_ratios_of_what_opponents_did():
    """o1's only bout: in 5:00 x1 landed 1 takedown, 2 submission attempts, 1 knockdown
    and 60 s of control. Per 15 minutes that is 3, 6 and 3; control is 12 s a minute."""
    fights, stats = _toy()
    history = fighter_fight_history(fights, stats)
    o1 = history.loc[(history["fighter_id"] == "o1") & (history["fight_id"] == "p1")]
    out = adjusted.absorbed_rates(o1.to_dict("records"))
    assert out == pytest.approx(
        {
            "td_absorbed_p15": 3.0,
            "sub_att_absorbed_p15": 6.0,
            "kd_absorbed_p15": 3.0,
            "ctrl_absorbed_pm": 12.0,
        }
    )


def test_absorbed_aggregates_are_nan_without_stats():
    assert all(np.isnan(v) for v in adjusted.absorbed_rates([]).values())


def test_diff_columns_are_slot_one_minus_slot_two():
    feats = _build(*_toy())
    for col in NEW_COLS:
        if not col.endswith("_diff"):
            continue
        stem = col[: -len("_diff")]
        lhs = (feats[f"{stem}_1"] - feats[f"{stem}_2"]).to_numpy(dtype=float)
        np.testing.assert_array_equal(lhs, feats[col].to_numpy(dtype=float), err_msg=col)


# ---------------------------------------------------------------------------
# Missing history: NaN, never zero
# ---------------------------------------------------------------------------


def test_a_debutant_has_nan_adjusted_and_recency_values_not_zeros():
    feats = _build(*_toy())
    for col in [*ADJUSTED_COLS, *RECENCY_COLS]:
        if col.endswith("_1"):
            stem = col[: -len("_1")]
            assert pd.isna(_side(feats, "f1", "f", stem)), f"{stem} set for a debutant"
            assert pd.isna(_side(feats, "f4", "o4", stem)), f"{stem} set for a debutant"


def test_bouts_without_stats_are_skipped_not_counted_as_zero():
    """f's middle bout has no round statistics. It neither enters the adjusted mean nor
    consumes a recency step: the remaining weights are 1 and 2^(-1/3)."""
    fights = pd.DataFrame(
        [
            _fight("p1", D0, "o1", "x1"),
            _fight("f1", D1, "f", "o1"),
            _fight("f2", D2, "f", "o2"),
            _fight("f3", D3, "f", "o3"),
            _fight("f4", D4, "f", "o4"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat("p1", "o1", sig_str_landed=5, sig_str_att=10),
            _stat("p1", "x1", sig_str_landed=15, sig_str_att=30),
            _stat("f1", "f", sig_str_landed=20, sig_str_att=25),
            _stat("f1", "o1", sig_str_landed=10, sig_str_att=20),
            _stat("f3", "f", sig_str_landed=10, sig_str_att=20),
            _stat("f3", "o3", sig_str_landed=10, sig_str_att=20),
        ]
    )
    feats = _build(fights, stats)
    assert _side(feats, "f4", "f", "ewm_sig_str_landed_pm") == pytest.approx(
        (2 + 4 * W1) / (1 + W1)
    )
    assert _side(feats, "f4", "f", "adj_sig_str_landed_pm_career") == pytest.approx(1.0)
    # Straight after the stat-less bout, f's recency mean is still f1's 4.0 alone. Had the
    # bout counted as zero it would be (0 + 4 * 2^(-1/3)) / (1 + 2^(-1/3)).
    assert _side(feats, "f3", "f", "ewm_sig_str_landed_pm") == pytest.approx(4.0)


# 360 fighter-bouts from 1994-99 have round statistics but no control time. The per-bout
# and absorbed paths leave such a bout out of the control metrics entirely; the base
# career rate keeps its run-3 arithmetic, so the default build stays byte-identical.


def _ctrl_toy():
    """o's first bout has round statistics but no control time; its second has both."""
    fights = pd.DataFrame(
        [
            _fight("c1", D0, "o", "x"),
            _fight("c2", D1, "o", "y"),
            _fight("c3", D2, "f", "o"),
            _fight("c4", D3, "f", "z"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat("c1", "o", sig_str_landed=10, sig_str_att=20, ctrl_sec=np.nan),
            _stat("c1", "x", sig_str_landed=10, sig_str_att=20, kd=1, ctrl_sec=np.nan),
            # o controls 30 s and concedes 60 s in 5:00: 6 and 12 s a minute.
            _stat("c2", "o", sig_str_landed=10, sig_str_att=20, ctrl_sec=30),
            _stat("c2", "y", sig_str_landed=10, sig_str_att=20, ctrl_sec=60),
            # f controls 120 s of 5:00 against o: 24 s a minute.
            _stat("c3", "f", sig_str_landed=10, sig_str_att=20, ctrl_sec=120),
            _stat("c3", "o", sig_str_landed=10, sig_str_att=20),
        ]
    )
    return fights, stats


def test_a_bout_without_control_time_is_left_out_of_the_absorbed_control_rate():
    """o conceded 60 s of control in the one bout that recorded it: 12 s a minute over
    that bout's 5:00, not 60 s over both bouts' 10:00. Knockdowns were recorded in both,
    so that aggregate keeps both: 1 in 10:00 is 1.5 per 15 minutes."""
    fights, stats = _ctrl_toy()
    history = fighter_fight_history(fights, stats)
    prior = history.loc[(history["fighter_id"] == "o") & (history["date"] < D2)]
    out = adjusted.absorbed_rates(prior.to_dict("records"))
    assert out["ctrl_absorbed_pm"] == pytest.approx(12.0)
    assert out["kd_absorbed_p15"] == pytest.approx(1.5)


def test_a_bout_without_control_time_has_no_per_bout_control_rate():
    feats = _build(*_ctrl_toy())
    # o's recency mean is c2's 6.0 alone. Read as zero, c1 would make it 6 / (1 + 2^(-1/3)).
    assert _side(feats, "c3", "o", "ewm_ctrl_pm") == pytest.approx(6.0)
    # f's residual subtracts the 12.0 o had conceded (24 - 12), not 6.0 (24 - 6).
    assert _side(feats, "c4", "f", "adj_ctrl_pm_career") == pytest.approx(12.0)
    # The base career rate is untouched: 30 s over both bouts' 10:00.
    assert _side(feats, "c3", "o", "ctrl_pm_career") == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Style-matchup interactions
#
# a, b and c each have one 5:00 bout on D0. On D1, a fights b (and is knocked out) and
# then c: a one-night tournament. On D2 a meets the debutant t. Pre-night career rates:
#
#            landed/min  sig.def  td/15  td.def  kd/15  kd absorbed/15
#     a         4.0       0.75     6.0    0.8     3.0       6.0
#     b         6.0       0.6      3.0    0.5     6.0       9.0
#     c         3.0       0.5      0.0    0.5     3.0       0.0
# ---------------------------------------------------------------------------

KO = {"method": "KO/TKO", "method_class": "KO/TKO"}
SUB = {"method": "Submission", "method_class": "SUB"}


def _style_toy(a: str = "a"):
    """``a`` names the fighter in a's role; renaming it changes which slot a occupies."""
    fights = pd.DataFrame(
        [
            _fight("s1", D0, a, "x1"),
            _fight("s2", D0, "b", "y1"),
            _fight("s3", D0, "c", "z1"),
            _fight("ab", D1, a, "b", winner="b") | KO,
            _fight("ac", D1, a, "c"),
            _fight("at", D2, a, "t"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat("s1", a, sig_str_landed=20, sig_str_att=40, td_landed=2, td_att=4, kd=1),
            _stat("s1", "x1", sig_str_landed=10, sig_str_att=40, td_landed=1, td_att=5, kd=2),
            _stat("s2", "b", sig_str_landed=30, sig_str_att=50, td_landed=1, td_att=2, kd=2),
            _stat("s2", "y1", sig_str_landed=20, sig_str_att=50, td_landed=2, td_att=4, kd=3),
            _stat("s3", "c", sig_str_landed=15, sig_str_att=30, kd=1),
            _stat("s3", "z1", sig_str_landed=10, sig_str_att=20, td_landed=1, td_att=2),
            # A lopsided first bout of the night. Had it leaked into a's second bout, a's
            # rates there would be 2.2 landed, 0.4 defence, 0.5 td defence, 7.5 absorbed.
            _stat("ab", a, sig_str_landed=2, sig_str_att=10),
            _stat("ab", "b", sig_str_landed=50, sig_str_att=60, td_landed=3, td_att=3, kd=3),
            _stat("ac", a, sig_str_landed=10, sig_str_att=20),
            _stat("ac", "c", sig_str_landed=5, sig_str_att=10),
        ]
    )
    return fights, stats


def _row(feats: pd.DataFrame, fight_id: str, cols: list[str]) -> pd.Series:
    return feats.loc[feats["fight_id"] == fight_id, cols].iloc[0]


def test_style_interactions_match_the_hand_computed_products():
    """a (slot 1) against b (slot 2), from both fighters' pre-fight career rates."""
    got = _row(_build(*_style_toy()), "ab", STYLE_COLS)
    expected = {
        "td_threat_1": 6.0 * (1 - 0.5),  # a's takedowns against b's takedown defence
        "td_threat_2": 3.0 * (1 - 0.8),
        "td_threat_diff": 3.0 - 0.6,
        "strike_exchange_1": 4.0 * (1 - 0.6),  # a's strikes against b's strike defence
        "strike_exchange_2": 6.0 * (1 - 0.75),
        "strike_exchange_diff": 1.6 - 1.5,
        "grappler_vs_striker": (6.0 - 3.0) * (6.0 - 4.0),
        "ko_power_vs_chin_1": 3.0 * 9.0,  # a's knockdown rate against b's absorbed rate
        "ko_power_vs_chin_2": 6.0 * 6.0,
        "ko_power_vs_chin_diff": 27.0 - 36.0,
    }
    assert got.to_dict() == pytest.approx(expected)


@pytest.mark.leakage
def test_a_second_bout_on_one_night_uses_the_pre_night_style_rates():
    """a's bout with c comes after a's knockout loss to b on the same night, but must be
    described from a's pre-night rates. With the leak, strike_exchange_2 would be
    3.0 * (1 - 0.4) = 1.8 and ko_power_vs_chin_2 would be 3.0 * 7.5 = 22.5."""
    got = _row(_build(*_style_toy()), "ac", STYLE_COLS)
    expected = {
        "td_threat_1": 6.0 * (1 - 0.5),
        "td_threat_2": 0.0 * (1 - 0.8),
        "td_threat_diff": 3.0,
        "strike_exchange_1": 4.0 * (1 - 0.5),
        "strike_exchange_2": 3.0 * (1 - 0.75),
        "strike_exchange_diff": 2.0 - 0.75,
        "grappler_vs_striker": (6.0 - 0.0) * (3.0 - 4.0),
        "ko_power_vs_chin_1": 3.0 * 0.0,
        "ko_power_vs_chin_2": 3.0 * 6.0,
        "ko_power_vs_chin_diff": -18.0,
    }
    assert got.to_dict() == pytest.approx(expected)


def test_style_interactions_are_nan_when_either_side_has_no_history():
    """Any NaN input gives NaN: both sides' interactions need the other's rates."""
    feats = _build(*_style_toy())
    for fight_id in ("s1", "at"):  # two debutants; then a against the debutant t
        assert _row(feats, fight_id, STYLE_COLS).isna().all(), fight_id


def test_swapping_the_slots_mirrors_the_style_columns():
    """Renaming a to "m" puts b in slot 1 of their bout. The paired columns swap, the
    diffs change sign, and grappler_vs_striker -- a product of two differences that both
    change sign -- is unchanged."""
    ab = _row(_build(*_style_toy()), "ab", STYLE_COLS).to_frame().T
    ba = _row(_build(*_style_toy(a="m")), "ab", STYLE_COLS).to_frame().T
    swapped = mirror(ab.astype(float)).reset_index(drop=True)
    pd.testing.assert_frame_equal(swapped, ba.astype(float).reset_index(drop=True))
    assert ba["grappler_vs_striker"].iloc[0] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Decline signals
#
# f loses by KO on D0 (absorbing 1 knockdown), wins a decision on D1, is submitted on D2
# (absorbing 2 knockdowns), has a no contest without statistics on D3, and fights again
# on D4. Every bout is 5:00.
# ---------------------------------------------------------------------------


def _decline_toy():
    # The no contest is deliberately given a KO/TKO method: a bout without a winner is
    # experience, never a loss, whatever the method column says.
    no_contest = {"winner_id": None, "is_decisive": False} | KO
    fights = pd.DataFrame(
        [
            _fight("g1", D0, "f", "o1", winner="o1") | KO,
            _fight("g2", D1, "f", "o2"),
            _fight("g3", D2, "f", "o3", winner="o3") | SUB,
            _fight("g4", D3, "f", "o4") | no_contest,
            _fight("g5", D4, "f", "o5"),
            _fight("g6", D5, "o1", "o3"),  # two fighters who have never lost
            _fight("g7", D5, "o2", "o4"),  # a decision loss; a no contest without stats
        ]
    )
    stats = pd.DataFrame(
        [
            _stat("g1", "f", sig_str_landed=10, sig_str_att=20),
            _stat("g1", "o1", sig_str_landed=20, sig_str_att=40, kd=1),
            _stat("g2", "f", sig_str_landed=25, sig_str_att=50),
            _stat("g2", "o2", sig_str_landed=5, sig_str_att=20),
            _stat("g3", "f", sig_str_landed=15, sig_str_att=30),
            _stat("g3", "o3", sig_str_landed=10, sig_str_att=20, kd=2),
        ]
    )
    return fights, stats


DECLINE_STEMS = [c[:-2] for c in DECLINE_COLS if c.endswith("_1")]


def _decline(feats: pd.DataFrame, fight_id: str, fighter: str) -> dict[str, float]:
    return {stem: _side(feats, fight_id, fighter, stem) for stem in DECLINE_STEMS}


def test_decline_signals_match_the_hand_computed_values():
    """At g5 f's last three bouts are g2, g3 and the no contest g4.

    * 365 days since the debut on D0.
    * One finish loss in the window: g3's submission (g1's knockout has dropped out).
    * Knockdowns absorbed: 0 in g2 + 2 in g3; g4 has no statistics, so adds nothing.
    * Absorbed per minute: (5 + 10) / 10 over the window against (20 + 5 + 10) / 15 over
      the career; landed: (25 + 15) / 10 against (10 + 25 + 15) / 15.
    * 214 days since g3 (2019-06-01 to 2020-01-01).
    """
    got = _decline(_build(*_decline_toy()), "g5", "f")
    assert got == pytest.approx(
        {
            "years_since_ufc_debut": 365 / 365.25,
            "finished_losses_last3": 1.0,
            "kd_absorbed_last3": 2.0,
            "absorbed_trend": 1.5 - 35 / 15,
            "landed_trend": 4.0 - 50 / 15,
            "days_since_last_finish_loss": 214.0,
        }
    )


def test_decline_signals_build_up_one_bout_at_a_time():
    feats = _build(*_decline_toy())
    # g2: one bout behind f, a knockout loss 59 days earlier. One bout is both the window
    # and the career, so the trends are exactly zero.
    assert _decline(feats, "g2", "f") == pytest.approx(
        {
            "years_since_ufc_debut": 59 / 365.25,
            "finished_losses_last3": 1.0,
            "kd_absorbed_last3": 1.0,
            "absorbed_trend": 0.0,
            "landed_trend": 0.0,
            "days_since_last_finish_loss": 59.0,
        }
    )
    # g4: g1, g2, g3 -- two finish losses, three knockdowns, the last loss 92 days back.
    assert _decline(feats, "g4", "f") == pytest.approx(
        {
            "years_since_ufc_debut": 243 / 365.25,
            "finished_losses_last3": 2.0,
            "kd_absorbed_last3": 3.0,
            "absorbed_trend": 0.0,
            "landed_trend": 0.0,
            "days_since_last_finish_loss": 92.0,
        }
    )


def test_a_debutant_has_zero_tenure_and_no_other_decline_value():
    """years_since_ufc_debut = 0.0 is a true value for a debutant, flagged by is_debut;
    every other decline value has no bout to come from, so is NaN, never 0."""
    feats = _build(*_decline_toy())
    for fight_id, fighter in (("g1", "f"), ("g1", "o1"), ("g5", "o5")):
        got = _decline(feats, fight_id, fighter)
        assert got.pop("years_since_ufc_debut") == 0.0
        assert _side(feats, fight_id, fighter, "is_debut")
        assert all(np.isnan(v) for v in got.values()), (fight_id, fighter, got)


def test_decline_counts_are_zero_not_nan_for_a_fighter_who_was_never_finished():
    """o1 and o3 have only won; o2's one loss was a decision. Their finish-loss counts are
    a true 0, and the days since a finish loss that never happened are NaN."""
    feats = _build(*_decline_toy())
    for fight_id, fighter in (("g6", "o1"), ("g6", "o3"), ("g7", "o2")):
        assert _side(feats, fight_id, fighter, "finished_losses_last3") == 0.0
        assert np.isnan(_side(feats, fight_id, fighter, "days_since_last_finish_loss"))
    assert _side(feats, "g6", "o3", "kd_absorbed_last3") == 0.0
    assert _side(feats, "g6", "o1", "kd_absorbed_last3") == 0.0


def test_a_no_contest_counts_as_a_bout_but_not_as_a_loss():
    """o4's only bout is the no contest, which recorded no statistics: a real 0 finish
    losses, but no knockdown count and no strike trend to report."""
    got = _decline(_build(*_decline_toy()), "g7", "o4")
    assert got["finished_losses_last3"] == 0.0
    assert np.isnan(got["days_since_last_finish_loss"])
    assert np.isnan(got["kd_absorbed_last3"])
    assert np.isnan(got["absorbed_trend"]) and np.isnan(got["landed_trend"])
    assert got["years_since_ufc_debut"] == pytest.approx(182 / 365.25)


@pytest.mark.leakage
def test_a_fighters_two_bouts_on_one_night_share_the_pre_night_decline_values():
    """a is knocked out by b and then fights c on the same night. The knockout must not
    reach a's second bout; it appears only from a's next date onwards."""
    feats = _build(*_style_toy())
    pre_night = {
        "years_since_ufc_debut": 59 / 365.25,
        "finished_losses_last3": 0.0,
        "kd_absorbed_last3": 2.0,
        "absorbed_trend": 0.0,
        "landed_trend": 0.0,
    }
    for fight_id in ("ab", "ac"):
        got = _decline(feats, fight_id, "a")
        assert np.isnan(got.pop("days_since_last_finish_loss")), fight_id
        assert got == pytest.approx(pre_night), fight_id
    # On D2 the night is history: one finish loss 92 days back, 2 + 3 + 0 knockdowns.
    after = _decline(feats, "at", "a")
    assert after["finished_losses_last3"] == 1.0
    assert after["days_since_last_finish_loss"] == 92.0
    assert after["kd_absorbed_last3"] == 5.0


# ---------------------------------------------------------------------------
# Point in time
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_an_opponents_later_fight_cannot_move_an_earlier_adjustment():
    """Mutate o1's bout *after* it met f. Every feature dated on or before that bout is
    byte-identical, and so is f's adjustment at f4 -- o1's baseline for f1 was frozen on
    f1's date. o1's own later features do move, so the mutation is genuinely in play."""
    fights, stats = _toy()
    fights = pd.concat(
        [fights, pd.DataFrame([_fight("q1", D2, "o1", "q"), _fight("r1", D5, "o1", "r")])],
        ignore_index=True,
    )
    base_stats = pd.concat(
        [
            stats,
            pd.DataFrame(
                [
                    _stat("q1", "o1", sig_str_landed=5, sig_str_att=10),
                    _stat("q1", "q", sig_str_landed=15, sig_str_att=30),
                ]
            ),
        ],
        ignore_index=True,
    )
    mutated_stats = base_stats.copy()
    m = (mutated_stats["fight_id"] == "q1") & (mutated_stats["fighter_norm"] == "n_q")
    mutated_stats.loc[m, ["sig_str_landed", "sig_str_att", "td_landed", "td_att", "kd"]] = [
        140,
        150,
        9,
        9,
        4,
    ]

    a = _build(fights, base_stats)
    b = _build(fights, mutated_stats)
    cols = ["fight_id", *TOY_BASE, *NEW_COLS]

    early_a = a.loc[a["date"] <= D2, cols].sort_values("fight_id").reset_index(drop=True)
    early_b = b.loc[b["date"] <= D2, cols].sort_values("fight_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a, early_b, check_exact=True)

    for col in [c[:-2] for c in NEW_COLS if c.endswith("_1")]:
        va, vb = _side(a, "f4", "f", col), _side(b, "f4", "f", col)
        assert (pd.isna(va) and pd.isna(vb)) or va == vb, f"{col} moved for f at f4"

    moved = _side(a, "r1", "o1", "ewm_sig_str_absorbed_pm") != _side(
        b, "r1", "o1", "ewm_sig_str_absorbed_pm"
    )
    assert moved, "the mutation never reached o1's own later features; the test proves nothing"


@pytest.mark.leakage
def test_a_fighters_two_bouts_on_one_night_use_the_pre_night_baseline():
    """a had absorbed 3.0 a minute before the night, then fights b and c on the same date.
    b's and c's adjustments must subtract 3.0 -- not a's average after the other bout.
    With the leak, a's baseline for c would be (15 + 50) / 10 = 6.5, giving -2.5."""
    fights = pd.DataFrame(
        [
            _fight("p1", D0, "a", "p"),
            _fight("n1", D1, "a", "b", winner="a"),
            _fight("n2", D1, "a", "c", winner="a"),
            _fight("t1", D2, "b", "y"),
            _fight("t2", D2, "c", "z"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat("p1", "a", sig_str_landed=5, sig_str_att=10),
            _stat("p1", "p", sig_str_landed=15, sig_str_att=30),
            _stat("n1", "a", sig_str_landed=5, sig_str_att=10),
            _stat("n1", "b", sig_str_landed=50, sig_str_att=60),
            _stat("n2", "a", sig_str_landed=5, sig_str_att=10),
            _stat("n2", "c", sig_str_landed=20, sig_str_att=40),
        ]
    )
    feats = _build(fights, stats)
    assert _side(feats, "t1", "b", "adj_sig_str_landed_pm_career") == pytest.approx(10.0 - 3.0)
    assert _side(feats, "t2", "c", "adj_sig_str_landed_pm_career") == pytest.approx(4.0 - 3.0)
    # And a enters both bouts that night with the same pre-night state.
    for col in [c[:-2] for c in ADJUSTED_COLS + RECENCY_COLS if c.endswith("_1")]:
        first, second = _side(feats, "n1", "a", col), _side(feats, "n2", "a", col)
        assert (pd.isna(first) and pd.isna(second)) or first == second, col
    assert _side(feats, "n1", "a", "ewm_sig_str_absorbed_pm") == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Registry and the families keyword
# ---------------------------------------------------------------------------


def test_registry_holds_base_then_the_new_families():
    assert list(FEATURE_FAMILIES)[:5] == ["base", "adjusted", "recency", "style", "decline"]
    assert FEATURE_FAMILIES["base"] == FEATURE_COLS
    assert len(FEATURE_COLS) == 117
    assert FEATURE_FAMILIES["adjusted"] == ADJUSTED_COLS
    assert FEATURE_FAMILIES["recency"] == RECENCY_COLS
    assert FEATURE_FAMILIES["style"] == STYLE_COLS
    assert FEATURE_FAMILIES["decline"] == DECLINE_COLS
    assert [c for cols in FEATURE_FAMILIES.values() for c in cols] == ALL_FEATURE_COLS
    assert len(set(ALL_FEATURE_COLS)) == len(ALL_FEATURE_COLS), "duplicate feature name"


def test_adjusted_and_recency_columns_are_named_per_metric_and_slot():
    slots = ("1", "2", "diff")
    assert [f"adj_{m}_career_{s}" for m in RATE_METRICS for s in slots] == ADJUSTED_COLS
    assert [f"ewm_{m}_{s}" for m in RATE_METRICS for s in slots] == RECENCY_COLS
    assert len(ADJUSTED_COLS) + len(RECENCY_COLS) == 60


def test_style_and_decline_columns_are_named_per_slot():
    """Ruling S1: three paired interactions plus the one orientation-invariant column.
    Ruling S2: six per-fighter decline metrics, each _1, _2 and _diff."""
    slots = ("1", "2", "diff")
    assert [
        *(f"td_threat_{s}" for s in slots),
        *(f"strike_exchange_{s}" for s in slots),
        "grappler_vs_striker",
        *(f"ko_power_vs_chin_{s}" for s in slots),
    ] == STYLE_COLS
    metrics = (
        "years_since_ufc_debut",
        "finished_losses_last3",
        "kd_absorbed_last3",
        "absorbed_trend",
        "landed_trend",
        "days_since_last_finish_loss",
    )
    assert [f"{m}_{s}" for m in metrics for s in slots] == DECLINE_COLS
    assert len(STYLE_COLS) + len(DECLINE_COLS) == 28


def test_every_new_column_has_a_declared_orientation():
    """The symmetric wrapper must be able to mirror every column a family adds."""
    frame = pd.DataFrame(np.zeros((1, len(ALL_FEATURE_COLS))), columns=ALL_FEATURE_COLS)
    mirror(frame)  # raises KeyError on a column with no declared orientation


def test_the_mirror_and_source_tables_cover_every_rate_metric():
    assert tuple(adjusted.MIRROR) == RATE_METRICS
    assert tuple(adjusted.SOURCES) == RATE_METRICS


def test_requested_families_append_after_the_base_columns():
    fights, stats = _toy()
    assert list(_build(fights, stats, families=("base",)).columns) == META_COLS + TOY_BASE
    assert list(_build(fights, stats, families=("recency",)).columns) == (
        META_COLS + TOY_BASE + RECENCY_COLS
    )
    # Registry order, whatever order the caller lists the families in.
    assert list(_build(fights, stats, families=("recency", "base", "adjusted")).columns) == (
        META_COLS + TOY_BASE + ADJUSTED_COLS + RECENCY_COLS
    )
    assert list(_build(fights, stats, families=("decline", "style")).columns) == (
        META_COLS + TOY_BASE + STYLE_COLS + DECLINE_COLS
    )


def test_an_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown feature famil"):
        _build(*_toy(), families=("base", "telepathy"))


def test_a_non_positive_half_life_raises():
    with pytest.raises(ValueError, match="half"):
        _build(*_toy(), recency_half_life_fights=0.0)


def test_the_half_life_is_in_config():
    assert float(get_config().features["recency_half_life_fights"]) == 3.0


# ---------------------------------------------------------------------------
# The real tables: the published pipeline is untouched
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def tables(cfg):
    try:
        return {name: db.read_table(cfg.paths.processed_dir, name) for name in db.TABLES}
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data`")


@pytest.fixture(scope="module")
def default_build(tables, cfg):
    return build_features(
        tables["fights"],
        tables["fight_stats"],
        tables["fighters"],
        rolling_windows=tuple(cfg.features["rolling_windows"]),
        elo_params=cfg.elo,
    )


@pytest.fixture(scope="module")
def full_build(tables, cfg):
    return build_features(
        tables["fights"],
        tables["fight_stats"],
        tables["fighters"],
        rolling_windows=tuple(cfg.features["rolling_windows"]),
        elo_params=cfg.elo,
        families=tuple(FEATURE_FAMILIES),
        recency_half_life_fights=float(cfg.features["recency_half_life_fights"]),
    )


def _digest(df: pd.DataFrame) -> bytes:
    return pd.util.hash_pandas_object(df, index=False).values.tobytes()


def test_the_default_build_is_byte_identical_to_the_base_slice_of_the_full_build(
    default_build, full_build
):
    """Adding families must not perturb a single base value, row or column position."""
    assert list(default_build.columns) == META_COLS + FEATURE_COLS
    assert list(full_build.columns) == META_COLS + ALL_FEATURE_COLS
    restricted = full_build.loc[:, META_COLS + FEATURE_COLS]
    pd.testing.assert_frame_equal(default_build, restricted, check_exact=True)
    assert _digest(default_build) == _digest(restricted)


def test_the_default_build_reproduces_the_published_feature_table(default_build, cfg):
    """``features.parquet`` (written by the unchanged pipeline) still equals the default
    build, so the closed run-3 inputs, the demo export and ``predict`` cannot move."""
    try:
        published = db.read_table(cfg.paths.processed_dir, "features")
    except FileNotFoundError:
        pytest.skip("features absent; run `make features`")
    published = published.drop(columns=["is_priced"])
    assert list(published.columns) == list(default_build.columns)
    assert _digest(published) == _digest(default_build)


def test_no_new_column_is_entirely_null(full_build):
    empty = [c for c in NEW_COLS if full_build[c].isna().all()]
    assert not empty, f"entirely null: {empty}"


def test_style_and_trend_columns_are_their_definitions_over_the_base_columns(full_build):
    """Ruling S1/S2 restated over the real table: each interaction and trend equals its
    formula applied to the same row's leak-tested base career and w3 columns, exactly.
    (ko_power_vs_chin needs knockdowns absorbed, which no base column holds.)"""
    f = full_build
    expected = {
        "td_threat_1": f["td_landed_p15_career_1"] * (1 - f["td_def_career_2"]),
        "td_threat_2": f["td_landed_p15_career_2"] * (1 - f["td_def_career_1"]),
        "strike_exchange_1": f["sig_str_landed_pm_career_1"] * (1 - f["sig_str_def_career_2"]),
        "strike_exchange_2": f["sig_str_landed_pm_career_2"] * (1 - f["sig_str_def_career_1"]),
        "grappler_vs_striker": (f["td_landed_p15_career_1"] - f["td_landed_p15_career_2"])
        * (f["sig_str_landed_pm_career_2"] - f["sig_str_landed_pm_career_1"]),
    }
    for s in ("1", "2"):
        for trend, metric in (
            ("absorbed_trend", "sig_str_absorbed_pm"),
            ("landed_trend", "sig_str_landed_pm"),
        ):
            expected[f"{trend}_{s}"] = f[f"{metric}_w3_{s}"] - f[f"{metric}_career_{s}"]
    for col, values in expected.items():
        np.testing.assert_array_equal(
            f[col].to_numpy(dtype=float), values.to_numpy(dtype=float), err_msg=col
        )


def test_new_families_are_nan_for_a_debutant_apart_from_tenure(full_build):
    """A debutant's tenure is a true 0.0; every other decline value, and every style
    interaction (each needs both sides' rates), is NaN -- never 0."""
    debuts = full_build.loc[full_build["is_debut_1"]]
    assert len(debuts) > 100
    assert (debuts["years_since_ufc_debut_1"] == 0.0).all()
    veterans = full_build.loc[~full_build["is_debut_1"], "years_since_ufc_debut_1"]
    assert (veterans > 0).all(), "a fighter with a prior bout has a positive tenure"
    blank = [c for c in DECLINE_COLS if c.endswith("_1") and not c.startswith("years_since")]
    for col in [*blank, *STYLE_COLS]:
        assert debuts[col].isna().all(), f"{col} is populated for debutants; must be NaN"


def test_decline_values_are_in_plausible_ranges(full_build):
    """Bounds that hold by construction. A finish loss is strictly before the fight, so
    no sooner than the last bout; tenure covers at least the time since the last bout."""
    for s in ("1", "2"):
        losses = full_build[f"finished_losses_last3_{s}"].dropna()
        assert losses.isin([0.0, 1.0, 2.0, 3.0]).all()
        assert (full_build[f"kd_absorbed_last3_{s}"].dropna() >= 0).all()
        since_loss = full_build[f"days_since_last_finish_loss_{s}"]
        has_loss = since_loss.notna()
        assert has_loss.any()
        assert (since_loss[has_loss] >= full_build.loc[has_loss, f"days_since_last_{s}"]).all()
        assert (since_loss[has_loss] > 0).all()
        tenure_days = full_build[f"years_since_ufc_debut_{s}"] * 365.25
        last = full_build[f"days_since_last_{s}"]
        assert (tenure_days[last.notna()] >= last[last.notna()] - 1e-6).all()


def test_new_columns_are_in_plausible_ranges(full_build):
    """Bounds that hold by construction, so a breach is a unit or ratio error: ratio
    metrics live in [0, 1], so do their recency means, and a residual of one ratio
    against another lies in [-1, 1].

    ``eps`` absorbs float rounding only: a weighted mean of bouts that were all 1.0
    can come out as 1.0000000000000002."""
    eps = 1e-12
    for metric in ("sig_str_acc", "sig_str_def", "td_acc", "td_def"):
        for s in ("1", "2"):
            ewm = full_build[f"ewm_{metric}_{s}"].dropna()
            adj = full_build[f"adj_{metric}_career_{s}"].dropna()
            assert ewm.between(-eps, 1 + eps).all(), f"ewm_{metric}_{s}"
            assert adj.between(-1 - eps, 1 + eps).all(), f"adj_{metric}_career_{s}"
    # The same physical ceiling tests/test_features.py applies to the career rate.
    rate = full_build["ewm_sig_str_landed_pm_1"].dropna()
    assert rate.between(0, 120).all()
