"""The Glicko-2 rating (the "glicko" family): Glickman's algorithm, and its point-in-time use.

The rating engine is checked against the worked example in Glickman's own paper
("Example of the Glicko-2 system", 2013, section 3). The rest pins how the feature builder
uses it: a fight never sees itself, a card is simultaneous, a tournament night is one
rating period, idle time inflates the RD up to its initial value, and draws and no
contests carry no rating information. ``tests/test_leakage.py`` also runs every glicko
column through the real-data mutation, same-card and tournament-night tests.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments.runner import family_kwargs
from ufc_predictor.features.glicko2 import Glicko2
from ufc_predictor.features.pit import GLICKO_COLS, build_features
from ufc_predictor.models.symmetric import mirror

# Glickman's display-to-internal scale, written out rather than imported so these tests
# check the module's constant instead of trusting it.
SCALE = 173.7178

# Glickman's recommended defaults: the values config/config.yaml fixes a priori.
PARAMS = {
    "tau": 0.5,
    "initial_rating": 1500.0,
    "initial_rd": 350.0,
    "initial_vol": 0.06,
    "period_days": 30,
}

D0 = dt.date(2019, 1, 1)


def _day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n)


def _engine(**overrides) -> Glicko2:
    return Glicko2(**(PARAMS | overrides))


def _fight(fid, date, f1, f2, winner="", *, decisive=True):
    """One bout in canonical orientation (fighter_1 < fighter_2 by id). ``winner``
    defaults to ``f1``; a non-decisive bout keeps whatever winner it is given."""
    a, b = sorted([f1, f2])
    return {
        "fight_id": fid,
        "event_id": f"e_{fid}",
        "event_name": f"Event {fid}",
        "date": date,
        "fighter_1": a,
        "fighter_2": b,
        "name_1": f"n_{a}",
        "name_2": f"n_{b}",
        "winner_id": (winner or f1) if decisive else (winner or None),
        "method": "DEC",
        "method_class": "DEC",
        "finish_round": 3,
        "finish_time_sec": 300,
        "weight_class": "Lightweight",
        "title_bout": False,
        "scheduled_rounds": 3,
        "is_decisive": decisive,
    }


def _build(rows, **kw):
    kw.setdefault("glicko_params", PARAMS)
    return build_features(
        pd.DataFrame(rows),
        pd.DataFrame(),
        pd.DataFrame(),
        rolling_windows=(3,),
        families=("glicko",),
        **kw,
    )


def _side(feats: pd.DataFrame, fight_id: str, fighter: str, stem: str) -> float:
    """``stem`` for ``fighter`` in ``fight_id``, read from whichever slot they occupy."""
    row = feats.loc[feats["fight_id"] == fight_id].iloc[0]
    slot = "1" if row["fighter_1"] == fighter else "2"
    return float(row[f"{stem}_{slot}"])


def _row(feats: pd.DataFrame, fight_id: str) -> pd.Series:
    return feats.loc[feats["fight_id"] == fight_id, GLICKO_COLS].iloc[0]


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


def test_matches_glickmans_worked_example():
    g = Glicko2(tau=0.5, initial_rating=1500, initial_rd=200, initial_vol=0.06, period_days=30)
    g.set_state("p", rating=1500, rd=200, vol=0.06)
    g.set_state("a", rating=1400, rd=30, vol=0.06)
    g.set_state("b", rating=1550, rd=100, vol=0.06)
    g.set_state("c", rating=1700, rd=300, vol=0.06)
    g.update_period("p", [("a", 1.0), ("b", 0.0), ("c", 0.0)])
    r, rd, vol = g.rating("p")
    assert r == pytest.approx(1464.06, abs=0.01)
    assert rd == pytest.approx(151.52, abs=0.01)
    assert vol == pytest.approx(0.05999, abs=1e-5)


def test_update_period_leaves_the_opponents_alone():
    g = _engine()
    g.set_state("p", rating=1500, rd=200, vol=0.06)
    g.set_state("a", rating=1400, rd=30, vol=0.06)
    g.update_period("p", [("a", 1.0)])
    assert g.rating("a") == pytest.approx((1400.0, 30.0, 0.06))


def test_a_new_player_starts_at_the_initial_state():
    g = _engine()
    assert g.rating("nobody") == pytest.approx((1500.0, 350.0, 0.06))
    mu, phi, sigma = g.pre_fight("nobody", D0)
    assert (mu, phi, sigma) == pytest.approx((0.0, 350.0 / SCALE, 0.06))


# ---------------------------------------------------------------------------
# Same-date simultaneity
# ---------------------------------------------------------------------------

PRE_DATE = {"a": (1600.0, 80.0, 0.06), "b": (1500.0, 120.0, 0.06), "c": (1450.0, 200.0, 0.059)}


def _seeded() -> Glicko2:
    g = _engine()
    for player, (rating, rd, vol) in PRE_DATE.items():
        g.set_state(player, rating=rating, rd=rd, vol=vol)
    return g


def test_two_fights_on_one_date_update_from_the_same_pre_date_state():
    """a beats b and b beats c on one date. Every participant is updated against the
    others' pre-date states -- b's result against c cannot reach a's update, nor a's
    result against b reach c's -- which a fresh engine per player reproduces."""
    g = _seeded()
    g.update_date(D0, [("a", "b", 1.0), ("b", "c", 1.0)])

    for player, results in {
        "a": [("b", 1.0)],
        "b": [("a", 0.0), ("c", 1.0)],
        "c": [("b", 0.0)],
    }.items():
        ref = _seeded()
        ref.update_period(player, results)
        assert g.rating(player) == pytest.approx(ref.rating(player), rel=1e-12), player

    # The test bites: applying the bouts one after another gives b a different rating.
    sequential = _seeded()
    sequential.update_period("a", [("b", 1.0)])
    sequential.update_period("b", [("a", 0.0), ("c", 1.0)])
    assert g.rating("b")[0] != pytest.approx(sequential.rating("b")[0], abs=1e-6)


def test_the_order_of_a_dates_results_does_not_matter():
    one, two = _seeded(), _seeded()
    one.update_date(D0, [("a", "b", 1.0), ("b", "c", 1.0)])
    two.update_date(D0, [("c", "b", 0.0), ("b", "a", 0.0)])
    for player in PRE_DATE:
        assert one.rating(player) == pytest.approx(two.rating(player), rel=1e-12), player


def test_a_date_cannot_be_split_or_revisited():
    g = _engine()
    g.update_date(D0, [("a", "b", 1.0)])
    with pytest.raises(ValueError, match="increas"):
        g.update_date(D0, [("c", "d", 1.0)])
    with pytest.raises(ValueError, match="increas"):
        g.update_date(_day(-1), [("c", "d", 1.0)])
    # Nor can a pre-fight state be read on or before a player's last rated date.
    with pytest.raises(ValueError, match="on or before"):
        g.pre_fight("a", D0)


@pytest.mark.parametrize(
    "results",
    [[("a", "a", 1.0)], [("a", "b", 1.5)], [("a", "b", -0.5)], [("a", "b", float("nan"))]],
)
def test_update_date_rejects_an_impossible_result(results):
    with pytest.raises(ValueError):
        _engine().update_date(D0, results)


def test_a_fights_rating_is_its_cards_pre_date_rating():
    """Flip a fight's winner. Nothing dated on or before that fight moves -- not the fight
    itself, and not its card-mate -- while the participants' later fights do."""
    rows = [
        _fight("m1", D0, "a", "b"),
        _fight("m2", D0, "c", "d"),
        _fight("m3", _day(20), "a", "c"),  # mutated below
        _fight("m4", _day(20), "b", "d"),  # m3's card-mate
        _fight("m5", _day(50), "a", "d"),
        _fight("m6", _day(90), "c", "b"),
    ]
    flipped = [r | {"winner_id": "c"} if r["fight_id"] == "m3" else r for r in rows]
    base, mutated = _build(rows), _build(flipped)

    def early(feats: pd.DataFrame) -> pd.DataFrame:
        return feats.loc[feats["date"] <= _day(20), ["fight_id", *GLICKO_COLS]]

    def label(feats: pd.DataFrame) -> float:
        return feats.loc[feats["fight_id"] == "m3", "label"].iloc[0]

    pd.testing.assert_frame_equal(early(base), early(mutated), check_exact=True)
    # The mutation landed, and it reaches both participants' next fights.
    assert label(base) != label(mutated)
    assert _side(base, "m5", "a", "glicko_mu") > _side(mutated, "m5", "a", "glicko_mu")
    assert _side(base, "m6", "c", "glicko_mu") < _side(mutated, "m6", "c", "glicko_mu")


# ---------------------------------------------------------------------------
# A one-night tournament is one rating period
# ---------------------------------------------------------------------------


def test_a_fighters_two_bouts_on_one_night_do_not_see_each_other():
    """a beats x before the night, then loses to b and beats c on the same night. Both
    bouts describe a from the pre-night state, and the night enters a's rating once, as
    one period holding both results against b's and c's pre-night states."""
    rows = [
        _fight("p1", D0, "a", "x"),
        _fight("p2", D0, "b", "y"),
        _fight("p3", D0, "c", "z", winner="z"),
        _fight("ab", _day(10), "a", "b", winner="b"),
        _fight("ac", _day(10), "a", "c"),
        _fight("at", _day(20), "a", "t"),
    ]
    feats = _build(rows)

    pre = _engine()
    pre.update_date(D0, [("a", "x", 1.0), ("b", "y", 1.0), ("c", "z", 0.0)])
    pre_night = {p: pre.rating(p) for p in ("a", "b", "c")}
    for fight_id in ("ab", "ac"):
        assert _side(feats, fight_id, "a", "glicko_mu") == pytest.approx(pre_night["a"][0])
        assert _side(feats, fight_id, "a", "glicko_rd") == pytest.approx(pre_night["a"][1])
    assert _side(feats, "ab", "b", "glicko_mu") == pytest.approx(pre_night["b"][0])
    assert _side(feats, "ac", "c", "glicko_mu") == pytest.approx(pre_night["c"][0])

    # After the night: one period, from the pre-night states (10 days later, no idle period).
    ref = _engine()
    for player, (rating, rd, vol) in pre_night.items():
        ref.set_state(player, rating=rating, rd=rd, vol=vol)
    ref.update_period("a", [("b", 0.0), ("c", 1.0)])
    after_rating, after_rd, _ = ref.rating("a")
    assert _side(feats, "at", "a", "glicko_mu") == pytest.approx(after_rating, rel=1e-12)
    assert _side(feats, "at", "a", "glicko_rd") == pytest.approx(after_rd, rel=1e-12)


# ---------------------------------------------------------------------------
# Idle time, draws and no contests
# ---------------------------------------------------------------------------


def test_rd_grows_with_each_idle_period_and_is_capped_at_the_initial_rd():
    """phi = min(sqrt(phi_u^2 + n sigma_u^2), phi_initial) with n = days // 30; the rating
    and the volatility do not change while a fighter is idle."""
    g = _engine()
    g.update_date(D0, [("a", "b", 1.0)])
    rating, rd, vol = g.rating("a")
    mu_u, phi_u = (rating - 1500.0) / SCALE, rd / SCALE

    for days, n in ((1, 0), (29, 0), (30, 1), (59, 1), (60, 2), (95, 3), (730, 24)):
        mu, phi, sigma = g.pre_fight("a", _day(days))
        assert phi == pytest.approx(math.sqrt(phi_u**2 + n * vol**2), rel=1e-12), days
        assert mu == pytest.approx(mu_u, rel=1e-12)
        assert sigma == vol
    # About 350 idle periods would pass the initial RD; there the cap holds.
    assert g.pre_fight("a", _day(40 * 365))[1] == pytest.approx(350.0 / SCALE, rel=1e-12)


def test_the_builder_reports_the_inflated_rd_after_a_layoff():
    feats = _build([_fight("w1", D0, "a", "b"), _fight("w2", _day(100), "a", "c")])
    g = _engine()
    g.update_date(D0, [("a", "b", 1.0)])
    _, rd, vol = g.rating("a")
    expected = SCALE * math.sqrt((rd / SCALE) ** 2 + 3 * vol**2)  # 100 // 30 = 3 periods
    assert _side(feats, "w2", "a", "glicko_rd") == pytest.approx(expected, rel=1e-12)
    assert _side(feats, "w2", "a", "glicko_rd") > rd


def test_draws_and_no_contests_neither_move_the_rating_nor_reset_the_idle_clock():
    """a's draw and no contest (one with a winner recorded but not decisive, as 24 real
    bouts are) leave a's state at the next fight exactly as if they never happened, and
    the RD keeps inflating from a's last decisive bout. Their opponents stay unrated."""
    decisive = [_fight("w1", D0, "a", "b"), _fight("w2", _day(100), "a", "e")]
    extra = [
        _fight("dr", _day(40), "a", "c", decisive=False),
        _fight("nc", _day(80), "a", "d", winner="a", decisive=False),
        _fight("c2", _day(120), "c", "f"),
    ]
    with_extra, without = _build(decisive + extra), _build(decisive)
    for stem in ("glicko_mu", "glicko_rd"):
        assert _side(with_extra, "w2", "a", stem) == _side(without, "w2", "a", stem), stem
    assert _side(with_extra, "c2", "c", "glicko_mu") == pytest.approx(1500.0)
    assert _side(with_extra, "c2", "c", "glicko_rd") == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# The seven columns
# ---------------------------------------------------------------------------


def test_glicko_expected_1_matches_the_hand_computed_value():
    """a beats b on D0 (both new: 1500 / 350 / 0.06) and meets the debutant c ten days on.

    After D0, a is at 1662.311 / 290.319 (a newcomer beating a newcomer, tau 0.5). Then
      mu_1 - mu_2 = 162.311 / 173.7178                    = 0.934337
      phi_1 = 290.319 / 173.7178 = 1.671210,  phi_2 = 350 / 173.7178 = 2.014762
      sqrt(phi_1^2 + phi_2^2)                             = 2.617672
      g = 1 / sqrt(1 + 3 * 2.617672^2 / pi^2)             = 0.569542
      glicko_expected_1 = 1 / (1 + exp(-0.569542 * 0.934337)) = 0.629983
    """
    feats = _build([_fight("ab", D0, "a", "b"), _fight("ac", _day(10), "a", "c")])
    got = _row(feats, "ac")
    assert got["glicko_mu_1"] == pytest.approx(1662.311, abs=1e-3)
    assert got["glicko_rd_1"] == pytest.approx(290.319, abs=1e-3)
    assert got["glicko_mu_2"] == pytest.approx(1500.0)
    assert got["glicko_rd_2"] == pytest.approx(350.0)
    assert got["glicko_mu_diff"] == pytest.approx(162.311, abs=1e-3)
    assert got["glicko_rd_sum"] == pytest.approx(640.319, abs=1e-3)
    assert got["glicko_expected_1"] == pytest.approx(0.629983, abs=1e-6)

    # Two newcomers: an even bout, two initial ratings, two initial RDs.
    first = _row(feats, "ab")
    assert first.to_dict() == pytest.approx(
        {
            "glicko_mu_1": 1500.0,
            "glicko_mu_2": 1500.0,
            "glicko_rd_1": 350.0,
            "glicko_rd_2": 350.0,
            "glicko_mu_diff": 0.0,
            "glicko_expected_1": 0.5,
            "glicko_rd_sum": 700.0,
        }
    )


def test_swapping_the_slots_mirrors_the_glicko_columns():
    """Renaming a to "m" puts c in slot 1. The pairs swap, the diff changes sign,
    glicko_expected_1 becomes one minus itself and glicko_rd_sum is unchanged."""

    def toy(a: str):
        return _build([_fight("ab", D0, a, "b"), _fight("ac", _day(10), a, "c")])

    ac = _row(toy("a"), "ac").to_frame().T.astype(float).reset_index(drop=True)
    ca = _row(toy("m"), "ac").to_frame().T.astype(float).reset_index(drop=True)
    pd.testing.assert_frame_equal(mirror(ac), ca, check_exact=False, rtol=1e-12, atol=1e-12)


def test_the_columns_are_the_seven_of_ruling_g3():
    assert GLICKO_COLS == [
        "glicko_mu_1",
        "glicko_mu_2",
        "glicko_rd_1",
        "glicko_rd_2",
        "glicko_mu_diff",
        "glicko_expected_1",
        "glicko_rd_sum",
    ]


# ---------------------------------------------------------------------------
# Parameters: from config, never a silent default
# ---------------------------------------------------------------------------


def test_requesting_glicko_without_parameters_raises():
    with pytest.raises(ValueError, match="glicko_params"):
        _build([_fight("ab", D0, "a", "b")], glicko_params=None)


@pytest.mark.parametrize(
    "params",
    [
        {k: v for k, v in PARAMS.items() if k != "tau"},  # missing key
        PARAMS | {"k_factor": 32.0},  # unknown key
        PARAMS | {"tau": 0.0},
        PARAMS | {"initial_rd": -1.0},
        PARAMS | {"initial_vol": 0.0},
        PARAMS | {"period_days": 0},
        PARAMS | {"period_days": 30.5},
    ],
)
def test_bad_parameters_raise(params):
    with pytest.raises(ValueError):
        _build([_fight("ab", D0, "a", "b")], glicko_params=params)


def test_parameters_are_ignored_when_glicko_is_not_requested():
    rows = [_fight("ab", D0, "a", "b"), _fight("ac", _day(10), "a", "c")]
    plain = build_features(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), rolling_windows=(3,))
    with_params = build_features(
        pd.DataFrame(rows),
        pd.DataFrame(),
        pd.DataFrame(),
        rolling_windows=(3,),
        glicko_params=PARAMS,
    )
    pd.testing.assert_frame_equal(plain, with_params, check_exact=True)
    assert not set(GLICKO_COLS) & set(plain.columns)


def test_the_config_holds_glickmans_defaults():
    assert get_config().glicko2 == PARAMS


def test_family_kwargs_pass_the_configured_glicko_parameters():
    cfg = get_config()
    kwargs = family_kwargs(cfg)
    assert kwargs["glicko_params"] == cfg.glicko2
    assert kwargs["recency_half_life_fights"] == 3.0
