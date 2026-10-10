"""Feature correctness tests: do the features mean what their names claim?

``tests/test_leakage.py`` proves the features cannot see the future. These tests prove
they are also *right* -- a feature can be perfectly point-in-time correct and still be
arithmetically wrong, which would quietly cap the model's ceiling instead of inflating
it. Small hand-built fixtures are used where the expected value can be computed by hand.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.features.elo import build_elo, expected_score
from ufc_predictor.features.pit import (
    FEATURE_COLS,
    ROUND_SECONDS,
    build_features,
    fighter_fight_history,
)
from ufc_predictor.storage import db

D1 = dt.date(2020, 1, 1)
D2 = dt.date(2020, 6, 1)
D3 = dt.date(2021, 1, 1)


def _fight(fid, date, f1, f2, winner, *, method="KO/TKO", rnd=1, secs=60, decisive=True):
    """One fight row in canonical orientation (fighter_1 < fighter_2 by id)."""
    a, b = sorted([f1, f2])
    return {
        "fight_id": fid,
        "event_id": f"e_{fid}",
        "event_name": f"Event {fid}",
        "date": date,
        "pair_key": "|".join(sorted([f"n_{a}", f"n_{b}"])),
        "fighter_1": a,
        "fighter_2": b,
        "name_1": f"n_{a}",
        "name_2": f"n_{b}",
        "winner_id": winner,
        "method": method,
        "method_class": method,
        "finish_round": rnd,
        "finish_time_sec": secs,
        "weight_class": "Lightweight",
        "title_bout": False,
        "scheduled_rounds": 3,
        "is_decisive": decisive,
    }


def _stat_row(fid, fighter_norm, rnd, **kw):
    row = {
        "fight_id": fid,
        "fighter_norm": fighter_norm,
        "round": rnd,
        "kd": 0,
        "sub_att": 0,
        "rev": 0,
        "ctrl_sec": 0,
        "sig_str_landed": 0,
        "sig_str_att": 0,
        "total_str_landed": 0,
        "total_str_att": 0,
        "td_landed": 0,
        "td_att": 0,
        "head_landed": 0,
        "head_att": 0,
        "body_landed": 0,
        "body_att": 0,
        "leg_landed": 0,
        "leg_att": 0,
        "distance_landed": 0,
        "distance_att": 0,
        "clinch_landed": 0,
        "clinch_att": 0,
        "ground_landed": 0,
        "ground_att": 0,
    }
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


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_expected_score_is_symmetric_and_half_at_parity():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)


def test_expected_score_matches_hand_computation():
    """A 400-point gap is 10:1 under the standard logistic form."""
    assert expected_score(1900, 1500, scale=400.0) == pytest.approx(10 / 11, abs=1e-9)


def test_elo_updates_by_the_hand_computed_delta():
    """At parity with K=32, the winner gains exactly 16 points."""
    fights = pd.DataFrame([_fight("f1", D1, "a", "b", "a"), _fight("f2", D2, "a", "b", "a")])
    elo = build_elo(fights, initial_rating=1500.0, k_factor=32.0, scale=400.0).set_index("fight_id")
    assert elo.loc["f1", "elo_1"] == pytest.approx(1500.0)
    assert elo.loc["f2", "elo_1"] == pytest.approx(1516.0)
    assert elo.loc["f2", "elo_2"] == pytest.approx(1484.0)


def test_elo_is_zero_sum():
    fights = pd.DataFrame([_fight("f1", D1, "a", "b", "a"), _fight("f2", D2, "a", "b", "b")])
    elo = build_elo(fights, initial_rating=1500.0, k_factor=32.0)
    for row in elo.itertuples(index=False):
        assert row.elo_1 + row.elo_2 == pytest.approx(3000.0)


def test_elo_ignores_draws_but_other_fights_still_count():
    """A draw must not move ratings, while a subsequent win still does."""
    fights = pd.DataFrame(
        [
            _fight("f1", D1, "a", "b", None, decisive=False),
            _fight("f2", D2, "a", "b", "a"),
            _fight("f3", D3, "a", "b", "a"),
        ]
    )
    elo = build_elo(fights, initial_rating=1500.0, k_factor=32.0).set_index("fight_id")
    assert elo.loc["f2", "elo_1"] == pytest.approx(1500.0), "the draw moved Elo"
    assert elo.loc["f3", "elo_1"] == pytest.approx(1516.0)


def test_elo_does_not_advance_within_a_single_date():
    """Two bouts for the same fighter on one night must share one pre-fight rating."""
    fights = pd.DataFrame(
        [
            _fight("f1", D1, "a", "b", "a"),
            _fight("f2", D1, "a", "c", "a"),
        ]
    )
    elo = build_elo(fights, initial_rating=1500.0, k_factor=32.0).set_index("fight_id")
    a1 = elo.loc["f1", "elo_1"] if "a" < "b" else elo.loc["f1", "elo_2"]
    a2 = elo.loc["f2", "elo_1"] if "a" < "c" else elo.loc["f2", "elo_2"]
    assert a1 == pytest.approx(a2) == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# Rolling statistics: hand-computed
# ---------------------------------------------------------------------------


def test_strike_rate_is_per_minute_of_fight_time():
    """One prior fight: 50 significant strikes landed over 5:00 => 10.0 per minute."""
    fights = pd.DataFrame(
        [
            _fight("f1", D1, "a", "b", "a", rnd=1, secs=ROUND_SECONDS),
            _fight("f2", D2, "a", "b", "a"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat_row("f1", "n_a", 1, sig_str_landed=50, sig_str_att=100),
            _stat_row("f1", "n_b", 1, sig_str_landed=20, sig_str_att=80),
        ]
    )
    feats = build_features(fights, stats, _fighters(["a", "b"]), rolling_windows=(3,))
    row = feats.loc[feats["fight_id"] == "f2"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    other = "2" if slot == "1" else "1"

    assert row[f"sig_str_landed_pm_career_{slot}"] == pytest.approx(10.0)
    assert row[f"sig_str_acc_career_{slot}"] == pytest.approx(0.5)
    # Absorbed: the opponent landed 20 over the same 5 minutes.
    assert row[f"sig_str_absorbed_pm_career_{slot}"] == pytest.approx(4.0)
    # Defence: 1 - 20/80.
    assert row[f"sig_str_def_career_{slot}"] == pytest.approx(0.75)
    # And the opponent's mirror values.
    assert row[f"sig_str_landed_pm_career_{other}"] == pytest.approx(4.0)
    assert row[f"sig_str_def_career_{other}"] == pytest.approx(0.5)


def test_duration_spans_completed_rounds_plus_the_final_clock():
    """A round-3 finish at 2:00 is 2 full rounds (600s) plus 120s = 12 minutes."""
    fights = pd.DataFrame(
        [
            _fight("f1", D1, "a", "b", "a", rnd=3, secs=120),
            _fight("f2", D2, "a", "b", "a"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat_row("f1", "n_a", 1, sig_str_landed=120, sig_str_att=120),
            _stat_row("f1", "n_b", 1, sig_str_landed=0, sig_str_att=10),
        ]
    )
    feats = build_features(fights, stats, _fighters(["a", "b"]), rolling_windows=(3,))
    row = feats.loc[feats["fight_id"] == "f2"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"sig_str_landed_pm_career_{slot}"] == pytest.approx(10.0)


def test_takedowns_are_per_fifteen_minutes():
    """3 takedowns in 5 minutes is 9.0 per 15 minutes."""
    fights = pd.DataFrame(
        [
            _fight("f1", D1, "a", "b", "a", rnd=1, secs=ROUND_SECONDS),
            _fight("f2", D2, "a", "b", "a"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat_row("f1", "n_a", 1, sig_str_att=1, td_landed=3, td_att=4),
            _stat_row("f1", "n_b", 1, sig_str_att=1, td_landed=0, td_att=2),
        ]
    )
    feats = build_features(fights, stats, _fighters(["a", "b"]), rolling_windows=(3,))
    row = feats.loc[feats["fight_id"] == "f2"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    other = "2" if slot == "1" else "1"
    assert row[f"td_landed_p15_career_{slot}"] == pytest.approx(9.0)
    # a defended both of b's attempts.
    assert row[f"td_def_career_{slot}"] == pytest.approx(1.0)
    # b defended 1 of a's 4.
    assert row[f"td_def_career_{other}"] == pytest.approx(0.25)


def test_rolling_window_keeps_only_the_most_recent_n_fights():
    """With window=2, the oldest of three prior fights must drop out."""
    fights = pd.DataFrame(
        [
            _fight("f1", dt.date(2019, 1, 1), "a", "b", "a", rnd=1, secs=ROUND_SECONDS),
            _fight("f2", dt.date(2019, 6, 1), "a", "c", "a", rnd=1, secs=ROUND_SECONDS),
            _fight("f3", dt.date(2019, 9, 1), "a", "d", "a", rnd=1, secs=ROUND_SECONDS),
            _fight("f4", D3, "a", "e", "a"),
        ]
    )
    stats = pd.DataFrame(
        [
            _stat_row("f1", "n_a", 1, sig_str_landed=300, sig_str_att=300),
            _stat_row("f1", "n_b", 1, sig_str_att=1),
            _stat_row("f2", "n_a", 1, sig_str_landed=30, sig_str_att=30),
            _stat_row("f2", "n_c", 1, sig_str_att=1),
            _stat_row("f3", "n_a", 1, sig_str_landed=30, sig_str_att=30),
            _stat_row("f3", "n_d", 1, sig_str_att=1),
        ]
    )
    feats = build_features(
        fights, stats, _fighters(["a", "b", "c", "d", "e"]), rolling_windows=(2,)
    )
    row = feats.loc[feats["fight_id"] == "f4"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    # Last two fights: 60 strikes over 10 minutes = 6.0/min. Career: 360/15 = 24.0/min.
    assert row[f"sig_str_landed_pm_w2_{slot}"] == pytest.approx(6.0)
    assert row[f"sig_str_landed_pm_career_{slot}"] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Records, streaks, layoff
# ---------------------------------------------------------------------------


def test_win_rate_and_streaks_count_only_prior_fights():
    fights = pd.DataFrame(
        [
            _fight("f1", dt.date(2019, 1, 1), "a", "b", "a"),
            _fight("f2", dt.date(2019, 6, 1), "a", "c", "a"),
            _fight("f3", dt.date(2019, 9, 1), "a", "d", "d"),
            _fight("f4", D3, "a", "e", "a"),
        ]
    )
    feats = build_features(
        fights, pd.DataFrame(), _fighters(["a", "b", "c", "d", "e"]), rolling_windows=(3,)
    )
    row = feats.loc[feats["fight_id"] == "f4"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"n_prior_fights_{slot}"] == 3
    assert row[f"win_rate_{slot}"] == pytest.approx(2 / 3)
    assert row[f"loss_streak_{slot}"] == 1
    assert row[f"win_streak_{slot}"] == 0


def test_days_since_last_fight_is_measured_to_the_event_date():
    fights = pd.DataFrame(
        [
            _fight("f1", dt.date(2020, 1, 1), "a", "b", "a"),
            _fight("f2", dt.date(2020, 3, 1), "a", "c", "a"),
        ]
    )
    feats = build_features(fights, pd.DataFrame(), _fighters(["a", "b", "c"]), rolling_windows=(3,))
    row = feats.loc[feats["fight_id"] == "f2"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"days_since_last_{slot}"] == pytest.approx(60.0)


def test_finish_rate_counts_only_wins_by_stoppage():
    fights = pd.DataFrame(
        [
            _fight("f1", dt.date(2019, 1, 1), "a", "b", "a", method="KO/TKO"),
            _fight("f2", dt.date(2019, 6, 1), "a", "c", "a", method="DEC"),
            _fight("f3", D3, "a", "d", "a"),
        ]
    )
    feats = build_features(
        fights, pd.DataFrame(), _fighters(["a", "b", "c", "d"]), rolling_windows=(3,)
    )
    row = feats.loc[feats["fight_id"] == "f3"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"finish_rate_{slot}"] == pytest.approx(0.5)


def test_draws_count_as_experience_but_break_no_streak():
    fights = pd.DataFrame(
        [
            _fight("f1", dt.date(2019, 1, 1), "a", "b", "a"),
            _fight("f2", dt.date(2019, 6, 1), "a", "c", None, decisive=False),
            _fight("f3", D3, "a", "d", "a"),
        ]
    )
    feats = build_features(
        fights, pd.DataFrame(), _fighters(["a", "b", "c", "d"]), rolling_windows=(3,)
    )
    row = feats.loc[feats["fight_id"] == "f3"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"n_prior_fights_{slot}"] == 2, "a draw is still a fight"
    assert row[f"win_streak_{slot}"] == 1, "a draw should not reset the win streak"
    assert row[f"win_rate_{slot}"] == pytest.approx(0.5), "1 win in 2 fights"


# ---------------------------------------------------------------------------
# Debut handling
# ---------------------------------------------------------------------------


def test_a_debutant_gets_nan_rates_and_a_flag_not_zeros():
    fights = pd.DataFrame([_fight("f1", D1, "a", "b", "a")])
    feats = build_features(fights, pd.DataFrame(), _fighters(["a", "b"]), rolling_windows=(3,))
    row = feats.iloc[0]
    for slot in ("1", "2"):
        assert row[f"is_debut_{slot}"]
        assert row[f"n_prior_fights_{slot}"] == 0
        assert pd.isna(row[f"win_rate_{slot}"])
        assert pd.isna(row[f"days_since_last_{slot}"])
        assert pd.isna(row[f"sig_str_landed_pm_career_{slot}"])
        assert not row[f"has_stats_history_{slot}"]


def test_a_fighter_with_history_but_no_round_stats_is_flagged():
    """Pre-2001 fights often lack per-round statistics. That is missing data, not zero."""
    fights = pd.DataFrame([_fight("f1", D1, "a", "b", "a"), _fight("f2", D2, "a", "b", "a")])
    feats = build_features(fights, pd.DataFrame(), _fighters(["a", "b"]), rolling_windows=(3,))
    row = feats.loc[feats["fight_id"] == "f2"].iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"n_prior_fights_{slot}"] == 1
    assert not row[f"is_debut_{slot}"]
    assert not row[f"has_stats_history_{slot}"]
    assert pd.isna(row[f"sig_str_landed_pm_career_{slot}"])


# ---------------------------------------------------------------------------
# Differences and physical attributes
# ---------------------------------------------------------------------------


def test_difference_columns_equal_slot_one_minus_slot_two():
    cfg = get_config()
    try:
        feats = db.read_table(cfg.paths.processed_dir, "features")
    except FileNotFoundError:
        pytest.skip("features absent; run `make features`")

    for col in FEATURE_COLS:
        if not col.endswith("_diff"):
            continue
        base = col[: -len("_diff")]
        if f"{base}_1" not in feats.columns:
            continue
        lhs = (feats[f"{base}_1"] - feats[f"{base}_2"]).to_numpy(dtype=float)
        rhs = feats[col].to_numpy(dtype=float)
        both_nan = np.isnan(lhs) & np.isnan(rhs)
        assert np.allclose(lhs[~both_nan], rhs[~both_nan], equal_nan=True, atol=1e-9), col


def test_physical_attributes_come_from_the_fighter_table():
    fights = pd.DataFrame([_fight("f1", D1, "a", "b", "a")])
    fighters = _fighters(["a", "b"])
    fighters.loc[fighters["fighter_id"] == "a", "reach_cm"] = 200.0
    feats = build_features(fights, pd.DataFrame(), fighters, rolling_windows=(3,))
    row = feats.iloc[0]
    slot = "1" if row["fighter_1"] == "a" else "2"
    assert row[f"reach_{slot}"] == pytest.approx(200.0)
    expected_diff = 200.0 - 183.0 if slot == "1" else 183.0 - 200.0
    assert row["reach_diff"] == pytest.approx(expected_diff)


def test_history_table_links_opponent_statistics_correctly():
    fights = pd.DataFrame([_fight("f1", D1, "a", "b", "a", rnd=1, secs=ROUND_SECONDS)])
    stats = pd.DataFrame(
        [
            _stat_row("f1", "n_a", 1, sig_str_landed=10, sig_str_att=20),
            _stat_row("f1", "n_b", 1, sig_str_landed=5, sig_str_att=30),
        ]
    )
    hist = fighter_fight_history(fights, stats).set_index("fighter_id")
    assert hist.loc["a", "own_sig_str_landed"] == 10
    assert hist.loc["a", "opp_sig_str_landed"] == 5
    assert hist.loc["b", "own_sig_str_att"] == 30
    assert hist.loc["b", "opp_sig_str_att"] == 20


# ---------------------------------------------------------------------------
# The real table
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_features():
    cfg = get_config()
    try:
        return db.read_table(cfg.paths.processed_dir, "features")
    except FileNotFoundError:
        pytest.skip("features absent; run `make features`")


def test_one_row_per_fight(real_features):
    cfg = get_config()
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    assert len(real_features) == len(fights)
    assert not real_features["fight_id"].duplicated().any()


def test_rates_are_in_plausible_ranges(real_features):
    """Catches unit errors, which point-in-time tests would happily pass.

    The per-minute bound is deliberately loose. A fighter whose only prior bout was a
    15-second knockout genuinely has a very high strike rate -- Song Kenan landed 14
    significant strikes in 15 seconds against Bobby Nash, which is exactly 56.0 per
    minute. That is arithmetic, not corruption: rates over tiny denominators are
    legitimately extreme, and they settle as a fighter accumulates fight time. Two
    strikes per second (120/min) is past any physical plausibility, so that is the bound.
    """
    f = real_features
    rate = f["sig_str_landed_pm_career_1"].dropna()
    assert rate.between(0, 120).all(), f"max {rate.max()} strikes/min is physically impossible"
    assert rate.median() < 10, "median strike rate should sit in single digits"

    # Ratios are bounded by construction; a breach means a numerator/denominator swap.
    for col in ("sig_str_acc_career_1", "sig_str_def_career_1", "td_def_career_1"):
        assert f[col].dropna().between(0, 1).all(), col
    assert f["win_rate_1"].dropna().between(0, 1).all()
    assert f["finish_rate_1"].dropna().between(0, 1).all()
    assert f["days_since_last_1"].dropna().ge(0).all()
    assert f["n_prior_fights_1"].ge(0).all()


def test_priced_and_decisive_set_is_the_documented_size(real_features):
    """The backtest scores model and market on exactly this set."""
    n = int((real_features["is_priced"] & real_features["label"].notna()).sum())
    assert n == 6790, (
        f"expected 6,790 priced decisive fights, found {n}. 6,824 fights are priced; "
        "34 are draws, DQs, overturned results or no contests with no binary winner."
    )


def test_no_feature_is_entirely_null(real_features):
    for col in FEATURE_COLS:
        assert real_features[col].notna().any(), f"{col} is null for every fight"


def test_feature_count_is_stable(real_features):
    """36 pairwise/bout features + 10 rate metrics x 3 windows x 3 slots = 126."""
    assert len(FEATURE_COLS) == 126
    assert len(set(FEATURE_COLS)) == len(FEATURE_COLS), "duplicate feature name"
    for col in FEATURE_COLS:
        assert col in real_features.columns
