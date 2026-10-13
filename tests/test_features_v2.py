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
    FEATURE_COLS,
    FEATURE_FAMILIES,
    RATE_METRICS,
    RECENCY_COLS,
    ROUND_SECONDS,
    build_features,
    feature_columns,
    fighter_fight_history,
)
from ufc_predictor.storage import db

META_COLS = ["fight_id", "date", "fighter_1", "fighter_2", "label", "is_decisive"]
NEW_FAMILIES = ("adjusted", "recency")
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
    for col in [*ADJUSTED_COLS, *RECENCY_COLS]:
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
    cols = ["fight_id", *TOY_BASE, *ADJUSTED_COLS, *RECENCY_COLS]

    early_a = a.loc[a["date"] <= D2, cols].sort_values("fight_id").reset_index(drop=True)
    early_b = b.loc[b["date"] <= D2, cols].sort_values("fight_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a, early_b, check_exact=True)

    for col in [c[:-2] for c in ADJUSTED_COLS + RECENCY_COLS if c.endswith("_1")]:
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
    assert list(FEATURE_FAMILIES)[:3] == ["base", "adjusted", "recency"]
    assert FEATURE_FAMILIES["base"] == FEATURE_COLS
    assert len(FEATURE_COLS) == 117
    assert FEATURE_FAMILIES["adjusted"] == ADJUSTED_COLS
    assert FEATURE_FAMILIES["recency"] == RECENCY_COLS
    assert [c for cols in FEATURE_FAMILIES.values() for c in cols] == ALL_FEATURE_COLS
    assert len(set(ALL_FEATURE_COLS)) == len(ALL_FEATURE_COLS), "duplicate feature name"


def test_adjusted_and_recency_columns_are_named_per_metric_and_slot():
    slots = ("1", "2", "diff")
    assert [f"adj_{m}_career_{s}" for m in RATE_METRICS for s in slots] == ADJUSTED_COLS
    assert [f"ewm_{m}_{s}" for m in RATE_METRICS for s in slots] == RECENCY_COLS
    assert len(ADJUSTED_COLS) + len(RECENCY_COLS) == 60


def test_the_mirror_table_covers_every_rate_metric():
    assert tuple(adjusted.MIRROR) == RATE_METRICS


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
    empty = [c for c in ADJUSTED_COLS + RECENCY_COLS if full_build[c].isna().all()]
    assert not empty, f"entirely null: {empty}"


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
