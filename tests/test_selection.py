"""Per-fold top-k feature selection: the grouping rules, the fit, and no peeking.

Feature selection is a classic leakage path -- a selector that sees the rows it will be
scored on picks columns that only *look* predictive. ``select_top_k`` is handed one fold's
training rows and nothing else, so a column that predicts the label only on the scored
rows must never be chosen. The grouping rules (whole ``_1/_2/_diff`` groups first, then a
partial group's ``_diff``) are pinned on hand-set gains, where every answer is checkable
on paper, and on a synthetic frame whose gains come from a real fit.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments import runner, selection, stages
from ufc_predictor.experiments.runner import ExperimentSpec, run_config, training_pool
from ufc_predictor.experiments.selection import feature_groups, pick_top_k, select_top_k
from ufc_predictor.features.pit import ALL_FEATURE_COLS, FEATURE_COLS
from ufc_predictor.models import tune
from ufc_predictor.models.lgbm import LightGBMModel

#: Training rows end here; the "scored" rows run from here to the end of the selection
#: window (2014). Nothing in this file reads 2015 or later.
SPLIT = dt.date(2012, 1, 1)
SHORT = dt.date(2010, 6, 30)  # a short slice of the selection window keeps tests fast


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def panel(cfg):
    try:
        return runner.base_panel(cfg)
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data && make features`")


@pytest.fixture(scope="module")
def selection_rows(cfg, panel):
    """The base panel, cut at the end of the selection window."""
    rows = panel.loc[panel["date"] <= cfg.windows.selection_end].reset_index(drop=True)
    assert rows["date"].max() < cfg.windows.confirmation_start
    return rows


@pytest.fixture(scope="module")
def train(selection_rows, cfg):
    pool, _ = training_pool(
        selection_rows, SPLIT, ExperimentSpec(), int(cfg.backtest["min_train_fights"])
    )
    return pool


@pytest.fixture(scope="module")
def top25(train, cfg):
    return select_top_k(train, FEATURE_COLS, 25, cfg)


@pytest.fixture(scope="module")
def gains25(train, cfg):
    return selection.gain_importance(train, FEATURE_COLS, cfg)


def _ranked(features: list[str], gains: dict[str, float]) -> list[str]:
    groups = feature_groups(features)
    return sorted(groups, key=lambda b: (-sum(gains[c] for c in groups[b]), b))


def _synthetic(cfg, n: int = 2000) -> pd.DataFrame:
    """A date-ordered frame whose columns carry the label with chosen noise levels."""
    rng = np.random.default_rng(cfg.seed)
    y = rng.integers(0, 2, n)
    frame = {
        "date": [dt.date(2000, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "label": y.astype(float),
    }

    def group(base: str, noise: float | None, members=("_1", "_2", "_diff")) -> None:
        one = (y if noise else 0) + rng.normal(0, noise or 1.0, n)
        two = (1 - y if noise else 0) + rng.normal(0, noise or 1.0, n)
        values = {"_1": one, "_2": two, "_diff": one - two}
        for m in members:
            frame[f"{base}{m}"] = values[m]

    # Listed noise-first so the output order is visibly the input order, not the rank.
    group("noise", None)
    group("flat", None, members=("_1", "_2"))
    group("medium", 1.5)
    group("strong", 0.4)
    return pd.DataFrame(frame)


SYNTH_FEATURES = [
    "noise_1",
    "noise_2",
    "noise_diff",
    "flat_1",
    "flat_2",
    "medium_1",
    "medium_2",
    "medium_diff",
    "strong_1",
    "strong_2",
    "strong_diff",
]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def test_groups_cover_every_feature_once_with_the_ruled_singletons():
    groups = feature_groups(ALL_FEATURE_COLS)
    members = [c for cols in groups.values() for c in cols]
    assert sorted(members) == sorted(ALL_FEATURE_COLS)
    assert groups["elo"] == ["elo_1", "elo_2", "elo_diff"]
    assert groups["is_debut"] == ["is_debut_1", "is_debut_2"]
    assert groups["glicko_rd"] == ["glicko_rd_1", "glicko_rd_2"]
    for alone in ("title_bout", "scheduled_rounds", "grappler_vs_striker", "glicko_rd_sum"):
        assert groups[alone] == [alone]
    assert groups["glicko_expected"] == ["glicko_expected_1"]  # a lone _1


def test_groups_reject_duplicate_or_ambiguous_columns():
    with pytest.raises(ValueError, match="duplicate"):
        feature_groups(["elo_1", "elo_1"])
    with pytest.raises(ValueError, match="ambiguous"):
        feature_groups(["elo", "elo_1"])


# ---------------------------------------------------------------------------
# Ranking and the three passes, on hand-set gains
# ---------------------------------------------------------------------------

ABC = ["a_1", "a_2", "a_diff", "b_1", "b_2", "b_diff", "c_1", "c_2"]


def _gains(**group_gains: float) -> dict[str, float]:
    """Spread each group's total gain evenly over its members."""
    groups = feature_groups(ABC)
    return {
        c: total / len(groups[base]) for base, total in group_gains.items() for c in groups[base]
    }


def test_pass_one_stops_at_the_first_group_that_does_not_fit():
    features = [*ABC, "s"]
    gains = {**_gains(a=9.0, b=6.0, c=5.0), "s": 0.0}  # rank a, b, c, s
    # b (3) does not fit after a (3) at k=5, so pass 1 stops: the smaller c, which would
    # fit, is not taken; b's _diff and then b_1 fill the slots instead.
    assert pick_top_k(features, gains, 5) == ["a_1", "a_2", "a_diff", "b_1", "b_diff"]


def test_a_zero_gain_singleton_no_longer_takes_the_last_slot_from_a_better_diff():
    features = [*ABC, "s"]
    gains = {**_gains(a=9.0, b=6.0, c=5.0), "s": 0.0}
    # One slot left after a: it goes to b_diff (gain 2), not to s (gain 0) for fitting.
    assert pick_top_k(features, gains, 4) == ["a_1", "a_2", "a_diff", "b_diff"]


def test_pass_two_adds_the_next_groups_diff_and_passes_over_a_group_without_one():
    gains = _gains(a=9.0, c=8.0, b=6.0)  # c (no _diff) ranks above b
    assert pick_top_k(ABC, gains, 4) == ["a_1", "a_2", "a_diff", "b_diff"]


def test_pass_three_fills_a_partial_group_diff_first_then_1():
    gains = _gains(c=9.0, a=8.0, b=6.0)  # c, a whole (5); b_diff (6); then b_1
    assert pick_top_k(ABC, gains, 7) == ["a_1", "a_2", "a_diff", "b_1", "b_diff", "c_1", "c_2"]
    gains = _gains(a=9.0, b=8.0, c=6.0)  # a, b whole (6); c has no _diff -> c_1
    assert pick_top_k(ABC, gains, 7) == ["a_1", "a_2", "a_diff", "b_1", "b_2", "b_diff", "c_1"]


def test_groups_rank_by_summed_gain_not_their_best_member():
    gains = {**_gains(b=9.0), "a_1": 8.0, "a_2": 0.0, "a_diff": 0.0}
    assert pick_top_k(ABC[:6], gains, 3) == ["b_1", "b_2", "b_diff"]


def test_ties_break_by_base_name_and_output_keeps_the_input_order():
    features = ["z_1", "z_2", "z_diff", "a_1", "a_2", "a_diff", "s"]
    gains = dict.fromkeys(features, 0.0)
    assert pick_top_k(features, gains, 3) == ["a_1", "a_2", "a_diff"]
    assert pick_top_k(features, gains, 4) == ["a_1", "a_2", "a_diff", "s"]
    assert pick_top_k(features, gains, 5) == ["z_diff", "a_1", "a_2", "a_diff", "s"]


def test_k_at_or_above_the_feature_count_returns_the_features_unchanged():
    gains = _gains(a=1.0, b=2.0, c=3.0)
    assert pick_top_k(ABC, gains, len(ABC)) == ABC
    assert pick_top_k(ABC, gains, 100) == ABC
    with pytest.raises(ValueError, match="positive"):
        pick_top_k(ABC, gains, 0)
    with pytest.raises(KeyError):
        pick_top_k(ABC, {"a_1": 1.0}, 3)


# ---------------------------------------------------------------------------
# The fit (ruling K1), on a synthetic frame
# ---------------------------------------------------------------------------


def test_the_fit_uses_config_seed_params_and_regime_without_calibration(cfg, monkeypatch):
    seen: list[dict] = []

    class Spy(LightGBMModel):
        def __init__(self, **kwargs):
            seen.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(selection, "LightGBMModel", Spy)
    frame = _synthetic(cfg)
    select_top_k(frame, SYNTH_FEATURES, 3, cfg)
    fit = cfg.lightgbm_fit
    assert seen == [
        {
            "seed": cfg.seed,
            "params": dict(cfg.lightgbm),  # params=None means the config's
            "feature_cols": SYNTH_FEATURES,
            "validation_fraction": fit["validation_fraction"],
            "early_stopping_rounds": fit["early_stopping_rounds"],
            "min_validation_rows": fit["min_validation_rows"],
            "calibrate": False,
        }
    ]
    custom = {**cfg.lightgbm, "num_leaves": 7}
    select_top_k(frame, SYNTH_FEATURES, 3, cfg, params=custom)
    assert seen[-1]["params"] == custom


def test_a_strong_group_is_kept_whole_and_the_next_contributes_its_diff(cfg):
    frame = _synthetic(cfg)
    gains = selection.gain_importance(frame, SYNTH_FEATURES, cfg)
    total = {
        base: sum(gains[c] for c in cols) for base, cols in feature_groups(SYNTH_FEATURES).items()
    }
    assert total["strong"] > total["medium"] > max(total["noise"], total["flat"])
    assert select_top_k(frame, SYNTH_FEATURES, 4, cfg) == [
        "medium_diff",
        "strong_1",
        "strong_2",
        "strong_diff",
    ]


def test_importance_is_total_gain_not_split_count(cfg):
    """A binary column is split rarely but decisively; a fuzzy one often but weakly.

    Here ``fuzzy`` collects roughly ten times the splits and ``sharp`` roughly five times
    the gain, so only a gain ranking keeps ``sharp``.
    """
    rng = np.random.default_rng(cfg.seed)
    n = 2000
    y = rng.integers(0, 2, n)
    sharp = np.where(rng.random(n) < 0.15, 1 - y, y).astype(float)  # 85% the label
    one, two = y + rng.normal(0, 3.0, n), (1 - y) + rng.normal(0, 3.0, n)
    frame = pd.DataFrame(
        {
            "date": [dt.date(2000, 1, 1) + dt.timedelta(days=i) for i in range(n)],
            "label": y.astype(float),
            "fuzzy_1": one,
            "fuzzy_2": two,
            "fuzzy_diff": one - two,
            "sharp_1": sharp,
            "sharp_2": 1 - sharp,
            "sharp_diff": 2 * sharp - 1,
        }
    )
    features = ["fuzzy_1", "fuzzy_2", "fuzzy_diff", "sharp_1", "sharp_2", "sharp_diff"]
    assert select_top_k(frame, features, 3, cfg) == ["sharp_1", "sharp_2", "sharp_diff"]


# ---------------------------------------------------------------------------
# Real selection-window data
# ---------------------------------------------------------------------------


def test_k_is_respected_exactly_on_the_base_features(top25, gains25):
    assert len(top25) == 25 == len(set(top25))
    assert top25 == [c for c in FEATURE_COLS if c in set(top25)]  # model column order
    groups = feature_groups(FEATURE_COLS)
    for cols in groups.values():
        picked = [c for c in cols if c in top25]
        if picked and picked != cols:  # a partial group always carries its _diff
            assert any(c.endswith("_diff") for c in picked), cols
    # Pass 1 takes a prefix of the ranking and stops: no lower-ranked group that merely
    # fits (a singleton such as scheduled_rounds) jumps ahead of a better group's _diff.
    ranked = _ranked(FEATURE_COLS, gains25)
    whole = [b for b in ranked if set(groups[b]) <= set(top25)]
    assert whole == ranked[: len(whole)]


def test_selection_is_deterministic_under_the_seed(train, cfg, top25, gains25):
    assert select_top_k(train, FEATURE_COLS, 25, cfg) == top25
    assert pick_top_k(FEATURE_COLS, gains25, 25) == top25


@pytest.mark.leakage
def test_a_column_that_predicts_only_the_scored_rows_is_never_selected(selection_rows, cfg):
    rng = np.random.default_rng(cfg.seed)
    rows = selection_rows.copy()
    scored = rows["date"] >= SPLIT
    # Named to sort last, so a zero-gain tie can never hand it a slot by name order.
    planted = "zz_planted"
    rows[planted] = np.where(scored, rows["label"], rng.integers(0, 2, len(rows)))
    assert (rows.loc[scored, planted] == rows.loc[scored, "label"]).all()
    features = [*FEATURE_COLS, planted]

    # The runner's own pool for an event dated SPLIT: nothing on or after it.
    train, _ = training_pool(rows, SPLIT, ExperimentSpec(), int(cfg.backtest["min_train_fights"]))
    assert train["date"].max() < SPLIT
    assert planted not in select_top_k(train, features, 25, cfg)

    # The control: the same column, on rows where it does hold, is the top-gain pick.
    visible = train.assign(**{planted: train["label"]})
    gains = selection.gain_importance(visible, features, cfg)
    assert max(gains, key=gains.get) == planted
    assert planted in pick_top_k(features, gains, 25)


# ---------------------------------------------------------------------------
# The real call paths: what the runner and the re-tune actually hand the selector
# ---------------------------------------------------------------------------


@pytest.fixture
def selection_calls(monkeypatch):
    """Record the training frame of every ``select_top_k`` call, then delegate.

    The runner and ``stages.retune`` import ``select_top_k`` inside the function, so
    patching the module attribute intercepts every call they make.
    """
    calls: list[dict] = []
    real = selection.select_top_k

    def spy(train, features, k, cfg, *, params=None):
        calls.append({"max_date": train["date"].max(), "rows": len(train)})
        return real(train, features, k, cfg, params=params)

    monkeypatch.setattr(selection, "select_top_k", spy)
    return calls


@pytest.mark.leakage
def test_the_runner_selects_on_rows_strictly_before_every_folds_cutoff(cfg, panel, selection_calls):
    every_event = dataclasses.replace(
        cfg, backtest={**cfg.backtest, "retrain_every_n_events": 1}
    )  # a selection per event, so a leak cannot hide between refits
    out = run_config(
        every_event, ExperimentSpec(top_k=25), window="selection", panel=panel, through=SHORT
    )
    folds = out.attrs["folds"]
    assert len(selection_calls) == len(folds) > 1
    for call, fold in zip(selection_calls, folds, strict=True):
        assert call["max_date"] < fold["cutoff"], fold["fold"]
        assert call["rows"] == fold["train_rows"]
        assert len(fold["features"]) == 25
        assert set(fold["features"]) <= set(FEATURE_COLS)


@pytest.mark.leakage
def test_the_retune_selects_on_rows_strictly_before_its_holdout(
    cfg, panel, selection_calls, monkeypatch
):
    one = {key: [cfg.lightgbm[key]] for key in tune.SEARCH_SPACE}  # one candidate: fast
    monkeypatch.setattr(stages, "SEARCH_SPACE", one)
    result = stages.retune(ExperimentSpec(top_k=25), panel, cfg)
    _, holdout = stages.retune_split(panel, cfg)
    assert result["n_candidates"] == 1
    assert len(selection_calls) == 1
    assert selection_calls[0]["max_date"] < holdout["date"].min()
    assert selection_calls[0]["rows"] == result["train_rows"]
