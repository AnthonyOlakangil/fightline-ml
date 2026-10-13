"""Model interface and behaviour tests.

The point of a shared interface is that the backtest cannot treat one model more
favourably than another, so most of these tests are parametrized across all three. The
remainder pin the things that would make the eventual market comparison dishonest:
random validation, eval-window tuning, odds leaking into the features, or a seed that
does not actually control the result.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.features.pit import FEATURE_COLS
from ufc_predictor.models.base import PROB_EPS, Model, validate_predictions
from ufc_predictor.models.elo_model import EloModel
from ufc_predictor.models.lgbm import LightGBMModel
from ufc_predictor.models.market import MarketModel
from ufc_predictor.models.registry import MODEL_NAMES, build_all, build_model
from ufc_predictor.models.tune import (
    NOISE_TOLERANCE,
    capacity_rank,
    chronological_holdout,
    dev_frame,
    select_best,
)
from ufc_predictor.storage import db


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def dev_data(cfg):
    """Labelled dev-window fights joined to their closing prices.

    Dev window only: these tests must not consume the held-out period either.
    """
    try:
        feats = db.read_table(cfg.paths.processed_dir, "features")
        odds = db.read_table(cfg.paths.processed_dir, "odds")
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data && make features`")

    df = feats.merge(odds[["fight_id", "p1_market"]], on="fight_id", how="left")
    dev = df.loc[
        df["label"].notna() & df["p1_market"].notna() & (df["date"] <= cfg.windows.dev_end)
    ].sort_values(["date", "fight_id"], kind="stable")
    assert len(dev) > 1000
    return dev.reset_index(drop=True)


@pytest.fixture(scope="module")
def split(dev_data):
    n = int(len(dev_data) * 0.7)
    return dev_data.iloc[:n].copy(), dev_data.iloc[n:].copy()


# ---------------------------------------------------------------------------
# The shared interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_satisfies_the_interface(name, cfg, split):
    """One interface, so the backtest cannot favour one model over another."""
    train, test = split
    model = build_model(name, cfg)
    assert isinstance(model, Model)
    assert model.name == name

    returned = model.fit(train, train["label"].astype(int))
    assert returned is model, "fit must return self so calls can chain"

    p = model.predict_proba(test)
    assert isinstance(p, np.ndarray)
    assert p.shape == (len(test),)
    assert p.dtype == np.float64
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_are_clipped_away_from_zero_and_one(name, cfg, split):
    """Log loss is unbounded at the endpoints; an overconfident model must be penalised,
    not produce an infinity that breaks the metric."""
    train, test = split
    model = build_model(name, cfg).fit(train, train["label"].astype(int))
    p = model.predict_proba(test)
    assert p.min() >= PROB_EPS
    assert p.max() <= 1 - PROB_EPS


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_refitting_does_not_accumulate_state(name, cfg, split):
    """The walk-forward engine refits repeatedly; a model that carried state across
    calls would leak information between folds."""
    train, test = split
    a = build_model(name, cfg).fit(train, train["label"].astype(int)).predict_proba(test)

    # The first fit uses 80% of the slice -- enough to satisfy LightGBM's
    # minimum-validation-tail guard, since the point here is that the second fit fully
    # replaces the first, not that a tiny slice can be fitted.
    model = build_model(name, cfg)
    warm = train.iloc[: int(len(train) * 0.8)]
    model.fit(warm, warm["label"].astype(int))
    b = model.fit(train, train["label"].astype(int)).predict_proba(test)

    np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_are_deterministic_under_a_fixed_seed(name, cfg, split):
    train, test = split
    a = build_model(name, cfg).fit(train, train["label"].astype(int)).predict_proba(test)
    b = build_model(name, cfg).fit(train, train["label"].astype(int)).predict_proba(test)
    np.testing.assert_array_equal(a, b)


def test_build_all_returns_every_model_in_report_order(cfg):
    models = build_all(cfg)
    assert list(models) == list(MODEL_NAMES)
    assert models["market"].requires_fit is False


def test_unknown_model_name_raises(cfg):
    with pytest.raises(KeyError, match="unknown model"):
        build_model("xgboost", cfg)


# ---------------------------------------------------------------------------
# validate_predictions
# ---------------------------------------------------------------------------


def test_validate_predictions_rejects_wrong_shape():
    with pytest.raises(ValueError, match="1-D array"):
        validate_predictions(np.zeros((3, 2)), 3, "m")


def test_validate_predictions_rejects_row_mismatch():
    with pytest.raises(ValueError, match="predictions for"):
        validate_predictions(np.array([0.5, 0.5]), 3, "m")


def test_validate_predictions_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        validate_predictions(np.array([0.5, np.nan]), 2, "m")


def test_validate_predictions_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_predictions(np.array([0.5, 1.5]), 2, "m")


# ---------------------------------------------------------------------------
# Market baseline
# ---------------------------------------------------------------------------


def test_market_returns_the_devigged_closing_line():
    m = MarketModel()
    X = pd.DataFrame({"p1_market": [0.25, 0.5, 0.8]})
    np.testing.assert_allclose(m.predict_proba(X), [0.25, 0.5, 0.8])


def test_market_fit_is_a_noop():
    """The closing line is an observation, not something learned."""
    m = MarketModel()
    X = pd.DataFrame({"p1_market": [0.3, 0.7]})
    before = m.predict_proba(X)
    m.fit(X, pd.Series([1, 0]))
    np.testing.assert_array_equal(before, m.predict_proba(X))


def test_market_raises_on_unpriced_rows():
    """Unpriced fights must be excluded before scoring, never imputed: imputing would
    invent a market opinion that never existed."""
    m = MarketModel()
    with pytest.raises(ValueError, match="no closing price"):
        m.predict_proba(pd.DataFrame({"p1_market": [0.4, np.nan]}))


def test_market_raises_when_the_column_is_absent():
    with pytest.raises(KeyError, match="p1_market"):
        MarketModel().predict_proba(pd.DataFrame({"elo_diff": [0.0]}))


# ---------------------------------------------------------------------------
# Elo model
# ---------------------------------------------------------------------------


def test_elo_model_is_monotonic_in_the_rating_gap(cfg, split):
    train, _ = split
    model = EloModel(seed=cfg.seed).fit(train, train["label"].astype(int))
    probe = pd.DataFrame({"elo_diff": [-400.0, -100.0, 0.0, 100.0, 400.0]})
    p = model.predict_proba(probe)
    assert (np.diff(p) > 0).all(), "a larger rating edge must not lower the probability"
    assert abs(p[2] - 0.5) < 0.05, "a zero gap should sit near even money"


def test_elo_model_falls_back_to_the_analytic_curve_on_a_degenerate_slice(cfg):
    """A single-class training slice cannot fit a logistic; the model must still produce
    a sensible forecast rather than a constant."""
    X = pd.DataFrame({"elo_diff": [50.0, 100.0, 150.0]})
    model = EloModel(seed=cfg.seed).fit(X, pd.Series([1, 1, 1]))
    p = model.predict_proba(pd.DataFrame({"elo_diff": [0.0, 400.0]}))
    assert p[0] == pytest.approx(0.5, abs=1e-9)
    assert p[1] == pytest.approx(10 / 11, abs=1e-6)


def test_elo_model_raises_on_missing_elo():
    with pytest.raises(ValueError, match="elo_diff contains NaN"):
        EloModel(seed=1).fit(pd.DataFrame({"elo_diff": [0.0, np.nan]}), pd.Series([0, 1]))


# ---------------------------------------------------------------------------
# LightGBM: time-ordered validation, early stopping, dev-only calibration
# ---------------------------------------------------------------------------


def test_lightgbm_refuses_unsorted_training_data(cfg, split):
    """A chronological holdout is only meaningful if rows arrive in date order."""
    train, _ = split
    shuffled = train.sample(frac=1.0, random_state=cfg.seed)
    model = build_model("lightgbm", cfg)
    with pytest.raises(ValueError, match="not sorted by date"):
        model.fit(shuffled, shuffled["label"].astype(int))


def test_lightgbm_requires_a_date_column(cfg, split):
    """Without dates there is no way to build a time-ordered split, and falling back to
    a random one is exactly what must not happen silently."""
    train, _ = split
    model = build_model("lightgbm", cfg)
    with pytest.raises(KeyError, match="time-ordered"):
        model.fit(train.drop(columns=["date"]), train["label"].astype(int))


def test_lightgbm_early_stopping_selects_fewer_trees_than_the_cap(cfg, split):
    """The tree count is the most overfit-prone hyperparameter, so it is chosen by a
    chronological holdout rather than fixed by hand."""
    train, _ = split
    model = build_model("lightgbm", cfg).fit(train, train["label"].astype(int))
    assert model.best_iteration_ is not None
    assert 0 < model.best_iteration_ < int(cfg.lightgbm["n_estimators"])


def test_lightgbm_validation_slice_is_the_most_recent_data(cfg, split):
    """Verified behaviourally: changing only the most recent 20% of training rows must
    change the fitted model, because that tail is the validation set."""
    train, test = split
    base = build_model("lightgbm", cfg).fit(train, train["label"].astype(int))
    p_base = base.predict_proba(test)

    tail_start = int(len(train) * 0.8)
    perturbed = train.copy()
    perturbed.iloc[tail_start:, perturbed.columns.get_loc("label")] = (
        1 - perturbed.iloc[tail_start:]["label"].to_numpy()
    )
    p_perturbed = (
        build_model("lightgbm", cfg)
        .fit(perturbed, perturbed["label"].astype(int))
        .predict_proba(test)
    )
    assert not np.allclose(p_base, p_perturbed), "the recent tail does not affect the fit"


def test_lightgbm_calibration_is_fitted_and_changes_predictions(cfg, split):
    train, test = split
    uncal = LightGBMModel(seed=cfg.seed, params=cfg.lightgbm, calibrate=False)
    cal = LightGBMModel(seed=cfg.seed, params=cfg.lightgbm, calibrate=True)
    uncal.fit(train, train["label"].astype(int))
    cal.fit(train, train["label"].astype(int))

    assert cal._calibrator is not None
    assert uncal._calibrator is None
    assert not np.allclose(uncal.predict_proba(test), cal.predict_proba(test))


def test_lightgbm_rejects_a_single_class_training_slice(cfg, split):
    train, _ = split
    ones = train.copy()
    ones["label"] = 1.0
    with pytest.raises(ValueError, match="single outcome class"):
        build_model("lightgbm", cfg).fit(ones, ones["label"].astype(int))


def test_lightgbm_raises_on_missing_features(cfg, split):
    train, _ = split
    model = build_model("lightgbm", cfg).fit(train, train["label"].astype(int))
    with pytest.raises(KeyError, match="missing"):
        model.predict_proba(train.drop(columns=FEATURE_COLS[:3]))


def test_lightgbm_never_sees_an_odds_column(cfg):
    """If the model could see the closing line, beating the market would be circular."""
    model = build_model("lightgbm", cfg)
    banned = ("odds", "market", "vig", "implied", "overround", "moneyline")
    offenders = [c for c in model.feature_cols if any(b in c.lower() for b in banned)]
    assert not offenders, f"odds-derived features reached the model: {offenders}"


def test_lightgbm_seed_comes_from_config(cfg):
    model = build_model("lightgbm", cfg)
    assert model.seed == cfg.seed


# ---------------------------------------------------------------------------
# Tuning discipline
# ---------------------------------------------------------------------------


def test_tuning_frame_never_leaves_the_dev_window(cfg):
    dev = dev_frame(cfg)
    assert dev["date"].max() <= cfg.windows.dev_end
    assert dev["date"].max() < cfg.windows.eval_start
    assert dev["label"].notna().all()


def test_tuning_frame_is_date_ordered(cfg):
    dev = dev_frame(cfg)
    assert dev["date"].is_monotonic_increasing


def test_chronological_holdout_puts_later_fights_in_validation(cfg):
    """Not a random split: every validation fight must postdate every training fight."""
    dev = dev_frame(cfg)
    train, val = chronological_holdout(dev)
    assert len(train) > 0 and len(val) > 0
    assert train["date"].max() < val["date"].min()


def test_chronological_holdout_refuses_a_tiny_validation_tail():
    tiny = pd.DataFrame({"date": [dt.date(2012, 1, 1)] * 50, "label": [1.0, 0.0] * 25})
    with pytest.raises(RuntimeError, match="too small to tune on"):
        chronological_holdout(tiny)


def test_selection_prefers_the_most_regularized_tied_candidate():
    """Differences inside the noise tolerance must not decide the model.

    The nominal winner here is the deepest, highest-capacity candidate; selection must
    reject it in favour of a statistically indistinguishable but simpler one.
    """
    results = [
        {
            "log_loss": 0.6145,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "learning_rate": 0.02,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
        },
        {
            "log_loss": 0.6147,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 80,
            "learning_rate": 0.02,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
        },
        {
            "log_loss": 0.9000,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 80,
            "learning_rate": 0.02,
            "colsample_bytree": 0.5,
            "reg_lambda": 5.0,
        },
    ]
    selected, tied = select_best(results)
    assert len(tied) == 2, "the 0.90 candidate is not tied and must be excluded"
    assert selected["max_depth"] == 3
    assert selected["min_child_samples"] == 80


def test_noise_tolerance_is_small_but_nonzero():
    assert 0 < NOISE_TOLERANCE <= 0.005


def test_capacity_rank_treats_num_leaves_as_inactive_under_a_binding_depth():
    """depth 3 caps a tree at 8 leaves, so 15 and 31 leaves are the same model."""
    a = {
        "num_leaves": 15,
        "max_depth": 3,
        "min_child_samples": 80,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.7,
    }
    b = dict(a, num_leaves=31)
    assert capacity_rank(a) == capacity_rank(b)


def test_model_cards_train_on_all_decisive_but_score_only_priced(cfg):
    """Training may use unpriced fights; scoring may not.

    Withholding unpriced fights from training would discard real signal, but scoring a
    model on fights the market cannot be scored on would make the comparison
    incommensurable. The card report must therefore use a wider training set than
    scoring set.
    """
    from ufc_predictor.models.cards import dev_frame as cards_dev_frame

    dev = cards_dev_frame(cfg)
    assert dev["label"].notna().all()
    assert dev["date"].max() <= cfg.windows.dev_end

    n_priced = int(dev["p1_market"].notna().sum())
    assert n_priced < len(dev), "the left join should retain unpriced decisive fights"

    cards = json.loads((cfg.paths.reports_dir / "model_cards.json").read_text())
    assert cards["train_rows_all_decisive"] > cards["scored_rows_priced_only"]
    # Every model scored the identical row count.
    scored = {m["dev_holdout"]["n"] for m in cards["models"].values()}
    assert len(scored) == 1, f"models were scored on different row counts: {scored}"
    assert scored.pop() == cards["scored_rows_priced_only"]


def test_model_cards_report_no_eval_window_numbers(cfg):
    """Nothing may publish an evaluation-window figure before the backtest runs."""
    cards = json.loads((cfg.paths.reports_dir / "model_cards.json").read_text())
    assert cards["holdout_span"][1] <= str(cfg.windows.dev_end)
    assert "eval" not in json.dumps(cards["models"]).lower()


def test_config_holds_the_tuned_and_frozen_parameters(cfg):
    """The backtest reads these from config and must never re-tune. Config must hold
    exactly what the dev-only search selected, as recorded in reports/tuning.json."""
    import json

    tuning = json.loads((cfg.paths.reports_dir / "tuning.json").read_text(encoding="utf-8"))
    selected = tuning["selected"]
    assert tuning["dev_window"][1] == str(cfg.windows.dev_end)
    p = cfg.lightgbm
    for key in ("max_depth", "num_leaves", "min_child_samples", "learning_rate", "reg_lambda"):
        assert p[key] == selected[key], f"config {key}={p[key]} but tuning selected {selected[key]}"
    assert p["colsample_bytree"] == selected["colsample_bytree"]
    # n_estimators is a cap for early stopping, not a fitted count.
    assert p["n_estimators"] >= 1000

    f = cfg.lightgbm_fit
    assert f["calibrate"] is True
    assert 0 < f["validation_fraction"] < 0.5
    assert f["early_stopping_rounds"] > 0


# ---------------------------------------------------------------------------
# The calibrator may never claim certainty
# ---------------------------------------------------------------------------


def _separable_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fights whose outcome is near-certain at both extremes of one feature.

    Isotonic regression fitted on such a tail ends in steps that hold only wins or only
    losses -- exactly the condition that made run 2's calibrator emit 0 and 1.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    frame = pd.DataFrame(0.0, index=range(n), columns=FEATURE_COLS)
    frame["elo_diff"] = x
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame["label"] = (rng.uniform(size=n) < np.clip(0.5 + frame["elo_diff"], 0, 1)).astype(int)
    frame["date"] = [dt.date(2000, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    return frame


def test_calibrated_lightgbm_never_outputs_a_certainty(cfg):
    """Regression test for the run-2 defect: an isotonic step holding only one class
    mapped to exactly 0 or 1, and each miss then cost 13.8 nats."""
    frame = _separable_frame()
    model = build_model("lightgbm", cfg)
    model.fit(frame, frame["label"])
    p = model.predict_proba(frame)
    near_certain = (p <= 1e-5) | (p >= 1 - 1e-5)
    assert not near_certain.any(), f"{int(near_certain.sum())} calibrated predictions at 0 or 1"


def test_calibrated_lightgbm_stays_inside_the_configured_bounds(cfg, split):
    lo, hi = cfg.lightgbm_fit["calibration_bounds"]
    assert 0 < lo < 0.5 < hi < 1
    train, test = split
    for frame, fit_on in ((_separable_frame(), None), (test, train)):
        model = build_model("lightgbm", cfg)
        source = frame if fit_on is None else fit_on
        model.fit(source, source["label"].astype(int))
        p = model.predict_proba(frame)
        assert p.min() >= lo and p.max() <= hi, (p.min(), p.max())
