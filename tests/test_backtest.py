"""Backtest engine and metric tests.

The engine's correctness claim is narrow and checkable: for every scored fight, the
model that predicted it was fitted only on fights dated strictly earlier. That is
asserted here for *every* event, not only for the events where a refit happened --
between refits the model is stale, and staleness is easy to confuse with leakage if you
only check refit boundaries.

Metrics are checked against values computed by hand, because a plausible-looking but
wrong log loss would silently change the project's conclusion.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.backtest.metrics import (
    bootstrap_metric_ci,
    expected_calibration_error,
    paired_bootstrap_by_event,
    per_fight_log_loss,
    score,
)
from ufc_predictor.backtest.walkforward import VARIANT_NAME, load_panel
from ufc_predictor.config import get_config


@pytest.fixture(scope="module")
def cfg():
    return get_config()


# ---------------------------------------------------------------------------
# Metrics, hand-computed
# ---------------------------------------------------------------------------


def test_per_fight_log_loss_matches_hand_computation():
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    expected = np.array([-math.log(0.8), -math.log(0.7)])
    np.testing.assert_allclose(per_fight_log_loss(y, p), expected, rtol=1e-12)


def test_per_fight_log_loss_is_finite_at_the_endpoints():
    """Clipping must keep an overconfident, wrong prediction finite but severe."""
    v = per_fight_log_loss(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(v).all()
    assert v[0] > 10


def test_score_matches_hand_computation_on_a_small_fixture():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4])
    s = score(y, p)

    expected_ll = -np.mean([math.log(0.9), math.log(0.8), math.log(0.6), math.log(0.6)])
    expected_brier = np.mean([(0.9 - 1) ** 2, 0.2**2, (0.6 - 1) ** 2, 0.4**2])

    assert s["n"] == 4
    assert s["log_loss"] == pytest.approx(expected_ll, rel=1e-9)
    assert s["brier"] == pytest.approx(expected_brier, rel=1e-9)
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["base_rate"] == pytest.approx(0.5)
    assert s["auc"] == pytest.approx(1.0)


def test_score_rejects_an_empty_set():
    with pytest.raises(ValueError, match="empty set"):
        score(np.array([]), np.array([]))


def test_auc_is_nan_when_only_one_class_is_present():
    s = score(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))
    assert math.isnan(s["auc"])
    assert np.isfinite(s["log_loss"])


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    """Half the fights at p=0.5 won: predicted frequency equals observed frequency."""
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.5)
    ece, table = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert table["n"].sum() == 100


def test_ece_detects_systematic_overconfidence():
    """Predict 0.9 but win only half the time: the gap is 0.4."""
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.9)
    ece, _ = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_ece_bins_cover_the_unit_interval_and_handle_endpoints():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.0, 1.0, 0.5, 0.999999])
    ece, table = expected_calibration_error(y, p, n_bins=10)
    assert np.isfinite(ece)
    assert int(table["n"].sum()) == 4


# ---------------------------------------------------------------------------
# Paired bootstrap, resampled by event
# ---------------------------------------------------------------------------


def test_paired_bootstrap_detects_a_clear_model_advantage():
    rng = np.random.default_rng(0)
    n = 600
    events = np.repeat(np.arange(60), 10)
    loss_market = rng.normal(0.70, 0.05, n)
    loss_model = loss_market - 0.10  # uniformly better
    out = paired_bootstrap_by_event(loss_model, loss_market, events, n_resamples=2000, seed=1)
    assert out["observed"] == pytest.approx(-0.10, abs=1e-9)
    assert out["ci_hi"] < 0
    assert out["beats_market"] is True
    assert "beats the market" in out["interpretation"]


def test_paired_bootstrap_reports_no_difference_when_there_is_none():
    rng = np.random.default_rng(0)
    n = 600
    events = np.repeat(np.arange(60), 10)
    loss_market = rng.normal(0.70, 0.05, n)
    loss_model = loss_market + rng.normal(0.0, 0.05, n)
    out = paired_bootstrap_by_event(loss_model, loss_market, events, n_resamples=2000, seed=1)
    assert out["ci_lower"] <= 0 <= out["ci_hi"]
    assert out["beats_market"] is False
    assert "contains zero" in out["interpretation"]


def test_paired_bootstrap_reports_market_advantage_with_the_right_sign():
    events = np.repeat(np.arange(40), 10)
    loss_market = np.full(400, 0.60)
    loss_model = np.full(400, 0.70)
    out = paired_bootstrap_by_event(loss_model, loss_market, events, n_resamples=500, seed=1)
    assert out["observed"] == pytest.approx(0.10)
    assert out["ci_lower"] > 0
    assert "market beats the model" in out["interpretation"]


def test_event_resampling_gives_wider_intervals_than_fight_resampling():
    """The reason for resampling events: fights on a card are correlated.

    With a per-event shared offset, treating fights as independent understates the
    uncertainty. Event resampling must produce the wider, honest interval.
    """
    rng = np.random.default_rng(7)
    n_events, per_event = 50, 12
    events = np.repeat(np.arange(n_events), per_event)
    event_offset = rng.normal(0.0, 0.08, n_events)  # whole cards move together
    diff = np.repeat(event_offset, per_event) + rng.normal(0.0, 0.01, n_events * per_event)

    loss_market = np.full(len(diff), 0.65)
    loss_model = loss_market + diff

    by_event = paired_bootstrap_by_event(loss_model, loss_market, events, n_resamples=4000, seed=3)
    # Pretend every fight is its own event: that is fight-level resampling.
    by_fight = paired_bootstrap_by_event(
        loss_model, loss_market, np.arange(len(diff)), n_resamples=4000, seed=3
    )
    width_event = by_event["ci_hi"] - by_event["ci_lower"]
    width_fight = by_fight["ci_hi"] - by_fight["ci_lower"]
    assert width_event > 1.5 * width_fight, (
        f"event-resampled CI ({width_event:.4f}) should be materially wider than "
        f"fight-resampled ({width_fight:.4f}); otherwise the clustering is not handled"
    )


def test_paired_bootstrap_is_seeded():
    events = np.repeat(np.arange(30), 5)
    rng = np.random.default_rng(0)
    lm = rng.normal(0.7, 0.05, 150)
    a = paired_bootstrap_by_event(lm - 0.02, lm, events, n_resamples=500, seed=11)
    b = paired_bootstrap_by_event(lm - 0.02, lm, events, n_resamples=500, seed=11)
    assert a == b


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_by_event(np.zeros(5), np.zeros(4), np.arange(5), n_resamples=10, seed=0)


def test_bootstrap_metric_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.5).astype(float)
    p = np.full(500, 0.5)
    events = np.repeat(np.arange(50), 10)
    out = bootstrap_metric_ci(y, p, events, metric="log_loss", n_resamples=1000, seed=2)
    assert out["ci_lower"] <= out["point"] <= out["ci_hi"]
    assert out["point"] == pytest.approx(math.log(2), abs=1e-9)


def test_bootstrap_metric_ci_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unsupported metric"):
        bootstrap_metric_ci(
            np.zeros(3), np.zeros(3), np.arange(3), metric="auc", n_resamples=10, seed=0
        )


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel(cfg):
    try:
        return load_panel(cfg)
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data && make features`")


def test_panel_keeps_unpriced_fights_for_training(panel):
    """Training uses every decisive fight; only scoring needs a closing line."""
    assert (~panel["scorable"]).any(), "unpriced decisive fights should be retained"
    assert panel["label"].notna().all()


def test_panel_is_date_ordered(panel):
    assert panel["date"].is_monotonic_increasing


def test_panel_scorable_count_reconciles_to_6790(panel):
    assert int(panel["scorable"].sum()) == 6790


def test_panel_windows_reconcile_exactly(panel, cfg):
    """dev + gap + eval must account for every scorable fight, with nothing stranded."""
    s = panel.loc[panel["scorable"]]
    before = int((s["date"] < cfg.windows.dev_start).sum())
    dev = int(((s["date"] >= cfg.windows.dev_start) & (s["date"] <= cfg.windows.dev_end)).sum())
    gap = int(((s["date"] > cfg.windows.dev_end) & (s["date"] < cfg.windows.eval_start)).sum())
    ev = int(((s["date"] >= cfg.windows.eval_start) & (s["date"] <= cfg.windows.eval_end)).sum())
    after = int((s["date"] > cfg.windows.eval_end).sum())

    assert before == 0
    assert gap == 0
    assert after == 0
    assert dev == 2154
    assert ev == 4636
    assert before + dev + gap + ev + after == 6790


def test_every_scorable_fight_has_a_market_probability(panel):
    s = panel.loc[panel["scorable"]]
    assert s["p1_market"].notna().all()
    assert s["p1_market"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Walk-forward invariants, exercised on the dev window only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dev_run(cfg):
    """Run the engine over the dev window.

    Development and testing happen here so the evaluation period stays untouched until
    the single scored run.
    """
    from ufc_predictor.backtest.walkforward import run_walk_forward

    return run_walk_forward(cfg, through=cfg.windows.dev_end, verbose=False)


@pytest.mark.leakage
def test_training_cutoff_is_respected_for_every_scored_event(dev_run, panel):
    """The mandated assertion, checked per *event* rather than per fold.

    Between refits the model is stale, which is fine; what must never happen is a model
    predicting an event using data from that event or later.
    """
    predictions, folds = dev_run
    cutoffs = folds.set_index("fold")["train_date_max"].to_dict()
    starts = folds.set_index("fold")["cutoff_date"].to_dict()

    assert len(predictions) > 500
    for row in predictions.itertuples(index=False):
        assert cutoffs[row.fold] < row.date, (
            f"fight {row.fight_id} on {row.date} was predicted by fold {row.fold}, "
            f"whose training data reaches {cutoffs[row.fold]}"
        )
        assert starts[row.fold] <= row.date, "a fold predicted an event before its cutoff"


@pytest.mark.leakage
def test_no_fold_trains_on_or_after_its_own_cutoff(dev_run):
    _, folds = dev_run
    assert len(folds) > 1
    for row in folds.itertuples(index=False):
        assert row.train_date_max < row.cutoff_date


@pytest.mark.leakage
def test_no_fight_is_scored_twice(dev_run):
    predictions, _ = dev_run
    assert not predictions["fight_id"].duplicated().any()


@pytest.mark.leakage
def test_training_set_grows_monotonically_across_folds(dev_run):
    """A shrinking training set would mean the window is sliding rather than expanding,
    which is a different experiment than the one being reported."""
    _, folds = dev_run
    rows = folds.sort_values("fold")["train_rows"].to_numpy()
    assert (np.diff(rows) >= 0).all()


def test_retrain_cadence_matches_configuration(dev_run, cfg):
    predictions, folds = dev_run
    cadence = int(cfg.backtest["retrain_every_n_events"])
    events_scored = predictions["event_id"].nunique()
    expected_folds = math.ceil(events_scored / cadence)
    assert len(folds) == expected_folds, (
        f"{events_scored} events at cadence {cadence} implies {expected_folds} folds, "
        f"got {len(folds)}"
    )


def test_every_model_predicts_every_scored_fight(dev_run):
    predictions, _ = dev_run
    for col in ("p_market", "p_elo", "p_lightgbm"):
        assert predictions[col].notna().all(), f"{col} has gaps"
        assert predictions[col].between(0, 1).all()


def test_the_variant_abstains_before_it_has_enough_history(dev_run):
    """The stacker must return NaN during warm-up rather than silently echoing an input."""
    predictions, _ = dev_run
    col = f"p_{VARIANT_NAME}"
    assert col in predictions.columns
    assert predictions[col].isna().any(), "expected a warm-up period with no variant"
    assert predictions[col].notna().any(), "the variant never became available"
    # Once available it must stay available.
    available = predictions.sort_values("date")[col].notna().to_numpy()
    first_true = int(np.argmax(available))
    assert available[first_true:].all(), "the variant dropped out after warming up"


def test_all_models_are_scored_on_identical_fights(dev_run):
    """A like-for-like comparison requires the same rows, not merely the same count."""
    predictions, _ = dev_run
    base = set(predictions["fight_id"])
    for col in ("p_market", "p_elo", "p_lightgbm"):
        assert set(predictions.loc[predictions[col].notna(), "fight_id"]) == base


def test_walk_forward_is_deterministic(cfg):
    """Same seed, same numbers -- the project's determinism requirement."""
    from ufc_predictor.backtest.walkforward import run_walk_forward

    cut = dt.date(2012, 12, 31)
    a, _ = run_walk_forward(cfg, through=cut, verbose=False)
    b, _ = run_walk_forward(cfg, through=cut, verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_dev_run_never_reaches_the_eval_window(dev_run, cfg):
    predictions, folds = dev_run
    assert predictions["date"].max() <= cfg.windows.dev_end
    assert folds["cutoff_date"].max() <= cfg.windows.dev_end


def test_every_fold_early_stops_and_calibrates(dev_run, cfg):
    """No fold may quietly fit the full tree ceiling.

    Three early dev folds originally did: their validation tail fell below
    `min_validation_rows`, so they fitted 2,000 unvalidated, uncalibrated trees while
    being reported in the same column as properly validated folds. Every fold must now
    early-stop well short of the ceiling and carry a calibrator.
    """
    _, folds = dev_run
    ceiling = int(cfg.lightgbm["n_estimators"])
    assert (folds["trees"] < ceiling).all(), (
        f"folds reached the {ceiling}-tree ceiling: early stopping never fired for "
        f"{folds.loc[folds['trees'] >= ceiling, 'fold'].tolist()}"
    )
    assert folds["calibrated"].all(), "a fold was fitted without calibration"


def test_min_train_fights_guarantees_a_usable_validation_tail(cfg):
    """The two settings must be consistent, or folds start raising mid-backtest."""
    min_train = int(cfg.backtest["min_train_fights"])
    frac = float(cfg.lightgbm_fit["validation_fraction"])
    min_val = int(cfg.lightgbm_fit["min_validation_rows"])
    assert round(min_train * frac) >= min_val
    assert min_train - round(min_train * frac) >= min_val


def test_lightgbm_refuses_to_fit_without_a_chronological_holdout(cfg, panel):
    """Fail loudly rather than silently becoming a different, unregularized model."""
    from ufc_predictor.models.lgbm import LightGBMModel

    tiny = panel.head(400).sort_values(["date", "fight_id"])
    model = LightGBMModel(
        seed=cfg.seed, params=cfg.lightgbm, min_validation_rows=200, validation_fraction=0.2
    )
    with pytest.raises(ValueError, match="Refusing to fit without"):
        model.fit(tiny, tiny["label"].astype(int))
