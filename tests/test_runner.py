"""The development walk-forward runner: cutoffs, windows, same-date isolation."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments import runner
from ufc_predictor.experiments.protocol import EvalWindowClosed
from ufc_predictor.experiments.runner import ExperimentSpec, run_config

THROUGH = dt.date(2010, 9, 30)  # a short slice of the selection window keeps tests fast


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
def every_event(cfg):
    """Refit before every event, so a leak cannot hide between refits."""
    return dataclasses.replace(cfg, backtest={**cfg.backtest, "retrain_every_n_events": 1})


SHORT = dt.date(2010, 6, 30)


@pytest.fixture(scope="module")
def baseline(cfg, panel):
    return run_config(cfg, ExperimentSpec(), window="selection", panel=panel, through=THROUGH)


def test_runner_scores_only_priced_fights_inside_the_selection_window(cfg, baseline, panel):
    assert baseline["date"].min() >= cfg.windows.dev_start
    assert baseline["date"].max() <= THROUGH
    priced = panel.loc[
        panel["p1_market"].notna()
        & (panel["date"] >= cfg.windows.dev_start)
        & (panel["date"] <= THROUGH)
    ]
    assert set(baseline["fight_id"]) == set(priced["fight_id"])
    assert baseline["p_model"].between(0, 1).all() and baseline["p_market"].notna().all()


@pytest.mark.leakage
def test_every_fold_trains_strictly_before_the_events_it_predicts(baseline):
    assert (pd.to_datetime(baseline["train_max_date"]) < pd.to_datetime(baseline["date"])).all()


@pytest.mark.leakage
def test_a_poisoned_fight_on_a_predicted_date_changes_no_prediction_on_or_before_it(
    every_event, panel
):
    """Review 1, finding 5: insert fights dated on a predicted event's date, with a label
    pattern no model could ignore, and check nothing scored on or before that date moves."""
    cfg = every_event
    baseline = run_config(cfg, ExperimentSpec(), window="selection", panel=panel, through=SHORT)
    poison_date = sorted(baseline["date"].unique())[len(baseline["date"].unique()) // 2]
    template = panel.loc[panel["date"] < cfg.windows.dev_start].tail(400).copy()
    template["fight_id"] = [f"poison-{i}" for i in range(len(template))]
    template["event_id"] = "poison-event"
    template["date"] = poison_date
    template["p1_market"] = np.nan  # unpriced, so never scored itself
    template["label"] = (template["elo_diff"] < 0).astype(float)  # the opposite of reality
    poisoned = pd.concat([panel, template], ignore_index=True)
    again = run_config(cfg, ExperimentSpec(), window="selection", panel=poisoned, through=SHORT)
    keep = baseline["date"] <= poison_date
    a = baseline.loc[keep].set_index("fight_id")["p_model"]
    b = again.set_index("fight_id").loc[a.index, "p_model"]
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


@pytest.mark.leakage
def test_two_events_on_one_date_never_train_on_each_other(every_event, panel):
    """Split a real event in two (same date, different ids): predictions must not change."""
    cfg = every_event
    d = sorted(
        panel.loc[
            panel["p1_market"].notna() & (panel["date"] >= cfg.windows.dev_start), "date"
        ].unique()
    )[3]
    ev = panel.loc[panel["date"] == d, "event_id"].iloc[0]
    rows = panel.index[(panel["event_id"] == ev)]
    split = panel.copy()
    split.loc[rows[: len(rows) // 2], "event_id"] = f"{ev}-a"
    a = run_config(cfg, ExperimentSpec(), window="selection", panel=panel, through=d)
    b = run_config(cfg, ExperimentSpec(), window="selection", panel=split, through=d)
    pd.testing.assert_series_equal(
        a.set_index("fight_id")["p_model"].sort_index(),
        b.set_index("fight_id")["p_model"].sort_index(),
    )
    assert (b.loc[b["date"] == d, "train_max_date"] < d).all()


@pytest.mark.leakage
def test_the_runner_refuses_the_closed_evaluation_window(cfg, panel):
    with pytest.raises(EvalWindowClosed):
        runner.assert_window_scores(pd.DataFrame({"date": [dt.date(2016, 3, 5)]}), cfg, "selection")


def test_selection_never_returns_a_confirmation_year_fight(cfg, panel):
    out = run_config(
        cfg, ExperimentSpec(), window="selection", panel=panel, through=dt.date(2015, 2, 1)
    )
    assert out["date"].max() <= cfg.windows.selection_end


def test_train_window_floor_extends_back_to_the_minimum(cfg, panel):
    out = run_config(
        cfg,
        ExperimentSpec(train_window="last_8_years"),
        window="selection",
        panel=panel,
        through=THROUGH,
    )
    assert (out["train_rows"] >= int(cfg.backtest["min_train_fights"])).all()
    assert out["floor_extended"].any()  # only ~947 fights fall in the 8 years before 2010


def test_spec_validates_its_choices():
    with pytest.raises(ValueError, match="family"):
        ExperimentSpec(features=("base", "astrology"))
    with pytest.raises(ValueError, match="train_window"):
        ExperimentSpec(train_window="last_3_days")
    with pytest.raises(ValueError, match="calibration"):
        ExperimentSpec(calibration="beta")


def test_symmetric_and_logistic_specs_run(cfg, panel):
    for spec in (ExperimentSpec(symmetric=True), ExperimentSpec(model="logistic")):
        out = run_config(cfg, spec, window="selection", panel=panel, through=dt.date(2010, 5, 31))
        assert len(out) > 0 and out["p_model"].between(0, 1).all()
