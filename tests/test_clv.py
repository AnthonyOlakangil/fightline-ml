"""Closing-line value on the development window, checked on constructed data."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.analysis import clv
from ufc_predictor.config import get_config
from ufc_predictor.experiments.protocol import EvalWindowClosed


@pytest.fixture(scope="module")
def cfg():
    c = get_config()
    return dataclasses.replace(c, backtest={**c.backtest, "bootstrap_resamples": 1000})


def synthetic(n_events=150, per_event=10, slope=0.5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_events * per_event
    open_p = rng.uniform(0.2, 0.8, n)
    e = rng.normal(0, 0.06, n)  # the model's disagreement with the open
    move = slope * e + rng.normal(0, 0.02, n)  # the line moves toward the model, by construction
    close_p = np.clip(open_p + move, 0.02, 0.98)
    y = (rng.uniform(size=n) < close_p).astype(float)  # the close is the better-informed price
    start = dt.date(2010, 3, 1)
    return pd.DataFrame(
        {
            "fight_id": [f"f{i}" for i in range(n)],
            "event_id": np.repeat([f"e{j}" for j in range(n_events)], per_event),
            "date": np.repeat(
                [start + dt.timedelta(days=7 * j) for j in range(n_events)], per_event
            ),
            "label": y,
            "open_p1": open_p,
            "close_p1": close_p,
            "p_model": np.clip(open_p + e, 0.02, 0.98),
        }
    )


def test_recovers_a_constructed_closing_line_value(cfg):
    out = clv.analyse(synthetic(), cfg)
    corr = out["correlation_edge_vs_move"]
    assert corr["point"] > 0.6 and corr["ci_lower"] > 0.5
    directional = out["move_toward_model_when_edge_over_5pp"]
    assert directional["point"] == pytest.approx(0.5 * directional["mean_abs_edge"], rel=0.25)
    assert directional["ci_lower"] > 0
    assert out["model_information_beyond_open"]["coef_logit_model"]["ci_lower"] > 0


def test_no_constructed_value_means_no_detected_value(cfg):
    out = clv.analyse(synthetic(slope=0.0), cfg)
    corr = out["correlation_edge_vs_move"]
    assert corr["ci_lower"] < 0 < corr["ci_hi"]


def test_close_beats_open_when_the_close_is_better_informed(cfg):
    out = clv.analyse(synthetic(), cfg)
    sanity = out["close_minus_open_log_loss"]
    assert sanity["observed"] < 0


@pytest.mark.leakage
def test_any_evaluation_window_fight_raises(cfg):
    frame = synthetic()
    frame.loc[0, "date"] = dt.date(2016, 1, 9)
    with pytest.raises(EvalWindowClosed):
        clv.analyse(frame, cfg)


def test_missing_open_or_close_is_refused_not_filled(cfg):
    frame = synthetic()
    frame.loc[3, "open_p1"] = np.nan
    with pytest.raises(ValueError, match="open"):
        clv.analyse(frame, cfg)
