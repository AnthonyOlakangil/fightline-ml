"""The pre-registered stages: Holm correction, the decision rule, candidate specs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import numpy as np
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments import runner, stages
from ufc_predictor.experiments.runner import ExperimentSpec


def test_holm_adjusts_step_down_and_stays_monotone():
    adj = stages.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted: a 0.01*3=0.03, c 0.03*2=0.06, b 0.04*1=0.04 -> monotone max -> 0.06
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert stages.holm({"only": 0.07}) == {"only": 0.07}
    assert stages.holm({"x": 0.6, "y": 0.5})["x"] == 1.0  # capped at 1


@pytest.mark.parametrize(
    ("results", "winner"),
    [
        # one clear improvement passes
        ([{"name": "a", "observed": -0.004, "p_one_sided": 0.01, "log_loss": 0.650}], "a"),
        # an improvement whose p is too large after Holm does not
        (
            [
                {"name": "a", "observed": -0.004, "p_one_sided": 0.06, "log_loss": 0.650},
                {"name": "b", "observed": -0.001, "p_one_sided": 0.40, "log_loss": 0.653},
            ],
            None,  # Holm: 0.06*2 = 0.12 >= 0.10
        ),
        # a "significant" result in the wrong direction is never adopted
        ([{"name": "a", "observed": 0.004, "p_one_sided": 0.001, "log_loss": 0.660}], None),
        # two pass: the lower log loss wins
        (
            [
                {"name": "a", "observed": -0.004, "p_one_sided": 0.001, "log_loss": 0.6500},
                {"name": "b", "observed": -0.006, "p_one_sided": 0.002, "log_loss": 0.6480},
            ],
            "b",
        ),
    ],
)
def test_decide_applies_the_preregistered_rule(results, winner):
    decided, rows = stages.decide(results, alpha=0.10)
    assert decided == winner
    assert [r["adopted"] for r in rows] == [r["name"] == winner for r in rows]
    assert all("p_holm" in r for r in rows)


def test_paired_test_is_one_sided_and_event_resampled():
    rng = np.random.default_rng(0)
    events = np.repeat(np.arange(100), 10)
    inc = rng.uniform(0.5, 0.8, 1000)
    better = inc - 0.02
    t = stages.paired_test(better, inc, events, n_resamples=2000, seed=1)
    assert t["observed"] == pytest.approx(-0.02)
    assert t["p_one_sided"] == 0.0 and t["ci_hi"] < 0
    worse = stages.paired_test(inc + 0.02, inc, events, n_resamples=2000, seed=1)
    assert worse["p_one_sided"] == 1.0
    assert t["n_events"] == 100


def test_retune_holdout_is_the_tail_of_2010_2014_and_never_2015():
    cfg = get_config()
    panel = runner.base_panel(cfg)
    train, val = stages.retune_split(panel, cfg)
    assert val["date"].max() <= cfg.windows.selection_end
    assert train["date"].max() < val["date"].min()
    sel = panel.loc[
        (panel["date"] >= cfg.windows.dev_start) & (panel["date"] <= cfg.windows.selection_end)
    ]
    assert abs(len(val) - round(0.2 * len(sel))) <= 15  # boundary-date ties move into the holdout
    assert val["date"].min() > cfg.windows.dev_start


def test_compare_stage_on_a_short_window_writes_every_candidate(tmp_path):
    cfg = get_config()
    cfg = dataclasses.replace(cfg, backtest={**cfg.backtest, "bootstrap_resamples": 500})
    panel = runner.base_panel(cfg)
    stage = {
        "name": "stage_5_calibration",
        "kind": "compare",
        "field": "calibration",
        "incumbent": "isotonic_bounded",
        "candidates": ["platt", "none"],
    }
    record, winner = stages.run_compare_stage(
        stage,
        ExperimentSpec(),
        panel,
        cfg,
        through=dt.date(2010, 8, 31),
        ledger=tmp_path / "x.json",
        alpha=0.10,
    )
    names = [c["name"] for c in record["candidates"]]
    assert names == ["platt", "none"]
    assert all(
        {"observed", "ci_lower", "ci_hi", "p_one_sided", "p_holm", "adopted", "log_loss"}
        <= c.keys()
        for c in record["candidates"]
    )
    assert record["incumbent"]["log_loss"] > 0
    ledger = json.loads((tmp_path / "x.json").read_text())
    assert len(ledger) == 3  # incumbent + two candidates
    assert isinstance(winner, ExperimentSpec)
