"""The 2015 confirmation: runs once, from a clean pushed commit, after the freeze.

No test here scores a real 2015 outcome. The scoring test builds a synthetic "2015" from
2014 rows with shuffled labels, so the confirmation year stays unseen until the one run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import subprocess

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments import confirm, runner
from ufc_predictor.experiments.runner import ExperimentSpec


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "remote", "add", "origin", str(remote))
    (r / "reports" / "dev").mkdir(parents=True)
    (r / "reports" / "dev" / "frozen.yaml").write_text("spec: {}\n")
    (r / "code.py").write_text("x = 1\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "freeze")
    _git(r, "push", "-q", "origin", "main")
    return r


def test_preflight_passes_on_a_clean_pushed_frozen_commit(repo):
    assert len(confirm.preflight(repo)) == 40


def test_a_second_confirmation_is_refused(repo):
    (repo / "reports" / "dev" / "confirmation.json").write_text("{}")
    with pytest.raises(confirm.ConfirmationError, match="once"):
        confirm.preflight(repo)


def test_a_dirty_tree_is_refused(repo):
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(confirm.ConfirmationError, match="clean"):
        confirm.preflight(repo)


def test_an_unpushed_commit_is_refused(repo):
    (repo / "code.py").write_text("x = 3\n")
    _git(repo, "commit", "-q", "-am", "local only")
    with pytest.raises(confirm.ConfirmationError, match="pushed"):
        confirm.preflight(repo)


def test_an_uncommitted_freeze_is_refused(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "code.py").write_text("x = 1\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "c")
    with pytest.raises(confirm.ConfirmationError, match="frozen"):
        confirm.preflight(r)


@pytest.fixture(scope="module")
def synthetic_2015():
    """Real fights through 2014; a fake 2015 made of 2014 bouts with shuffled labels."""
    cfg = get_config()
    panel = runner.base_panel(cfg)
    real = panel.loc[panel["date"] <= cfg.windows.selection_end]
    fake = real.loc[(real["date"] >= dt.date(2014, 1, 1)) & real["p1_market"].notna()].copy()
    fake = fake.loc[fake["date"] < dt.date(2014, 4, 1)]
    fake["date"] = fake["date"].map(lambda d: d.replace(year=2015))
    fake["fight_id"] = "fake-" + fake["fight_id"]
    fake["event_id"] = "fake-" + fake["event_id"]
    fake["label"] = np.random.default_rng(0).permutation(fake["label"].to_numpy())
    return cfg, pd.concat([real, fake], ignore_index=True)


def test_scoring_covers_only_2015_and_compares_both_configurations(synthetic_2015):
    cfg, panel = synthetic_2015
    cfg = dataclasses.replace(cfg, backtest={**cfg.backtest, "bootstrap_resamples": 500})
    out, preds = confirm.score(cfg, ExperimentSpec(calibration="platt"), ExperimentSpec(), panel)
    assert preds["date"].min() >= dt.date(2015, 1, 1) and preds["date"].max() <= dt.date(
        2015, 12, 31
    )
    assert set(preds["fight_id"]).issubset({f for f in panel["fight_id"] if f.startswith("fake-")})
    assert {"frozen_minus_baseline", "frozen_minus_market", "baseline_minus_market"} <= out.keys()
    for k in ("frozen_minus_baseline", "frozen_minus_market"):
        assert out[k]["ci_lower"] <= out[k]["observed"] <= out[k]["ci_hi"]
    assert out["n_fights"] == len(preds) > 100
