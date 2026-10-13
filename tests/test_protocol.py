"""The development protocol: windows, the eval-closed guard, the experiment ledger.

The 2016+ evaluation window is closed. These tests pin the guard that keeps every
Phase 8 scoring path out of it, and the split of the development window into a
selection window (2010-2014) and a once-only confirmation year (2015).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess

import pytest
import yaml

from ufc_predictor.config import DEFAULT_CONFIG_PATH, ConfigError, get_config, load_config
from ufc_predictor.experiments import protocol
from ufc_predictor.experiments.protocol import (
    EvalWindowClosed,
    append_experiment,
    assert_scorable,
    load_preregistration,
)


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.mark.leakage
def test_scoring_any_eval_date_raises(cfg):
    with pytest.raises(EvalWindowClosed):
        assert_scorable([dt.date(2016, 1, 2)], cfg, purpose="selection")


@pytest.mark.leakage
@pytest.mark.parametrize("purpose", ["selection", "confirmation", "dev"])
def test_every_purpose_refuses_the_eval_window(cfg, purpose):
    dates = [dt.date(2015, 6, 1), cfg.windows.eval_start]
    with pytest.raises(EvalWindowClosed):
        assert_scorable(dates, cfg, purpose=purpose)


@pytest.mark.leakage
def test_selection_cannot_touch_the_confirmation_year(cfg):
    with pytest.raises(ValueError, match="selection"):
        assert_scorable([dt.date(2015, 3, 1)], cfg, purpose="selection")


def test_confirmation_scores_only_2015(cfg):
    assert_scorable([dt.date(2015, 1, 3), dt.date(2015, 12, 19)], cfg, purpose="confirmation")
    with pytest.raises(ValueError, match="confirmation"):
        assert_scorable([dt.date(2014, 12, 20)], cfg, purpose="confirmation")


def test_selection_window_accepts_its_own_dates(cfg):
    assert_scorable([dt.date(2010, 1, 2), dt.date(2014, 12, 31)], cfg, purpose="selection")


def test_nothing_to_score_raises(cfg):
    with pytest.raises(ValueError, match="no dates"):
        assert_scorable([], cfg, purpose="selection")


def test_windows_are_ordered_and_disjoint(cfg):
    w = cfg.windows
    assert (
        w.dev_start
        <= w.selection_end
        < w.confirmation_start
        <= w.confirmation_end
        == w.dev_end
        < w.eval_start
    )


@pytest.mark.leakage
def test_a_selection_window_reaching_the_confirmation_year_is_rejected(tmp_path):
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["windows"]["selection_end"] = "2015-06-30"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ConfigError, match="selection_end"):
        load_config(p)


def test_preregistration_names_every_stage_and_the_rule():
    pre = load_preregistration()
    assert pre["selection_window"] == ["2010-01-01", "2014-12-31"]
    assert pre["alpha"] == 0.10
    names = [s["name"] for s in pre["stages"]]
    assert names == [
        "stage_0_retune",
        "stage_1_features",
        "stage_2_selection",
        "stage_3_train_window",
        "stage_4_symmetry",
        "stage_5_calibration",
        "stage_6_model",
    ]
    assert "Holm" in pre["decision_rule"]


def test_append_experiment_is_append_only_and_records_provenance(tmp_path, cfg):
    ledger = tmp_path / "experiments.json"
    first = append_experiment(
        {"stage": "s", "candidate": "a", "metrics": {"log_loss": 0.69}}, cfg, path=ledger
    )
    second = append_experiment(
        {"stage": "s", "candidate": "b", "metrics": {"log_loss": 0.68}}, cfg, path=ledger
    )
    assert (first, second) == (1, 2)
    entries = json.loads(ledger.read_text(encoding="utf-8"))
    assert [e["candidate"] for e in entries] == ["a", "b"]
    for e in entries:
        assert len(e["git_commit"]) == 40
        assert isinstance(e["worktree_clean"], bool)
        assert len(e["config_hash"]) == 64
        assert len(e["preregistration_hash"]) == 64
        assert e["run"] in (1, 2)


def test_append_experiment_refuses_a_ledger_with_missing_runs(tmp_path, cfg):
    """Deleting an inconvenient run leaves a gap in the numbering, and that raises.

    (Git history of the committed ledger is the stronger record; this catches the edit
    before it is committed.)
    """
    ledger = tmp_path / "experiments.json"
    ledger.write_text(json.dumps([{"run": 1}, {"run": 3}]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger"):
        append_experiment({"stage": "s", "candidate": "c", "metrics": {}}, cfg, path=ledger)


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_a_run_before_or_without_the_preregistration_commit_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README").write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "before")
    with pytest.raises(RuntimeError, match="pre-registration"):
        protocol.assert_after_preregistration(repo)

    (repo / "config").mkdir()
    (repo / "config" / "experiments.yaml").write_text("alpha: 0.1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "prereg")
    protocol.assert_after_preregistration(repo)

    (repo / "config" / "experiments.yaml").write_text("alpha: 0.2\n")
    _git(repo, "commit", "-q", "-am", "moved the goalposts")
    with pytest.raises(RuntimeError, match="changed since"):
        protocol.assert_after_preregistration(repo)
