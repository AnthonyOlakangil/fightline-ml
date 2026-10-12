"""Tests for metrics.json and the generated README block.

The project's credibility rests on every published figure being traceable to
``reports/metrics.json``. These tests re-render the README block from the JSON and
assert it matches the committed file, so a stale or hand-edited README fails the suite
rather than quietly misreporting the result.
"""

from __future__ import annotations

import json
import math

import pytest

from ufc_predictor.backtest.readme_block import END, START, render
from ufc_predictor.backtest.walkforward import VARIANT_NAME
from ufc_predictor.config import get_config

REPO_ROOT = get_config().paths.raw_dir.parent.parent


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def metrics(cfg):
    path = cfg.paths.reports_dir / "metrics.json"
    if not path.is_file():
        pytest.skip("metrics.json absent; run `make backtest`")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# metrics.json structure and internal consistency
# ---------------------------------------------------------------------------


def test_metrics_records_the_eval_run_count(metrics):
    """The number of times the held-out window has been scored must be on record."""
    assert isinstance(metrics["eval_run_count"], int)
    assert metrics["eval_run_count"] >= 1


def test_eval_ledger_length_matches_the_recorded_count(cfg, metrics):
    ledger = json.loads((cfg.paths.reports_dir / "eval_runs.json").read_text(encoding="utf-8"))
    assert len(ledger["runs"]) == metrics["eval_run_count"]
    for entry in ledger["runs"]:
        assert entry["git_sha"]
        assert entry["timestamp_utc"]


def test_reconciliation_adds_up(metrics):
    """dev + gap + eval must equal the scored total, and nothing may be stranded."""
    r = metrics["reconciliation"]
    assert r["dev"] + r["gap"] + r["eval"] == r["scored_total"]
    assert r["scored_total"] + r["unscored_warmup"] == r["in_range_for_this_run"]
    assert r["total_priced_decisive"] == 6790
    assert r["dev"] == 2154
    assert r["eval"] == 4636
    assert r["gap"] == 0
    assert r["unscored_warmup"] == 0


def test_hyperparameters_in_metrics_match_the_config_at_the_ledgered_commit(metrics):
    """Proof the settings that produced the result were committed *before* it existed.

    The scored run names the commit it ran from; that commit, which is in the pushed
    history, must hold exactly the hyperparameters and fitting regime the run reports.
    Comparing against the commit rather than the working copy is deliberate: the
    re-tuned settings are committed and pushed before the evaluation window is scored,
    so for that one commit the config is ahead of metrics.json by design.
    """
    import subprocess

    import yaml

    assert metrics["walk_forward"]["hyperparameters_frozen_from_dev"] is True
    if "git_commit" not in metrics:
        pytest.skip("evaluation run 1 predates full-commit provenance; disclosed in the README")
    commit = metrics["git_commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=True, cwd=REPO_ROOT
    )
    at_commit = yaml.safe_load(
        subprocess.check_output(
            ["git", "show", f"{commit}:config/config.yaml"], text=True, cwd=REPO_ROOT
        )
    )
    assert metrics["walk_forward"]["lightgbm"] == at_commit["lightgbm"]
    assert metrics["walk_forward"]["lightgbm_fit"] == at_commit["lightgbm_fit"]


def test_every_model_scored_the_same_eval_fights(metrics):
    """A like-for-like comparison needs one denominator."""
    ev = metrics["results"]["eval"]
    assert ev["market"]["n"] == 4636
    for name in ("elo", "lightgbm"):
        assert ev[name]["n"] == ev["market"]["n"]
        assert ev[name]["covers_all_fights"] is True


def test_bootstrap_resamples_events_not_fights(metrics):
    assert metrics["bootstrap"]["resampling_unit"] == "event"
    assert metrics["bootstrap"]["n_resamples"] >= 10000
    for p in metrics["paired_vs_market"]["eval"].values():
        # 421 eval events, far fewer than 4,636 fights -- the honest unit of independence.
        assert p["n_events_resampled"] == 421
        assert p["n_events_resampled"] < p["n_fights"]


def test_paired_intervals_bracket_their_point_estimates(metrics):
    for window in ("dev", "eval"):
        for name, p in metrics["paired_vs_market"][window].items():
            assert p["ci_lower"] <= p["observed"] <= p["ci_hi"], f"{window}/{name}"


def test_metric_values_are_in_valid_ranges(metrics):
    for window in ("dev", "eval"):
        for name, s in metrics["results"][window].items():
            assert 0 < s["log_loss"] < 3, f"{window}/{name}"
            assert 0 <= s["brier"] <= 1
            assert 0 <= s["accuracy"] <= 1
            assert 0 <= s["ece"] <= 1
            assert math.isnan(s["auc"]) or 0 <= s["auc"] <= 1


def test_no_model_is_wired_backwards(metrics):
    """An inverted label or probability shows up as AUC below 0.5.

    Log loss above ln 2 is *not* that signal: an overconfident model that still ranks
    fights the right way round can score worse than a coin flip. That is a result to
    disclose (see the next test), not a wiring bug.
    """
    for window in ("dev", "eval"):
        for name, s in metrics["results"][window].items():
            assert s["auc"] > 0.5, f"{window}/{name} has AUC {s['auc']:.4f}: wired backwards?"


def test_readme_discloses_every_model_worse_than_a_coin_flip(metrics):
    block = render(metrics)
    for window in ("dev", "eval"):
        for name, s in metrics["results"][window].items():
            if s["log_loss"] >= math.log(2):
                assert f"worse than a coin flip in the {window} window" in block, (window, name)


def test_readme_qualifies_a_variant_win_with_multiplicity_and_the_dev_result(metrics):
    """A narrow interval excluding zero is one of three comparisons; the generated text
    must say what it looks like after correcting for that, and what dev showed."""
    var = metrics["paired_vs_market"]["eval"][VARIANT_NAME]
    if not var["beats_market"]:
        pytest.skip("variant does not beat the market in this run")
    block = render(metrics)
    n = len(metrics["paired_vs_market"]["eval"])
    adjusted = min(1.0, var["p_value_two_sided"] * n)
    assert f"Bonferroni-adjusted across the {n} comparisons, p = {adjusted:.3f}" in block
    dev = metrics["paired_vs_market"]["dev"][VARIANT_NAME]
    assert f"{dev['observed']:+.4f}" in block


def test_by_year_rows_sum_to_the_scored_total(metrics):
    total = sum(row["n"] for row in metrics["by_year"])
    assert total == metrics["reconciliation"]["scored_total"]


def test_by_year_covers_every_year_in_range(metrics):
    years = [row["year"] for row in metrics["by_year"]]
    assert years == sorted(years)
    assert years[0] == 2010
    assert len(years) == len(set(years))


def test_variant_is_labelled_as_market_consuming(metrics):
    """The variant must never be presented as beating the market on its own."""
    note = metrics["variant_note"].lower()
    assert "closing line" in note
    assert "cannot support" in note
    assert VARIANT_NAME in metrics["results"]["eval"]


# ---------------------------------------------------------------------------
# The README block
# ---------------------------------------------------------------------------


def test_readme_has_the_result_markers(readme):
    assert START in readme
    assert END in readme


def test_readme_block_matches_metrics_json(metrics, readme):
    """The decisive test: the README is a rendering of metrics.json, not prose.

    If anyone edits a number in the README by hand, or regenerates metrics without
    refreshing the README, this fails.
    """
    expected = render(metrics)
    start = readme.index(START)
    end = readme.index(END) + len(END)
    assert readme[start:end] == expected, (
        "the README results block is out of date with reports/metrics.json; run `make report`"
    )


def test_readme_timestamp_has_a_single_utc_designator(metrics):
    """generated_at_utc already ends in +00:00; appending Z as well is malformed."""
    block = render(metrics)
    assert metrics["generated_at_utc"] in block
    assert "+00:00Z" not in block


def test_readme_states_the_eval_run_count(metrics, readme):
    assert f"Evaluation runs to date: **{metrics['eval_run_count']}**" in readme


def test_readme_reports_the_headline_log_losses(metrics, readme):
    ev = metrics["results"]["eval"]
    for name in ("market", "lightgbm", "elo"):
        assert f"{ev[name]['log_loss']:.4f}" in readme, f"{name} log loss missing from README"


def test_readme_discloses_the_verdict_against_the_market(metrics, readme):
    lgbm = metrics["paired_vs_market"]["eval"]["lightgbm"]
    if not lgbm["beats_market"] and lgbm["ci_lower"] > 0:
        assert "does not beat the market" in readme, (
            "the model loses to the market; the README must say so plainly"
        )


def test_readme_discloses_the_eval_window_glimpses(readme):
    """Honesty about how often the held-out window has been looked at."""
    lowered = readme.lower()
    assert "eval-window glimpse" in lowered or "glimpses of the evaluation window" in lowered
    assert "0.6489" in readme, (
        "the accidental Phase 2 eval-window measurement should be disclosed by value"
    )


def test_readme_states_the_exact_eval_n(metrics, readme):
    n = metrics["results"]["eval"]["market"]["n"]
    assert f"{n:,} fights" in readme


def test_readme_no_longer_claims_to_be_metric_free(readme):
    assert "No metrics appear below yet" not in readme


# ---------------------------------------------------------------------------
# Eval-ledger provenance: the settings must be committed before the window is scored
# ---------------------------------------------------------------------------


def _cfg_with_reports(cfg, path):
    import dataclasses

    return dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, reports_dir=path))


def test_ledger_entry_records_the_full_commit_a_clean_tree_and_a_reason(cfg, tmp_path):
    """A short SHA stamped on a dirty tree proves nothing about which code produced the
    numbers. Each entry carries the full commit, whether the tree matched it, and why the
    held-out window was scored."""
    from ufc_predictor.backtest import report

    git = {"commit": "a" * 40, "short": "a" * 7, "clean": True}
    count = report._append_eval_ledger(
        _cfg_with_reports(cfg, tmp_path), {"market": 0.6}, git=git, reason="why"
    )
    entry = json.loads((tmp_path / "eval_runs.json").read_text(encoding="utf-8"))["runs"][-1]
    assert count == 1
    assert entry["git_commit"] == "a" * 40
    assert entry["git_sha"] == "a" * 7
    assert entry["worktree_clean"] is True
    assert entry["reason"] == "why"


def test_scoring_the_eval_window_from_a_dirty_tree_is_refused(cfg, tmp_path, monkeypatch):
    """Refused before the walk-forward starts, so nothing is scored or written."""
    from ufc_predictor.backtest import report

    monkeypatch.setattr(
        report, "_git_state", lambda: {"commit": "b" * 40, "short": "b" * 7, "clean": False}
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("the walk-forward ran from a dirty tree")

    monkeypatch.setattr(report, "run_walk_forward", must_not_run)
    with pytest.raises(RuntimeError, match="uncommitted"):
        report.run(_cfg_with_reports(cfg, tmp_path), verbose=False, reason="test")
    assert not (tmp_path / "eval_runs.json").exists()


def test_scoring_the_eval_window_requires_a_reason(cfg, tmp_path, monkeypatch):
    from ufc_predictor.backtest import report

    monkeypatch.setattr(
        report, "_git_state", lambda: {"commit": "c" * 40, "short": "c" * 7, "clean": True}
    )
    monkeypatch.setattr(
        report, "run_walk_forward", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    with pytest.raises(ValueError, match="reason"):
        report.run(_cfg_with_reports(cfg, tmp_path), verbose=False, reason="  ")


def test_every_ledger_entry_after_the_first_is_fully_provenanced(cfg):
    """Run 1 predates this check and is reported as such in the README. Every later run
    must name its full commit, have been scored from a clean tree, and say why."""
    path = cfg.paths.reports_dir / "eval_runs.json"
    if not path.is_file():
        pytest.skip("no eval ledger")
    for entry in json.loads(path.read_text(encoding="utf-8"))["runs"][1:]:
        assert len(entry["git_commit"]) == 40, entry
        assert entry["worktree_clean"] is True, entry
        assert entry["reason"].strip(), entry
