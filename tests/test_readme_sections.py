"""Every generated README section must equal a fresh rendering of its JSON source.

Same contract as the results block in ``test_report.py``: a hand-edited or stale number
fails the suite. Also pins the provenance of the inputs -- the run-1 archive must be the
run the ledger recorded, and the diagnostics must stay inside the dev window.
"""

from __future__ import annotations

import json

import pytest

from ufc_predictor.backtest import readme_sections
from ufc_predictor.config import get_config

REPO_ROOT = get_config().paths.raw_dir.parent.parent
SECTIONS = (
    "HEADLINE",
    "DIAGNOSTICS",
    "TUNING",
    "DEVCARDS",
    "LEAK",
    "POSTHOC",
    "RUNS",
    "CALBOUND",
    "P8_STAGES",
    "P8_CONFIRM",
    "P8_CLV",
    "P8_FORWARD",
)


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def rendered(cfg):
    if not (cfg.paths.reports_dir / "metrics.json").is_file():
        pytest.skip("metrics.json absent; run `make backtest`")
    return readme_sections.render_all(cfg)


@pytest.fixture(scope="module")
def readme():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _load(cfg, *parts):
    return json.loads(cfg.paths.reports_dir.joinpath(*parts).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SECTIONS)
def test_readme_section_matches_its_source(name, rendered, readme):
    assert readme_sections.extract(readme, name) == rendered[name], (
        f"README section {name} is out of date with reports/; run `make report`"
    )


def test_archived_run_1_is_the_run_the_ledger_recorded(cfg):
    """The 'before' column of the leak comparison must be the real run 1, not a rerun."""
    archived = _load(cfg, readme_sections.RUN_1_ARCHIVE, "metrics.json")
    first = _load(cfg, "eval_runs.json")["runs"][0]
    assert archived["eval_run_count"] == first["run"] == 1
    assert archived["git_sha"] == first["git_sha"]
    headline = {k: round(v["log_loss"], 4) for k, v in archived["results"]["eval"].items()}
    assert headline == first["headline"]


def test_both_runs_scored_the_same_eval_fights(cfg):
    archived = _load(cfg, readme_sections.RUN_1_ARCHIVE, "metrics.json")
    current = _load(cfg, "metrics.json")
    assert archived["reconciliation"] == current["reconciliation"]
    assert archived["windows"] == current["windows"]


def test_diagnostics_never_reach_the_eval_window(cfg):
    d = _load(cfg, "leakage_diagnostics.json")
    dev_end = str(cfg.windows.dev_end)
    assert d["split"]["test_span"][1] <= dev_end
    assert d["attribute_leak"]["window"][1] <= dev_end


def test_attribute_leak_evidence_reproduces_from_the_data(cfg):
    """The README's evidence table is recomputable from the committed snapshot."""
    from ufc_predictor.features import diagnostics
    from ufc_predictor.storage import db

    try:
        tables = {name: db.read_table(cfg.paths.processed_dir, name) for name in db.TABLES}
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data`")
    feats = diagnostics.build(tables["fights"], tables, cfg)
    fresh = diagnostics.attribute_leak_evidence(tables, feats, cfg)
    assert fresh == _load(cfg, "leakage_diagnostics.json")["attribute_leak"]


def test_post_hoc_checks_reproduce_from_the_stored_predictions(cfg):
    """post_hoc.json is a pure function of the committed predictions CSV."""
    from ufc_predictor.backtest import post_hoc

    committed = _load(cfg, "post_hoc.json")
    fresh = json.loads(json.dumps(post_hoc.compute(cfg)))
    # Logistic fits and BLAS differ across platforms in the last few bits, so the CI
    # runner (Linux) cannot reproduce a macOS-written file bit-for-bit; agreement to
    # 1e-6 relative is far tighter than any figure the README prints.
    _assert_close(fresh, committed)
    assert committed["variant_decomposition"]["replay_matches_stored_variant"] is True


def _assert_close(a, b, path="post_hoc"):
    import math

    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_calibration_bound_evidence_is_dev_only_and_reproduces(cfg):
    from ufc_predictor.features import diagnostics

    committed = _load(cfg, "leakage_diagnostics.json")["calibration_bound"]
    assert committed["window"][1] <= str(cfg.windows.dev_end)
    assert committed["bounds"] == list(cfg.lightgbm_fit["calibration_bounds"])
    _assert_close(diagnostics.calibration_bound_evidence(cfg), committed, "calibration_bound")


@pytest.mark.parametrize("index", [0, 1])
def test_every_archived_run_is_the_run_the_ledger_recorded(cfg, index):
    """Each earlier column of the side-by-side table must be that run, not a rerun."""
    archive = readme_sections.RUN_ARCHIVES[index]
    archived = _load(cfg, archive, "metrics.json")
    entry = _load(cfg, "eval_runs.json")["runs"][index]
    assert archived["eval_run_count"] == entry["run"] == index + 1
    assert archived["git_sha"] == entry["git_sha"]
    headline = {k: round(v["log_loss"], 4) for k, v in archived["results"]["eval"].items()}
    assert headline == entry["headline"]
    assert archived["reconciliation"] == _load(cfg, "metrics.json")["reconciliation"]


def test_stored_lightgbm_predictions_respect_the_calibration_bound(cfg):
    """The committed run's predictions, not just a fixture, must stay inside the bound."""
    import pandas as pd

    preds = pd.read_csv(cfg.paths.reports_dir / "walkforward_predictions.csv")
    lo, hi = cfg.lightgbm_fit["calibration_bounds"]
    assert preds["p_lightgbm"].between(lo, hi).all()
