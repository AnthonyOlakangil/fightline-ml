"""The once-only 2015 confirmation of the frozen development configuration.

    uv run ufc-predictor confirm        # -> reports/dev/confirmation.json

Every Phase 8 model choice was made on 2010-2014. This scores the frozen configuration,
and the stage-0 baseline it is compared with, on every priced, decisive 2015 fight,
walk-forward, and reports each against the other and against the vig-free closing line.
No decision depends on the result: whatever it shows, the frozen configuration stands
(``config/experiments.yaml``).

It runs **once**. :func:`preflight` refuses if ``confirmation.json`` already exists, if
the working tree is not clean, if ``reports/dev/frozen.yaml`` is not committed, or if
HEAD has not been pushed -- so the frozen configuration is public before 2015 is scored.

Note what 2015 is not: the run-3 hyperparameters were tuned on a holdout that included
2015, and earlier diagnostics scored it. 2015 is unseen by every Phase 8 selection
decision, which is what this measures, not unseen by the project.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from ufc_predictor.backtest.metrics import per_fight_log_loss
from ufc_predictor.config import REPO_ROOT, Config, get_config
from ufc_predictor.experiments import runner
from ufc_predictor.experiments.runner import ExperimentSpec, run_config
from ufc_predictor.experiments.stages import load_spec, paired_test
from ufc_predictor.forward.ledger import first_commit_time
from ufc_predictor.provenance import OUTPUT_DIRS

FROZEN_RELPATH = "reports/dev/frozen.yaml"
CONFIRMATION_RELPATH = "reports/dev/confirmation.json"
PREDICTIONS_RELPATH = "reports/dev/confirmation_predictions.csv"


class ConfirmationError(RuntimeError):
    """The confirmation refused to run."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )


def preflight(root: Path = REPO_ROOT) -> str:
    """Every precondition for the single run; returns the HEAD commit.

    Raises:
        ConfirmationError: naming the first precondition that fails.
    """
    if (root / CONFIRMATION_RELPATH).exists():
        raise ConfirmationError(f"{CONFIRMATION_RELPATH} exists: the 2015 confirmation runs once")
    status = _git(root, "status", "--porcelain").stdout.splitlines()
    dirty = [line[3:] for line in status if line.strip() and not line[3:].startswith(OUTPUT_DIRS)]
    if dirty:
        raise ConfirmationError(f"the working tree is not clean ({dirty[:5]}); commit first")
    if not (root / FROZEN_RELPATH).is_file() or first_commit_time(root, FROZEN_RELPATH) is None:
        raise ConfirmationError(f"{FROZEN_RELPATH} is not committed; freeze before confirming")
    if _git(root, "diff", "--quiet", "HEAD", "--", FROZEN_RELPATH).returncode != 0:
        raise ConfirmationError(
            f"{FROZEN_RELPATH} has uncommitted edits; the frozen configuration is fixed"
        )
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "fetch", "--quiet", "origin")
    remote = _git(root, "branch", "-r", "--contains", head).stdout
    if "origin/" not in remote:
        raise ConfirmationError(
            f"HEAD {head[:7]} has not been pushed; push the frozen configuration first"
        )
    return head


def score(
    cfg: Config, frozen: ExperimentSpec, baseline: ExperimentSpec, panel: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Walk-forward over the confirmation year for both configurations, and compare."""
    pf = run_config(cfg, frozen, window="confirmation", panel=panel)
    pb = run_config(cfg, baseline, window="confirmation", panel=panel)
    if not pf["fight_id"].equals(pb["fight_id"]):
        raise RuntimeError("the frozen and baseline configurations scored different fights")
    y = pf["label"].to_numpy()
    loss = {
        "frozen": per_fight_log_loss(y, pf["p_model"].to_numpy()),
        "baseline": per_fight_log_loss(y, pb["p_model"].to_numpy()),
        "market": per_fight_log_loss(y, pf["p_market"].to_numpy()),
    }
    events = pf["event_id"].to_numpy()
    kw = {
        "n_resamples": int(cfg.backtest["bootstrap_resamples"]),
        "seed": cfg.seed,
        "ci": float(cfg.backtest["bootstrap_ci"]),
    }
    out = {
        "n_fights": len(pf),
        "n_events": int(pf["event_id"].nunique()),
        "date_span": [str(pf["date"].min()), str(pf["date"].max())],
        "log_loss": {k: float(v.mean()) for k, v in loss.items()},
        "frozen_minus_baseline": paired_test(loss["frozen"], loss["baseline"], events, **kw),
        "frozen_minus_market": paired_test(loss["frozen"], loss["market"], events, **kw),
        "baseline_minus_market": paired_test(loss["baseline"], loss["market"], events, **kw),
    }
    preds = pf.assign(p_baseline=pb["p_model"].to_numpy())
    return out, preds


def run(cfg: Config | None = None) -> dict:
    """Preflight, score 2015 once, and write the confirmation and its predictions."""
    cfg = cfg or get_config()
    head = preflight(REPO_ROOT)
    frozen_doc = yaml.safe_load((REPO_ROOT / FROZEN_RELPATH).read_text(encoding="utf-8"))
    frozen, baseline = load_spec(frozen_doc["spec"]), load_spec(frozen_doc["baseline_spec"])
    out, preds = score(cfg, frozen, baseline, runner.family_panel(cfg))
    record = {
        "git_commit": head,
        "frozen_spec": frozen.to_dict(),
        "baseline_spec": baseline.to_dict(),
        "note": "Scored once. No decision depends on it; the frozen configuration stands.",
        **out,
    }
    preds.to_csv(REPO_ROOT / PREDICTIONS_RELPATH, index=False)
    (REPO_ROOT / CONFIRMATION_RELPATH).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record
