"""The pre-registered development stages and their decision rule.

    uv run ufc-predictor experiments     # stages 0-6 + final re-tune -> reports/dev/

Everything here follows ``config/experiments.yaml``, committed before any experiment
ran; :func:`run_all` refuses to start from a commit that lacks it or has changed it.

* **Stage 0** re-tunes the run-3 baseline inside 2010-2014 only (the run-3 hyperparameters
  were tuned on a holdout that included 2015), with the same grid and tie rule as
  ``models/tune.py``.
* **Stages 1-6** each compare candidates against the stage incumbent on the selection
  window: paired per-fight log-loss difference, event bootstrap, one-sided p-value,
  Holm-adjusted within the stage. A candidate is adopted only if its difference is
  negative and its adjusted p is below ``alpha``; if several pass, the lowest log loss.
* **Final re-tune** repeats stage 0 for the frozen configuration and adopts the new
  parameters only under the same rule.

Every walk-forward run is appended to ``reports/dev/experiments.json``; every candidate,
adopted or not, is written to ``reports/dev/stages.json``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

from ufc_predictor.backtest.metrics import per_fight_log_loss
from ufc_predictor.config import REPO_ROOT, Config, get_config
from ufc_predictor.experiments import protocol, runner
from ufc_predictor.experiments.runner import ExperimentSpec, make_model, run_config
from ufc_predictor.features.pit import FEATURE_FAMILIES
from ufc_predictor.models.tune import SEARCH_SPACE, select_best
from ufc_predictor.provenance import OUTPUT_DIRS, git_state

#: Retune holdout share of the labelled selection-window fights (pre-registered).
RETUNE_HOLDOUT = 0.2


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_test(
    loss_candidate: np.ndarray,
    loss_incumbent: np.ndarray,
    event_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Mean(candidate - incumbent) with an event bootstrap and a one-sided p-value.

    The p-value is the share of resampled mean differences that are >= 0: small when the
    candidate is reliably better.
    """
    diff = np.asarray(loss_candidate, float) - np.asarray(loss_incumbent, float)
    frame = pd.DataFrame({"e": np.asarray(event_ids), "d": diff})
    g = frame.groupby("e")["d"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_resamples, len(sums)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "observed": float(diff.mean()),
        "ci_lower": float(lo),
        "ci_hi": float(hi),
        "ci_level": ci,
        "p_one_sided": float((boot >= 0).mean()),
        "n_fights": len(diff),
        "n_events": len(sums),
        "n_resamples": int(n_resamples),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def decide(results: list[dict], *, alpha: float) -> tuple[str | None, list[dict]]:
    """Apply the pre-registered rule to one stage's candidates.

    Each result needs ``name``, ``observed``, ``p_one_sided`` and ``log_loss``. Returns
    the adopted candidate's name (or None) and the results with ``p_holm`` and
    ``adopted`` filled in.
    """
    adjusted = holm({r["name"]: r["p_one_sided"] for r in results})
    passing = [r for r in results if r["observed"] < 0 and adjusted[r["name"]] < alpha]
    winner = min(passing, key=lambda r: (r["log_loss"], r["name"]))["name"] if passing else None
    rows = [{**r, "p_holm": adjusted[r["name"]], "adopted": r["name"] == winner} for r in results]
    return winner, rows


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _feature_candidate(name: str) -> tuple[str, ...]:
    if name == "all":
        return tuple(FEATURE_FAMILIES)
    parts = tuple(name.split("+"))
    unknown = [p for p in parts if p not in FEATURE_FAMILIES]
    if unknown:
        raise ValueError(
            f"pre-registered feature candidate {name!r} names unknown families {unknown}"
        )
    return parts


def candidate_specs(stage: dict, incumbent: ExperimentSpec) -> dict[str, ExperimentSpec]:
    """The stage's candidates as specs: the incumbent with one field changed."""
    field = stage["field"]
    out: dict[str, ExperimentSpec] = {}
    for value in stage["candidates"]:
        if field == "features":
            out[value] = dataclasses.replace(incumbent, features=_feature_candidate(value))
        elif field == "top_k":
            out[f"top_k_{value}"] = dataclasses.replace(incumbent, top_k=int(value))
        elif field == "symmetric":
            out["symmetric" if value else "asymmetric"] = dataclasses.replace(
                incumbent, symmetric=bool(value)
            )
        elif field in ("train_window", "calibration", "model"):
            out[str(value)] = dataclasses.replace(incumbent, **{field: value})
        else:
            raise ValueError(f"unknown pre-registered stage field {field!r}")
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class _Cache:
    """Walk-forward results by spec, so an incumbent is run once per invocation."""

    def __init__(self) -> None:
        self.runs: dict[str, pd.DataFrame] = {}


def _scored(
    spec: ExperimentSpec,
    panel: pd.DataFrame,
    cfg: Config,
    *,
    stage: str,
    candidate: str,
    ledger: Path | None,
    through: dt.date | None,
    cache: _Cache,
) -> pd.DataFrame:
    key = spec.key() + f"|{through}"
    if key in cache.runs:
        return cache.runs[key]
    preds = run_config(cfg, spec, window="selection", panel=panel, through=through)
    y = preds["label"].to_numpy()
    folds = preds.attrs.get("folds", [])
    metrics = {
        "n_fights": len(preds),
        "n_events": int(preds["event_id"].nunique()),
        "log_loss": float(per_fight_log_loss(y, preds["p_model"].to_numpy()).mean()),
        "market_log_loss": float(per_fight_log_loss(y, preds["p_market"].to_numpy()).mean()),
        "date_span": [str(preds["date"].min()), str(preds["date"].max())],
        "folds": len(folds),
        "folds_floor_extended": int(sum(f["floor_extended"] for f in folds)),
    }
    if ledger is not None:
        protocol.append_experiment(
            {"stage": stage, "candidate": candidate, "spec": spec.to_dict(), "metrics": metrics},
            cfg,
            path=ledger,
        )
    preds.attrs["metrics"] = metrics
    cache.runs[key] = preds
    return preds


def run_compare_stage(
    stage: dict,
    incumbent: ExperimentSpec,
    panel: pd.DataFrame,
    cfg: Config,
    *,
    alpha: float,
    ledger: Path | None,
    through: dt.date | None = None,
    cache: _Cache | None = None,
) -> tuple[dict, ExperimentSpec]:
    """Score the incumbent and every candidate; return the stage record and the winner."""
    cache = cache or _Cache()
    name = stage["name"]
    base = _scored(
        incumbent,
        panel,
        cfg,
        stage=name,
        candidate="incumbent",
        ledger=ledger,
        through=through,
        cache=cache,
    )
    base_loss = per_fight_log_loss(base["label"].to_numpy(), base["p_model"].to_numpy())
    specs = candidate_specs(stage, incumbent)
    results = []
    for cand_name, spec in specs.items():
        preds = _scored(
            spec,
            panel,
            cfg,
            stage=name,
            candidate=cand_name,
            ledger=ledger,
            through=through,
            cache=cache,
        )
        if not preds["fight_id"].equals(base["fight_id"]):
            raise RuntimeError(f"{name}/{cand_name} scored a different set of fights")
        loss = per_fight_log_loss(preds["label"].to_numpy(), preds["p_model"].to_numpy())
        test = paired_test(
            loss,
            base_loss,
            preds["event_id"].to_numpy(),
            n_resamples=int(cfg.backtest["bootstrap_resamples"]),
            seed=cfg.seed,
            ci=float(cfg.backtest["bootstrap_ci"]),
        )
        results.append(
            {
                "name": cand_name,
                "spec": spec.to_dict(),
                "log_loss": float(loss.mean()),
                **test,
                "folds_floor_extended": preds.attrs["metrics"]["folds_floor_extended"],
            }
        )
    winner, rows = decide(results, alpha=alpha)
    record = {
        "name": name,
        "kind": "compare",
        "field": stage["field"],
        "incumbent": {"spec": incumbent.to_dict(), **base.attrs["metrics"]},
        "candidates": rows,
        "adopted": winner,
    }
    return record, (specs[winner] if winner else incumbent)


def retune_split(panel: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout = the last 20% of labelled 2010-2014 fights; train = everything before it."""
    w = cfg.windows
    data = panel.loc[panel["date"] <= w.selection_end].sort_values(
        ["date", "fight_id"], kind="stable"
    )
    sel = data.loc[data["date"] >= w.dev_start]
    n_val = round(RETUNE_HOLDOUT * len(sel))
    boundary = sel["date"].iloc[len(sel) - n_val]
    val = data.loc[data["date"] >= boundary]  # boundary-date ties move into the holdout
    train = data.loc[data["date"] < boundary]
    protocol.assert_scorable(val["date"], cfg, purpose="selection")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def retune(spec: ExperimentSpec, panel: pd.DataFrame, cfg: Config) -> dict:
    """The stage-0 grid search for ``spec`` (calibration off, as in tune.py)."""
    if spec.model != "lightgbm":
        raise ValueError("only a LightGBM configuration is re-tuned")
    train, val = retune_split(panel, cfg)
    fixed = {k: v for k, v in cfg.lightgbm.items() if k not in SEARCH_SPACE}
    columns = spec.columns()
    if spec.top_k is not None:
        from ufc_predictor.experiments.selection import select_top_k

        columns = select_top_k(train, columns, spec.top_k, cfg, params=spec.params(cfg))
    keys = list(SEARCH_SPACE)
    results = []
    y_val = val["label"].astype(int).to_numpy()
    for combo in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        params = {**fixed, **dict(zip(keys, combo, strict=True))}
        cand = dataclasses.replace(spec.with_params(params), calibration="none")
        model = make_model(cand, cfg, columns).fit(train, train["label"].astype(int))
        p = model.predict_proba(val)
        inner = getattr(model, "inner", model)
        results.append(
            {
                **dict(zip(keys, combo, strict=True)),
                "log_loss": float(log_loss(y_val, p, labels=[0, 1])),
                "best_iteration": getattr(inner, "best_iteration_", None),
            }
        )
    selected, tied = select_best(results)
    params = {**fixed, **{k: selected[k] for k in keys}}
    return {
        "train_rows": len(train),
        "train_span": [str(train["date"].min()), str(train["date"].max())],
        "validation_rows": len(val),
        "validation_span": [str(val["date"].min()), str(val["date"].max())],
        "n_candidates": len(results),
        "n_tied": len(tied),
        "selected": selected,
        "params": params,
        "lowest_log_loss": min(r["log_loss"] for r in results),
    }


def run_all(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Run every pre-registered stage and write ``stages.json`` and ``frozen.yaml``."""
    cfg = cfg or get_config()
    prereg_commit = protocol.assert_after_preregistration(REPO_ROOT)
    state = git_state(ignore=OUTPUT_DIRS)
    if not state["clean"]:
        raise RuntimeError(
            "commit the code before running experiments; the ledger records the commit"
        )
    pre = protocol.load_preregistration()
    alpha = float(pre["alpha"])
    ledger = protocol.ledger_path(cfg)
    out_dir = cfg.paths.reports_dir / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = runner.family_panel(cfg)
    cache = _Cache()
    records: list[dict] = []

    stage0 = pre["stages"][0]
    if stage0["name"] != "stage_0_retune":
        raise RuntimeError("pre-registration does not start with stage_0_retune")
    tuned = retune(ExperimentSpec(), panel, cfg)
    incumbent = ExperimentSpec().with_params(tuned["params"])
    records.append(
        {"name": "stage_0_retune", "kind": "retune", **tuned, "baseline_spec": incumbent.to_dict()}
    )
    baseline = incumbent
    if verbose:
        print(f"stage_0_retune: {tuned['selected']}")

    for stage in pre["stages"][1:]:
        record, incumbent = run_compare_stage(
            stage, incumbent, panel, cfg, alpha=alpha, ledger=ledger, cache=cache
        )
        records.append(record)
        if verbose:
            print(f"{stage['name']}: adopted {record['adopted']}")

    final = {"name": "final_retune", "kind": "final_retune"}
    if pre["final_retune"]["enabled"] and incumbent.model == "lightgbm":
        re = retune(incumbent, panel, cfg)
        cand = incumbent.with_params(re["params"])
        base = _scored(
            incumbent,
            panel,
            cfg,
            stage="final_retune",
            candidate="incumbent",
            ledger=ledger,
            through=None,
            cache=cache,
        )
        preds = _scored(
            cand,
            panel,
            cfg,
            stage="final_retune",
            candidate="retuned",
            ledger=ledger,
            through=None,
            cache=cache,
        )
        y = base["label"].to_numpy()
        test = paired_test(
            per_fight_log_loss(y, preds["p_model"].to_numpy()),
            per_fight_log_loss(y, base["p_model"].to_numpy()),
            base["event_id"].to_numpy(),
            n_resamples=int(cfg.backtest["bootstrap_resamples"]),
            seed=cfg.seed,
            ci=float(cfg.backtest["bootstrap_ci"]),
        )
        winner, rows = decide(
            [{"name": "retuned", "log_loss": preds.attrs["metrics"]["log_loss"], **test}],
            alpha=alpha,
        )
        final.update(
            {
                "retune": re,
                "incumbent": base.attrs["metrics"],
                "candidates": rows,
                "adopted": winner,
            }
        )
        if winner:
            incumbent = cand
    else:
        final.update({"skipped": f"model is {incumbent.model}"})
    records.append(final)

    frozen_preds = _scored(
        incumbent,
        panel,
        cfg,
        stage="frozen",
        candidate="frozen",
        ledger=None,
        through=None,
        cache=cache,
    )
    frozen_preds.to_csv(out_dir / "frozen_selection_predictions.csv", index=False)
    result = {
        "preregistration_commit": prereg_commit,
        "git_commit": state["commit"],
        "selection_window": [str(cfg.windows.dev_start), str(cfg.windows.selection_end)],
        "alpha": alpha,
        "stages": records,
        "baseline_spec": baseline.to_dict(),
        "frozen_spec": incumbent.to_dict(),
        "frozen_selection_log_loss": frozen_preds.attrs["metrics"]["log_loss"],
        "frozen_selection_market_log_loss": frozen_preds.attrs["metrics"]["market_log_loss"],
    }
    (out_dir / "stages.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "frozen.yaml").write_text(
        "# The frozen development configuration (Phase 8), written by `ufc-predictor experiments`.\n"
        + yaml.safe_dump(
            {"spec": incumbent.to_dict(), "baseline_spec": baseline.to_dict()}, sort_keys=True
        ),
        encoding="utf-8",
    )
    return result


def load_spec(d: dict) -> ExperimentSpec:
    """An ExperimentSpec from its ``to_dict`` form."""
    params = d.get("lightgbm_params")
    return ExperimentSpec(
        features=tuple(d["features"]),
        top_k=d.get("top_k"),
        train_window=d["train_window"],
        symmetric=bool(d["symmetric"]),
        calibration=d["calibration"],
        model=d["model"],
        lightgbm_params=tuple(sorted(params.items())) if params else None,
    )
