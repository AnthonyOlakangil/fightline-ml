"""Post-hoc checks on the stored walk-forward predictions.

    make report        # writes reports/post_hoc.json

Both checks were prompted by the second adversarial review (``reports/review_run2.md``)
and read only ``reports/walkforward_predictions.csv``. Neither refits LightGBM, Elo or
the market, so the evaluation run they describe is unchanged. One of them does fit a new
*control* on evaluation-window history, and the README's glimpse disclosure lists it.

1. **What the market-anchored variant's gain is made of.** The variant is a walk-forward
   logistic stacker on ``logit(p_market)`` and ``logit(p_lightgbm)``. Replaying the same
   protocol with ``logit(p_market)`` alone isolates how much of the gain is the market
   recalibrating itself -- proportional de-vig leaves a favourite-longshot bias -- and
   how much LightGBM adds on top.
2. **Where LightGBM's excess log loss comes from.** The per-fold isotonic calibrator can
   map the most extreme raw scores to exactly 0 or 1, which are clipped to ``EPS``; each
   such prediction that misses costs ``-ln(EPS)`` nats.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ufc_predictor.backtest.metrics import EPS, paired_bootstrap_by_event, per_fight_log_loss
from ufc_predictor.backtest.walkforward import MIN_STACK_ROWS, VARIANT_NAME, _logit
from ufc_predictor.config import Config, get_config

#: A prediction this close to 0 or 1 is treated as a literal certainty.
CERTAINTY_TOLERANCE = 1e-5


def replay_stacker(preds: pd.DataFrame, inputs: list[str], cfg: Config) -> tuple[np.ndarray, list]:
    """Re-run the walk-forward stacker protocol on stored predictions.

    As in ``walkforward``: the stacker for fold ``k`` is fitted on every row predicted in
    an earlier fold, and abstains (NaN) until it has ``MIN_STACK_ROWS`` of them.

    Returns:
        Predictions aligned with ``preds``, and each fitted fold's coefficients.
    """
    out = np.full(len(preds), np.nan)
    coefs: list[dict] = []
    for fold in sorted(preds["fold"].unique()):
        hist = preds.loc[preds["fold"] < fold]
        rows = (preds["fold"] == fold).to_numpy()
        y = hist["label"].astype(int).to_numpy()
        if len(hist) < MIN_STACK_ROWS or len(np.unique(y)) < 2:
            continue
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=cfg.seed)
        clf.fit(np.column_stack([_logit(hist[c].to_numpy()) for c in inputs]), y)
        Z = np.column_stack([_logit(preds.loc[rows, c].to_numpy()) for c in inputs])
        out[rows] = np.clip(clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        coefs.append(
            {"fold": int(fold), **dict(zip(inputs, map(float, clf.coef_[0]), strict=True))}
        )
    return out, coefs


def _paired(frame: pd.DataFrame, col_a: str, col_b: str, cfg: Config) -> dict:
    y = frame["label"].to_numpy(dtype=float)
    return paired_bootstrap_by_event(
        per_fight_log_loss(y, frame[col_a].to_numpy(dtype=float)),
        per_fight_log_loss(y, frame[col_b].to_numpy(dtype=float)),
        frame["event_id"].to_numpy(),
        n_resamples=int(cfg.backtest.get("bootstrap_resamples", 10000)),
        seed=cfg.seed,
        ci=float(cfg.backtest.get("bootstrap_ci", 0.95)),
    )


def variant_decomposition(preds: pd.DataFrame, cfg: Config) -> dict:
    """Split the variant's eval gain into market recalibration and LightGBM's addition.

    Raises:
        RuntimeError: if replaying the stacker does not reproduce the stored variant,
            which would mean the replay is not the protocol it claims to be.
    """
    replayed, _ = replay_stacker(preds, ["p_market", "p_lightgbm"], cfg)
    stored = preds[f"p_{VARIANT_NAME}"].to_numpy()
    both = ~np.isnan(stored)
    if not np.array_equal(both, ~np.isnan(replayed)) or (
        np.max(np.abs(replayed[both] - stored[both])) > 1e-9
    ):
        raise RuntimeError("replaying the stacker does not reproduce the stored variant")

    recal, coefs = replay_stacker(preds, ["p_market"], cfg)
    _, variant_coefs = replay_stacker(preds, ["p_market", "p_lightgbm"], cfg)
    frame = preds.assign(p_market_recalibrated=recal)
    ev = frame.loc[
        (frame["date"] >= cfg.windows.eval_start) & frame["p_market_recalibrated"].notna()
    ]
    ev = ev.loc[ev[f"p_{VARIANT_NAME}"].notna()]
    eval_folds = set(ev["fold"])

    def coef_range(rows: list[dict], key: str) -> list[float]:
        vals = [r[key] for r in rows if r["fold"] in eval_folds]
        return [min(vals), max(vals)]

    return {
        "n_fights": len(ev),
        "replay_matches_stored_variant": True,
        "recalibrated_minus_market": _paired(ev, "p_market_recalibrated", "p_market", cfg),
        "variant_minus_recalibrated": _paired(
            ev, f"p_{VARIANT_NAME}", "p_market_recalibrated", cfg
        ),
        "variant_minus_market": _paired(ev, f"p_{VARIANT_NAME}", "p_market", cfg),
        "eval_fold_coefficients": {
            "recalibration_slope_on_logit_market": coef_range(coefs, "p_market"),
            "variant_weight_on_logit_market": coef_range(variant_coefs, "p_market"),
            "variant_weight_on_logit_lightgbm": coef_range(variant_coefs, "p_lightgbm"),
        },
    }


def certainty_audit(preds: pd.DataFrame, cfg: Config) -> dict:
    """How many LightGBM predictions are literal certainties, and what they cost."""
    lo, hi = (float(b) for b in cfg.lightgbm_fit["calibration_bounds"])
    out = {
        "tolerance": CERTAINTY_TOLERANCE,
        "cost_of_a_miss_nats": -math.log(EPS),
        "bounds": [lo, hi],
        "cost_of_a_miss_at_the_bound_nats": -math.log(lo),
    }
    windows = {
        "dev": (cfg.windows.dev_start, cfg.windows.dev_end),
        "eval": (cfg.windows.eval_start, cfg.windows.eval_end),
    }
    for name, (start, end) in windows.items():
        w = preds.loc[(preds["date"] >= start) & (preds["date"] <= end)]
        p = w["p_lightgbm"].to_numpy(dtype=float)
        y = w["label"].to_numpy(dtype=float)
        certain = (p <= CERTAINTY_TOLERANCE) | (p >= 1 - CERTAINTY_TOLERANCE)
        wrong = certain & ((p > 0.5) != (y > 0.5))
        at_bound = (np.abs(p - lo) < 1e-9) | (np.abs(p - hi) < 1e-9)
        loss = per_fight_log_loss(y, p)
        out[name] = {
            "n": len(w),
            "certain": int(certain.sum()),
            "certain_and_wrong": int(wrong.sum()),
            "at_bound": int(at_bound.sum()),
            "at_bound_and_wrong": int((at_bound & ((p > 0.5) != (y > 0.5))).sum()),
            "log_loss": float(loss.mean()),
            "log_loss_without_certainties": float(loss[~certain].mean()),
            "share_of_total_loss_from_wrong_certainties": float(loss[wrong].sum() / loss.sum()),
        }
    return out


def compute(cfg: Config | None = None) -> dict:
    """Both checks, from the stored predictions; writes nothing."""
    cfg = cfg or get_config()
    path = cfg.paths.reports_dir / "walkforward_predictions.csv"
    preds = pd.read_csv(path, parse_dates=["date"])
    preds["date"] = preds["date"].dt.date
    out = {
        "source": "reports/walkforward_predictions.csv",
        "note": (
            "Post-hoc checks on stored predictions, prompted by reports/review_run2.md. No "
            "evaluated model is refitted; the market-recalibration control is a new fit on "
            "walk-forward history and is disclosed as an evaluation-window look."
        ),
        "variant_decomposition": variant_decomposition(preds, cfg),
        "lightgbm_certainties": certainty_audit(preds, cfg),
    }
    return out


def run(cfg: Config | None = None) -> dict:
    """Compute both checks and write ``reports/post_hoc.json``."""
    cfg = cfg or get_config()
    out = compute(cfg)
    target = cfg.paths.reports_dir / "post_hoc.json"
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
