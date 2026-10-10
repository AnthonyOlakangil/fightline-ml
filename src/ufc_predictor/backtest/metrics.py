"""Scoring metrics and the paired bootstrap.

Two choices here are load-bearing for the headline claim.

**Paired comparison, not two separate point estimates.** "Model log loss 0.62 vs market
0.61" invites eyeballing a difference whose uncertainty is unstated. Every model is
instead compared to the market *per fight*, and the quantity with a confidence interval
is the mean difference. If that interval contains zero, the model has not been shown to
differ from the market, and the README says so.

**Resampling by event, not by fight.** Fights on one card are not independent draws: they
share a venue, a night, a judging crew, and a shared set of conditions, and a model's
errors tend to correlate within a card. Resampling individual fights would treat ~4,600
fights as ~4,600 independent observations and produce an interval that is too narrow.
Resampling whole events preserves the within-card correlation, which is the honest unit
of independence here -- there are 421 eval events, not 4,636 independent trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Matches the clipping in models.base, so a metric can never see 0 or 1.
EPS = 1e-6


def per_fight_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Negative log likelihood for each fight, which is what the paired test needs."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with equal-width bins, plus the reliability table.

    ECE is the average gap between predicted and realised frequency, weighted by how
    many fights fall in each bin. It is reported alongside log loss because a model can
    rank fights well (good AUC) while being systematically overconfident, and a bettor
    or a reader cares about whether "70%" means 70%.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False puts p == 1.0 in an extra bin; clip back.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": b,
                    "lower": float(edges[b]),
                    "upper": float(edges[b + 1]),
                    "n": 0,
                    "mean_predicted": float("nan"),
                    "observed_rate": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        mean_p = float(p[mask].mean())
        obs = float(y[mask].mean())
        ece += (n / len(p)) * abs(mean_p - obs)
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": n,
                "mean_predicted": mean_p,
                "observed_rate": obs,
                "gap": mean_p - obs,
            }
        )
    return float(ece), pd.DataFrame(rows)


def score(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> dict:
    """All headline metrics for one model on one set of fights."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    if len(y) == 0:
        raise ValueError("cannot score an empty set of fights")

    ece, _ = expected_calibration_error(y, p, n_bins=n_bins)
    out = {
        "n": len(y),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
        "ece": ece,
        "mean_prediction": float(p.mean()),
        "base_rate": float(y.mean()),
    }
    # AUC is undefined with a single class present.
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def paired_bootstrap_by_event(
    loss_model: np.ndarray,
    loss_market: np.ndarray,
    event_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Confidence interval for mean(loss_model - loss_market), resampling events.

    The statistic is the difference in mean per-fight log loss. Negative means the model
    beats the market. Events are the resampling unit, so a card's fights move together.

    Returns the observed difference, the CI, the share of resamples favouring the model,
    and a two-sided bootstrap p-value for the null that the difference is zero.
    """
    loss_model = np.asarray(loss_model, dtype=float)
    loss_market = np.asarray(loss_market, dtype=float)
    event_ids = np.asarray(event_ids)

    if not (len(loss_model) == len(loss_market) == len(event_ids)):
        raise ValueError("loss arrays and event ids must be the same length")
    if len(loss_model) == 0:
        raise ValueError("nothing to bootstrap")

    diff = loss_model - loss_market
    observed = float(diff.mean())

    # Collapse to per-event (sum, count) so a resample is a weighted mean rather than a
    # repeated concatenation. Equivalent, and fast enough for 10,000 resamples.
    frame = pd.DataFrame({"event_id": event_ids, "diff": diff})
    grouped = frame.groupby("event_id")["diff"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_events = len(sums)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_events, size=(n_resamples, n_events))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])

    # Two-sided bootstrap p-value: how often the resampled difference lands on the
    # opposite side of zero from the observed effect (doubled).
    tail = float((boot >= 0).mean()) if observed < 0 else float((boot <= 0).mean())
    p_value = min(1.0, 2.0 * tail)

    return {
        "statistic": "mean(log_loss_model - log_loss_market)",
        "observed": observed,
        "ci_lower": float(lo),
        "ci_hi": float(hi),
        "ci_level": ci,
        "n_resamples": int(n_resamples),
        "n_events_resampled": int(n_events),
        "n_fights": len(diff),
        "share_favouring_model": float((boot < 0).mean()),
        "p_value_two_sided": p_value,
        "beats_market": bool(hi < 0),
        "interpretation": (
            "negative means the model beats the market; the interval excludes zero"
            if hi < 0
            else (
                "the interval contains zero, so no difference from the market is demonstrated"
                if lo <= 0 <= hi
                else "positive and excluding zero: the market beats the model"
            )
        ),
    }


def bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    event_ids: np.ndarray,
    *,
    metric: str,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Event-resampled CI for a single model's log loss or Brier score."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)

    if metric == "log_loss":
        per_fight = per_fight_log_loss(y, p)
    elif metric == "brier":
        per_fight = (p - y) ** 2
    else:
        raise ValueError(f"unsupported metric {metric!r}")

    frame = pd.DataFrame({"event_id": event_ids, "v": per_fight})
    grouped = frame.groupby("event_id")["v"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_resamples, len(sums)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "point": float(per_fight.mean()),
        "ci_lower": float(lo),
        "ci_hi": float(hi),
        "ci_level": ci,
    }
