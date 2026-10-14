"""Closing-line value on the development window: does the line move toward the model?

    uv run ufc-predictor clv      # -> reports/clv.json

The model loses to the closing line (evaluation run 3). A weaker question can still be
asked of the opening line: when the model disagrees with the open, does the market move
toward the model before the close? If so, the model carries information the market
priced in later -- the usual definition of closing-line value. If not, its disagreement
is noise the market correctly ignored.

Both prices come from BestFightOdds (opening price and the midpoint of the closing range,
each de-vigged across the two fighters), so they share one definition. The model's
probabilities are the frozen development configuration's out-of-sample walk-forward
predictions: 2010-2014 from the selection runs and 2015 from the once-only confirmation.
Nothing from 2016 on is read: :func:`analyse` raises on any such date.

Reported, each with an event-bootstrap interval:

(a) ``close_minus_open_log_loss`` -- a sanity check: the close should beat the open;
(b) ``correlation_edge_vs_move`` -- correlation of the model's edge at the open,
    ``p_model - open``, with the move ``close - open``;
(c) ``move_toward_model_when_edge_over_5pp`` -- the mean of ``move * sign(edge)`` over
    fights where the model and the open differ by more than 5 points;
(d) ``model_information_beyond_open`` -- a logistic regression of the move's direction on
    ``logit(open)`` and ``logit(p_model)``; a positive ``logit(p_model)`` coefficient
    means the model predicts where the line goes beyond what the open already says.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ufc_predictor.backtest.metrics import per_fight_log_loss
from ufc_predictor.config import Config, get_config
from ufc_predictor.experiments.protocol import assert_scorable

EDGE_THRESHOLD = 0.05
REQUIRED = ("fight_id", "event_id", "date", "label", "open_p1", "close_p1", "p_model")


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class _Events:
    """Event-level resampling: the same resamples serve every quantity."""

    def __init__(self, event_ids: np.ndarray, *, n_resamples: int, seed: int, ci: float) -> None:
        codes, self.uniques = pd.factorize(pd.Series(event_ids), sort=True)
        self.codes = codes
        self.n = len(self.uniques)
        self.ci = ci
        rng = np.random.default_rng(seed)
        self.picks = rng.integers(0, self.n, size=(n_resamples, self.n))
        # How many times each event is drawn in each resample.
        self.weights = np.zeros((n_resamples, self.n))
        np.add.at(self.weights, (np.arange(n_resamples)[:, None], self.picks), 1.0)

    def sums(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        v = np.where(mask, values, 0.0) if mask is not None else values
        return np.bincount(self.codes, weights=v, minlength=self.n)

    def interval(self, point: float, boot: np.ndarray) -> dict:
        boot = boot[np.isfinite(boot)]
        lo, hi = np.quantile(boot, [(1 - self.ci) / 2, 1 - (1 - self.ci) / 2])
        return {
            "point": float(point),
            "ci_lower": float(lo),
            "ci_hi": float(hi),
            "ci_level": self.ci,
        }

    def mean(self, values: np.ndarray, mask: np.ndarray | None = None) -> dict:
        mask = np.ones(len(values), bool) if mask is None else mask
        s, c = self.sums(values, mask), self.sums(mask.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.weights @ s) / (self.weights @ c)
        return self.interval(values[mask].mean(), boot)

    def correlation(self, x: np.ndarray, y: np.ndarray) -> dict:
        W = self.weights
        n, sx, sy = W @ self.sums(np.ones_like(x)), W @ self.sums(x), W @ self.sums(y)
        sxx, syy, sxy = W @ self.sums(x * x), W @ self.sums(y * y), W @ self.sums(x * y)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
        return self.interval(np.corrcoef(x, y)[0, 1], boot)

    def refit(self, frame: pd.DataFrame, stat) -> dict:
        rows = [np.flatnonzero(self.codes == k) for k in range(self.n)]
        boot = np.array(
            [stat(frame.iloc[np.concatenate([rows[i] for i in pick])]) for pick in self.picks]
        )
        return self.interval(stat(frame), boot)


def _coef(frame: pd.DataFrame) -> float:
    """The logit(p_model) coefficient in a logistic fit of the move's direction."""
    y = (frame["close_p1"] > frame["open_p1"]).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    X = np.column_stack([_logit(frame["open_p1"]), _logit(frame["p_model"])])
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(X, y).coef_[0][1])


def analyse(frame: pd.DataFrame, cfg: Config) -> dict:
    """All four quantities for one frame of development-window fights.

    Raises:
        EvalWindowClosed: if any fight is dated in the closed evaluation window.
        ValueError: if a required column is missing or an opening/closing price is NaN.
    """
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"CLV frame lacks columns {missing}")
    assert_scorable(frame["date"], cfg, purpose="dev")
    for col in ("open_p1", "close_p1", "p_model", "label"):
        if frame[col].isna().any():
            raise ValueError(
                f"{int(frame[col].isna().sum())} fights have no {col.replace('_p1', '')} value"
            )
    f = frame.copy()
    f["edge"] = f["p_model"] - f["open_p1"]
    f["move"] = f["close_p1"] - f["open_p1"]
    events = _Events(
        f["event_id"].to_numpy(),
        n_resamples=int(cfg.backtest["bootstrap_resamples"]),
        seed=cfg.seed,
        ci=float(cfg.backtest["bootstrap_ci"]),
    )
    y = f["label"].to_numpy(dtype=float)
    loss_close = per_fight_log_loss(y, f["close_p1"].to_numpy())
    loss_open = per_fight_log_loss(y, f["open_p1"].to_numpy())
    sanity = events.mean(loss_close - loss_open)

    edge, move = f["edge"].to_numpy(), f["move"].to_numpy()
    corr = events.correlation(edge, move)

    big = np.abs(edge) > EDGE_THRESHOLD
    directional = events.mean(move * np.sign(edge), mask=big)
    directional.update(
        {"n_fights": int(big.sum()), "mean_abs_edge": float(np.abs(edge[big]).mean())}
    )

    coef = events.refit(f, _coef)
    return {
        "n_fights": len(f),
        "n_events": int(f["event_id"].nunique()),
        "date_span": [str(f["date"].min()), str(f["date"].max())],
        "open_log_loss": float(loss_open.mean()),
        "close_log_loss": float(loss_close.mean()),
        "model_log_loss": float(per_fight_log_loss(y, f["p_model"].to_numpy()).mean()),
        "close_minus_open_log_loss": {
            "observed": sanity["point"],
            "ci_lower": sanity["ci_lower"],
            "ci_hi": sanity["ci_hi"],
        },
        "correlation_edge_vs_move": corr,
        "move_toward_model_when_edge_over_5pp": directional,
        "model_information_beyond_open": {"coef_logit_model": coef},
        "edge_threshold": EDGE_THRESHOLD,
    }


def run(cfg: Config | None = None) -> dict:
    """Join BestFightOdds lines to the frozen configuration's dev predictions; write clv.json."""
    cfg = cfg or get_config()
    from ufc_predictor.storage import db

    lines = db.read_table(cfg.paths.processed_dir, "bfo_lines")
    dev = cfg.paths.reports_dir / "dev"
    preds = pd.concat(
        [
            pd.read_csv(dev / "frozen_selection_predictions.csv", parse_dates=["date"]),
            pd.read_csv(dev / "confirmation_predictions.csv", parse_dates=["date"]),
        ],
        ignore_index=True,
    )
    preds["date"] = preds["date"].dt.date
    frame = preds.merge(
        lines[["fight_id", "open_p1", "close_p1"]],
        on="fight_id",
        how="inner",
        validate="one_to_one",
    )
    out = {
        "source": "BestFightOdds open and closing-range midpoint (data/raw/bfo_snapshot_*), "
        "frozen development configuration's out-of-sample predictions",
        "n_dev_predictions": len(preds),
        "n_with_bfo_lines": len(frame),
        **analyse(frame, cfg),
    }
    path = cfg.paths.reports_dir / "clv.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
