"""Scoring a forward card after it happens.

A prediction is scored only against the bout it was made for. Matching is by ESPN
competition id **and** the same two fighters; otherwise the bout is void, for a stated
reason, and never scored against whoever actually fought:

* ``void_replaced``   -- the slot was held, but against a different opponent;
* ``void_not_held``   -- cancelled, or absent from the results;
* ``void_draw_or_nc`` -- a draw or no contest, which carries no binary outcome.

The market comparison uses Kalshi's mid at the bout's segment start, normalized across
the two fighters. A side with no quote leaves the close missing, never 0.5.
"""

from __future__ import annotations

import math

import pandas as pd

from ufc_predictor.forward.espn import Result
from ufc_predictor.ingest.names import normalize_name

STATUSES = ("scored", "void_replaced", "void_not_held", "void_draw_or_nc")


def _pair(a: str, b: str) -> frozenset[str]:
    return frozenset((normalize_name(a), normalize_name(b)))


def _loss(y: float, p: float) -> float:
    if math.isnan(y) or p is None or math.isnan(p):
        return math.nan
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p) if y == 1.0 else -math.log(1 - p)


def close_probability(quotes: dict, ticker_a: str, ticker_b: str) -> float:
    qa, qb = quotes.get(ticker_a), quotes.get(ticker_b)
    if qa is None or qb is None:
        return math.nan
    mid_a, mid_b = sum(qa) / 2, sum(qb) / 2
    return mid_a / (mid_a + mid_b)


def settle_bouts(predictions: pd.DataFrame, results: list[Result], quotes: dict) -> pd.DataFrame:
    """One row per predicted bout: status, outcome for fighter_a, prices and losses."""
    by_id = {r.competition_id: r for r in results}
    by_pair = {_pair(*r.pair): r for r in results}
    rows = []
    for pred in predictions.itertuples(index=False):
        pair = _pair(pred.fighter_a, pred.fighter_b)
        result = by_id.get(pred.competition_id)
        if result is None or _pair(*result.pair) != pair:
            same_pair = by_pair.get(pair)
            if same_pair is not None:
                result = same_pair
            elif result is not None and result.completed:
                result, status = None, "void_replaced"
            else:
                result, status = None, "void_not_held"
        if result is not None:
            if not result.completed:
                status = "void_not_held"
            elif result.winner is None:
                status = "void_draw_or_nc"
            else:
                status = "scored"
        y = math.nan
        if status == "scored":
            y = 1.0 if normalize_name(result.winner) == normalize_name(pred.fighter_a) else 0.0
        p_close = close_probability(quotes, pred.kalshi_ticker_a, pred.kalshi_ticker_b)
        rows.append(
            {
                "competition_id": pred.competition_id,
                "fighter_a": pred.fighter_a,
                "fighter_b": pred.fighter_b,
                "status": status,
                "y_a": y,
                "p_model_a": float(pred.p_model_a),
                "kalshi_p_a": float(pred.kalshi_p_a),
                "kalshi_close_p_a": p_close,
                "ll_model": _loss(y, float(pred.p_model_a)),
                "ll_kalshi_predict": _loss(y, float(pred.kalshi_p_a)),
                "ll_kalshi_close": _loss(y, p_close),
            }
        )
    return pd.DataFrame(rows)


def summarize(settled: pd.DataFrame) -> dict:
    """Counts and mean log losses; market comparisons on bouts that have both prices."""
    scored = settled.loc[settled["status"] == "scored"]
    voids = settled.loc[settled["status"] != "scored", "status"].value_counts().sort_index()
    with_close = scored.loc[scored["ll_kalshi_close"].notna()]
    with_pred = scored.loc[scored["ll_kalshi_predict"].notna()]

    def mean(s: pd.Series) -> float | None:
        return float(s.mean()) if len(s) else None

    return {
        "n_predicted": len(settled),
        "n_scored": len(scored),
        "voids": {k: int(v) for k, v in voids.items()},
        "model_log_loss": mean(scored["ll_model"]),
        "n_with_kalshi_close": len(with_close),
        "kalshi_close_log_loss": mean(with_close["ll_kalshi_close"]),
        "model_minus_kalshi_close": mean(with_close["ll_model"] - with_close["ll_kalshi_close"]),
        "n_with_kalshi_at_prediction": len(with_pred),
        "model_minus_kalshi_at_prediction": mean(
            with_pred["ll_model"] - with_pred["ll_kalshi_predict"]
        ),
        "model_accuracy": mean(
            ((scored["p_model_a"] > 0.5) == (scored["y_a"] == 1.0)).astype(float)
        )
        if len(scored)
        else None,
    }
