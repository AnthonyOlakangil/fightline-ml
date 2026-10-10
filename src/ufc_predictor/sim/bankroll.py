"""Bankroll replay and the event-level bootstrap.

**Research backtest only. Not betting advice.**

Stakes are fractional Kelly with a hard cap. Full Kelly is wildly aggressive when the
edge estimate is wrong, and the whole point of this project's result is that the edge
estimate *is* wrong -- the model loses to the closing line. The cap bounds how much any
single mispriced parlay can cost.

The bootstrap resamples **events**, not individual bets, for the same reason the metric
bootstrap does: bets placed on one card share a model fit, a night, and correlated
errors, so treating them as independent would understate the spread of outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def replay(
    bets: pd.DataFrame,
    *,
    starting_bankroll: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    ruin_threshold: float,
) -> dict:
    """Replay bets chronologically and summarise the bankroll path.

    ``bets`` needs ``date``, ``event_id``, ``p_sim``, ``payout`` and ``realized_return``.
    Stakes compound: each is sized against the bankroll at that moment.
    """
    from ufc_predictor.sim.parlay import kelly_fraction_for

    bankroll = starting_bankroll
    ruin_level = starting_bankroll * ruin_threshold
    peak = bankroll
    max_drawdown = 0.0
    ruined = False
    path = [bankroll]
    staked_total = 0.0
    n_bets = 0

    for row in bets.sort_values(["date", "event_id"], kind="stable").itertuples(index=False):
        if bankroll <= ruin_level:
            ruined = True
            break
        f = kelly_fraction * kelly_fraction_for(row.p_sim, row.payout)
        f = min(f, max_bet_fraction)
        stake = bankroll * f
        if stake <= 0:
            continue
        bankroll += stake * row.realized_return
        staked_total += stake
        n_bets += 1
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak > 0 else 0.0)
        path.append(bankroll)
        if bankroll <= ruin_level:
            ruined = True
            break

    return {
        "ending_bankroll": float(bankroll),
        "starting_bankroll": float(starting_bankroll),
        "total_return": float(bankroll / starting_bankroll - 1.0),
        "n_bets": int(n_bets),
        "total_staked": float(staked_total),
        "roi_on_turnover": float((bankroll - starting_bankroll) / staked_total)
        if staked_total > 0
        else 0.0,
        "max_drawdown": float(max_drawdown),
        "ruined": bool(ruined),
        "path": [float(x) for x in path],
    }


def bootstrap_by_event(
    bets: pd.DataFrame,
    *,
    starting_bankroll: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    ruin_threshold: float,
    n_resamples: int,
    seed: int,
) -> dict:
    """Resample whole events with replacement and replay each resampled history.

    Produces the distribution of ending bankroll, max drawdown and ruin, which is the
    honest way to present a single backtest path: that path is one draw from a very wide
    distribution, not a forecast.
    """
    if bets.empty:
        return {
            "n_resamples": 0,
            "note": "no bets were placed, so there is no bankroll distribution",
        }

    events = bets["event_id"].unique()
    by_event = dict(iter(bets.groupby("event_id")))
    rng = np.random.default_rng(seed)

    endings: list[float] = []
    drawdowns: list[float] = []
    paths: list[list[float]] = []
    ruins = 0

    for _ in range(n_resamples):
        picks = rng.choice(events, size=len(events), replace=True)
        frames = [by_event[e] for e in picks]
        sample = pd.concat(frames, ignore_index=True)
        # Each resampled history is replayed in the order its events were drawn, which
        # keeps the compounding dynamic while shuffling which cards occur.
        sample["date"] = np.repeat(np.arange(len(picks)), [len(f) for f in frames])
        sample["event_id"] = sample["date"]
        res = replay(
            sample,
            starting_bankroll=starting_bankroll,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
            ruin_threshold=ruin_threshold,
        )
        endings.append(res["ending_bankroll"])
        drawdowns.append(res["max_drawdown"])
        paths.append(res["path"])
        ruins += int(res["ruined"])

    endings_arr = np.array(endings)

    # Percentile band across resampled paths. A path that ruins stops early, so it is
    # held at its final value rather than dropped -- otherwise the surviving paths
    # would make the later part of the band look far healthier than it was.
    max_len = max(len(p) for p in paths)
    grid = np.full((len(paths), max_len), np.nan)
    for i, pth in enumerate(paths):
        grid[i, : len(pth)] = pth
        grid[i, len(pth) :] = pth[-1]
    band = {
        "bet_index": list(range(max_len)),
        "p05": [float(x) for x in np.nanquantile(grid, 0.05, axis=0)],
        "median": [float(x) for x in np.nanquantile(grid, 0.50, axis=0)],
        "p95": [float(x) for x in np.nanquantile(grid, 0.95, axis=0)],
    }
    return {
        "n_resamples": int(n_resamples),
        "ending_bankroll": {
            "median": float(np.median(endings_arr)),
            "mean": float(endings_arr.mean()),
            "p05": float(np.quantile(endings_arr, 0.05)),
            "p95": float(np.quantile(endings_arr, 0.95)),
            "worst": float(endings_arr.min()),
            "best": float(endings_arr.max()),
        },
        "share_losing_money": float((endings_arr < starting_bankroll).mean()),
        "max_drawdown": {
            "median": float(np.median(drawdowns)),
            "p95": float(np.quantile(drawdowns, 0.95)),
            "worst": float(np.max(drawdowns)),
        },
        "ruin_probability": float(ruins / n_resamples),
        "ruin_threshold_fraction": float(ruin_threshold),
        "path_band": band,
    }
