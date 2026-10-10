"""Monte-Carlo parlay pricing on real moneyline odds.

**Research backtest only. Not betting advice.** Nothing here should be used to wager.

**Only real prices are used.** The committed odds snapshot contains moneylines and
nothing else, so the only legs that exist are "fighter X wins". Method-of-victory and
round props are *not* priced, because this project has no odds for them and inventing a
payout would turn a measurement into a fabrication.

**Correlation within a fight is exact, not assumed.** Each simulation draw resolves every
fight on the card once, and every candidate leg is evaluated against that same draw. Two
legs from the same fight are therefore perfectly dependent by construction: backing both
fighters in one bout yields a joint probability of exactly zero, and backing the same
fighter twice yields his own win probability rather than its square. Candidate generation
also refuses to place two legs from one fight, so the degenerate cases never reach a
stake -- the simulator's handling of them is a correctness check, exercised in the tests.

**Across fights the model treats bouts as conditionally independent** given their
features. That assumption is stated rather than hidden, and measured in the pipeline by
testing whether per-card residuals correlate.

A consequence worth stating plainly: with independent legs, a parlay's expected value is
``prod(p_i * payout_i) - 1``, so a parlay is attractive only when **every** leg is
individually attractive. Parlays do not create edge; they multiply it, and multiply
variance a great deal faster.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ufc_predictor.ingest.odds import american_to_decimal


@dataclass(frozen=True)
class Leg:
    """One moneyline selection."""

    fight_id: str
    fight_index: int  # position of the fight within the card, for the sim draw matrix
    side: int  # 1 = fighter_1 wins, 2 = fighter_2 wins
    model_p: float
    decimal_odds: float  # the book's actual price, vig included
    won: bool

    @property
    def payout_ev(self) -> float:
        """``p * payout`` for this leg; above 1.0 means positive expected value."""
        return self.model_p * self.decimal_odds


def build_legs(card: pd.DataFrame, prob_col: str) -> list[Leg]:
    """Both moneyline selections for every fight on a card.

    ``card`` must carry ``odds_1``/``odds_2`` (American, canonical orientation),
    ``label`` (1.0 when fighter_1 won) and the chosen probability column.
    """
    legs: list[Leg] = []
    for idx, row in enumerate(card.itertuples(index=False)):
        p1 = float(getattr(row, prob_col))
        if not np.isfinite(p1):
            continue  # the variant abstains during warm-up; no price, no leg
        won_1 = bool(row.label == 1.0)
        for side, p, american in ((1, p1, row.odds_1), (2, 1.0 - p1, row.odds_2)):
            if american is None or not np.isfinite(american):
                continue
            legs.append(
                Leg(
                    fight_id=row.fight_id,
                    fight_index=idx,
                    side=side,
                    model_p=p,
                    decimal_odds=american_to_decimal(float(american)),
                    won=(won_1 if side == 1 else not won_1),
                )
            )
    return legs


def simulate_card(card: pd.DataFrame, prob_col: str, n_sims: int, rng: np.random.Generator):
    """Draw ``n_sims`` resolutions of the whole card.

    Returns a boolean array of shape ``(n_sims, n_fights)`` that is True where fighter_1
    won that draw. Drawing per fight -- rather than per leg -- is what makes same-fight
    legs correctly dependent.
    """
    p1 = card[prob_col].to_numpy(dtype=float)
    valid = np.isfinite(p1)
    draws = np.zeros((n_sims, len(p1)), dtype=bool)
    if valid.any():
        draws[:, valid] = rng.random((n_sims, int(valid.sum()))) < p1[valid]
    return draws


def candidate_parlays(legs: list[Leg], max_legs: int) -> list[tuple[Leg, ...]]:
    """Every combination of up to ``max_legs`` legs, at most one leg per fight.

    Only legs with positive expected value are combined. That is not a heuristic: for
    independent legs the parlay's EV is ``prod(p_i * payout_i) - 1``, so including a leg
    with ``p * payout < 1`` strictly lowers the product. Pruning them cannot discard a
    parlay that would have ranked highly.
    """
    positive = [leg for leg in legs if leg.payout_ev > 1.0]
    out: list[tuple[Leg, ...]] = []
    for k in range(1, max_legs + 1):
        for combo in itertools.combinations(positive, k):
            if len({leg.fight_index for leg in combo}) == len(combo):
                out.append(combo)
    return out


def price_parlays(
    card: pd.DataFrame,
    prob_col: str,
    *,
    n_sims: int,
    max_legs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Price every candidate parlay on one card by simulation.

    Returns one row per candidate with the simulated joint probability, the analytic
    product for comparison, the book payout, expected value, and whether it actually won.
    """
    legs = build_legs(card, prob_col)
    if not legs:
        return pd.DataFrame()

    draws = simulate_card(card, prob_col, n_sims, rng)
    candidates = candidate_parlays(legs, max_legs)
    if not candidates:
        return pd.DataFrame()

    rows = []
    for combo in candidates:
        # Evaluate every leg against the same per-fight draw: this is where same-fight
        # dependence is handled exactly rather than assumed away.
        hit = np.ones(n_sims, dtype=bool)
        for leg in combo:
            outcome = draws[:, leg.fight_index]
            hit &= outcome if leg.side == 1 else ~outcome

        p_sim = float(hit.mean())
        p_analytic = float(np.prod([leg.model_p for leg in combo]))
        payout = float(np.prod([leg.decimal_odds for leg in combo]))
        rows.append(
            {
                "event_id": card["event_id"].iloc[0],
                "date": card["date"].iloc[0],
                "n_legs": len(combo),
                "fight_ids": "|".join(leg.fight_id for leg in combo),
                "sides": "".join(str(leg.side) for leg in combo),
                "p_sim": p_sim,
                "p_analytic": p_analytic,
                "payout": payout,
                # EV per unit staked, using the book's actual price (vig included).
                "ev": p_sim * payout - 1.0,
                "ev_analytic": p_analytic * payout - 1.0,
                "won": bool(all(leg.won for leg in combo)),
                "realized_return": (payout - 1.0) if all(leg.won for leg in combo) else -1.0,
            }
        )
    return pd.DataFrame(rows)


def kelly_fraction_for(p: float, payout: float) -> float:
    """Full-Kelly stake fraction for a binary bet at decimal ``payout``.

    ``b = payout - 1`` is the profit multiple. Returns 0 when the bet is not favourable,
    so a negative-edge parlay can never be staked.
    """
    b = payout - 1.0
    if b <= 0:
        return 0.0
    f = (p * b - (1.0 - p)) / b
    return max(0.0, float(f))
