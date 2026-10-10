"""Elo ratings, advanced one event-date at a time.

The ordinary Elo loop updates after every match. That is wrong here: fights on one card
happen simultaneously as far as a forecaster is concerned, so a bout must not be rated
using a result from its own card. Early UFC events compound this -- they were one-night
tournaments, and 67 fighter-days in this dataset contain more than one fight -- so a
fighter's second bout of the night must not know the outcome of their first.

Ratings are therefore snapshotted for an entire date, every fight on that date is rated
from the snapshot, and only then are the updates applied.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

DEFAULT_INITIAL = 1500.0
DEFAULT_K = 32.0
DEFAULT_SCALE = 400.0


def expected_score(rating_a: float, rating_b: float, scale: float = DEFAULT_SCALE) -> float:
    """Probability that A beats B under the logistic Elo model.

    >>> expected_score(1500.0, 1500.0)
    0.5
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def build_elo(
    fights: pd.DataFrame,
    *,
    initial_rating: float = DEFAULT_INITIAL,
    k_factor: float = DEFAULT_K,
    scale: float = DEFAULT_SCALE,
) -> pd.DataFrame:
    """Compute pre-fight Elo ratings for every fight.

    Args:
        fights: must contain fight_id, date, fighter_1, fighter_2, winner_id, is_decisive.

    Returns:
        fight_id, elo_1, elo_2, elo_diff -- all *pre-fight*, as of the end of the last
        event strictly before this fight's date.
    """
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    ratings: dict[str, float] = defaultdict(lambda: initial_rating)

    out_ids: list[str] = []
    out_1: list[float] = []
    out_2: list[float] = []

    for _date, day in ordered.groupby("date", sort=True):
        # 1. Rate every fight on this date from the pre-event snapshot.
        pending: list[tuple[str, str, float, float, float | None]] = []
        for row in day.itertuples(index=False):
            r1 = ratings[row.fighter_1]
            r2 = ratings[row.fighter_2]
            out_ids.append(row.fight_id)
            out_1.append(r1)
            out_2.append(r2)

            if not row.is_decisive or pd.isna(row.winner_id):
                # Draws and no contests carry no rating information here. They still
                # count as experience elsewhere; they just do not move Elo.
                continue
            score_1 = 1.0 if row.winner_id == row.fighter_1 else 0.0
            pending.append((row.fighter_1, row.fighter_2, r1, r2, score_1))

        # 2. Only now apply the updates, so nothing on this card saw another result.
        for f1, f2, r1, r2, score_1 in pending:
            exp_1 = expected_score(r1, r2, scale)
            delta = k_factor * (score_1 - exp_1)
            ratings[f1] = r1 + delta
            ratings[f2] = r2 - delta

    elo = pd.DataFrame({"fight_id": out_ids, "elo_1": out_1, "elo_2": out_2})
    elo["elo_diff"] = elo["elo_1"] - elo["elo_2"]
    return elo
