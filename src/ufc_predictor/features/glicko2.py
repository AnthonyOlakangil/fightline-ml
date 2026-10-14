"""Glicko-2 ratings, advanced one event-date at a time (the "glicko" family).

Elo carries one number per fighter. Glicko-2 (Glickman, "Example of the Glicko-2
system", 2013) adds how sure it is of that number -- the rating deviation, RD -- and how
erratic the fighter has been, the volatility. A debutant's rating is wide open; a
fighter with twenty bouts has a narrow one, which widens again during a layoff.

The algorithm is Glickman's, steps 2-8, on his internal scale (``mu = (r - 1500) /
173.7178``, ``phi = RD / 173.7178``), with the volatility solved by the Illinois
iteration to ``epsilon = 1e-6``. Its parameters are his recommended defaults, fixed a
priori in ``config/config.yaml`` (``glicko2``), not searched.

**Time.** A rating period is ``period_days`` long. As of date ``d``, a fighter last
rated on date ``u`` has had ``n = (d - u).days // period_days`` idle periods, and their
RD is ``min(sqrt(phi_u^2 + n * sigma_u^2), phi_initial)``. That is the RD a fight on
``d`` is described with and updated from; Glickman's step 6 then adds the fight's own
period as usual.

**Point in time.** Exactly as in ``features/elo.py``: every bout on a date is described
from the pre-date states, and only then are all of that date's results applied at once.
A fighter with two bouts on one night (the 1990s tournaments) is updated once, from the
pre-night state, with both results in a single rating period, so neither bout sees the
other. :meth:`Glicko2.update_date` refuses a date that is not later than the last one,
so a date cannot be split into two updates.

**What moves a rating.** Only decisive bouts with a winner (score 1 or 0). Draws and no
contests carry no rating information here, mirroring Elo, and do not reset the idle
clock.

Features, all pre-fight (``pit.py`` adds ``glicko_mu_diff``):

* ``glicko_mu_{1,2}`` -- the rating on the display scale;
* ``glicko_rd_{1,2}`` -- the inflated RD on the display scale;
* ``glicko_expected_1 = 1 / (1 + exp(-g(sqrt(phi_1^2 + phi_2^2)) * (mu_1 - mu_2)))`` on
  the internal scale, with ``g(phi) = 1 / sqrt(1 + 3 phi^2 / pi^2)``;
* ``glicko_rd_sum = glicko_rd_1 + glicko_rd_2``.

A debutant has the initial rating and RD: real values, not missing data -- that is what
Glicko-2 says about a newcomer -- and ``is_debut`` flags them.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import pandas as pd

#: Glickman's conversion between the display and internal scales.
SCALE = 173.7178
DISPLAY_CENTER = 1500.0
#: Convergence tolerance of the volatility iteration (Glickman 2013, step 5).
EPSILON = 1e-6

PARAM_KEYS = ("tau", "initial_rating", "initial_rd", "initial_vol", "period_days")

GLICKO_COLS: list[str] = [
    "glicko_mu_1",
    "glicko_mu_2",
    "glicko_rd_1",
    "glicko_rd_2",
    "glicko_mu_diff",
    "glicko_expected_1",
    "glicko_rd_sum",
]
#: What :func:`build_glicko` returns besides fight_id: every column but the diff.
RATED_COLS: list[str] = [c for c in GLICKO_COLS if c != "glicko_mu_diff"]

State = tuple[float, float, float]  # (mu, phi, sigma) on the internal scale


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def expected_score(mu: float, phi: float, mu_opp: float, phi_opp: float) -> float:
    """P(win) from both sides' uncertainty: ``g`` of the combined deviation.

    >>> expected_score(0.0, 2.0, 0.0, 2.0)
    0.5
    """
    combined = math.sqrt(phi * phi + phi_opp * phi_opp)
    return 1.0 / (1.0 + math.exp(-_g(combined) * (mu - mu_opp)))


def _volatility(phi: float, sigma: float, v: float, delta: float, tau: float) -> float:
    """Step 5: the new volatility, by the Illinois iteration."""
    a = math.log(sigma * sigma)
    d2, p2 = delta * delta, phi * phi

    def f(x: float) -> float:
        ex = math.exp(x)
        return ex * (d2 - p2 - v - ex) / (2.0 * (p2 + v + ex) ** 2) - (x - a) / (tau * tau)

    big_a = a
    if d2 > p2 + v:
        big_b = math.log(d2 - p2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        big_b = a - k * tau
    f_a, f_b = f(big_a), f(big_b)
    while abs(big_b - big_a) > EPSILON:
        big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(big_c)
        if f_c * f_b <= 0:
            big_a, f_a = big_b, f_b
        else:
            f_a /= 2.0
        big_b, f_b = big_c, f_c
    return math.exp(big_a / 2.0)


def _update(state: State, games: Sequence[tuple[float, float, float]], tau: float) -> State:
    """Steps 3-7 for one player over one rating period.

    ``games`` holds, per bout, the opponent's (mu, phi) and the player's score.
    """
    if not games:
        raise ValueError("a rating period needs at least one result")
    mu, phi, sigma = state
    v_inv = 0.0
    total = 0.0
    for mu_j, phi_j, score in games:
        g_j = _g(phi_j)
        e_j = 1.0 / (1.0 + math.exp(-g_j * (mu - mu_j)))
        v_inv += g_j * g_j * e_j * (1.0 - e_j)
        total += g_j * (score - e_j)
    v = 1.0 / v_inv
    sigma_new = _volatility(phi, sigma, v, v * total, tau)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    return mu + phi_new * phi_new * total, phi_new, sigma_new


def check_params(params: Mapping[str, float] | None) -> dict[str, float]:
    """The ``glicko2`` config block, validated. No silent defaults.

    Raises:
        ValueError: when absent, on a missing or unknown key, or an out-of-range value.
    """
    if params is None:
        raise ValueError(
            "the glicko family needs glicko_params (config.yaml `glicko2`); there is no default"
        )
    missing = [k for k in PARAM_KEYS if k not in params]
    unknown = sorted(set(params) - set(PARAM_KEYS))
    if missing or unknown:
        raise ValueError(f"glicko_params: missing keys {missing}, unknown keys {unknown}")
    out = {k: float(params[k]) for k in PARAM_KEYS}
    for key in ("tau", "initial_rd", "initial_vol"):
        if not out[key] > 0:
            raise ValueError(f"glicko_params.{key} must be positive, got {params[key]!r}")
    if not (out["period_days"] >= 1 and out["period_days"].is_integer()):
        raise ValueError(
            f"glicko_params.period_days must be a positive whole number, got "
            f"{params['period_days']!r}"
        )
    return out


class Glicko2:
    """Glicko-2 ratings for a population of players, advanced date by date."""

    def __init__(
        self,
        tau: float,
        initial_rating: float,
        initial_rd: float,
        initial_vol: float,
        period_days: int,
    ) -> None:
        params = check_params(
            {
                "tau": tau,
                "initial_rating": initial_rating,
                "initial_rd": initial_rd,
                "initial_vol": initial_vol,
                "period_days": period_days,
            }
        )
        self.tau = params["tau"]
        self.period_days = int(params["period_days"])
        self._initial: State = (
            (params["initial_rating"] - DISPLAY_CENTER) / SCALE,
            params["initial_rd"] / SCALE,
            params["initial_vol"],
        )
        self._states: dict[str, State] = {}
        self._last: dict[str, dt.date] = {}  # each player's last rated date
        self._clock: dt.date | None = None  # the last date update_date applied

    # --- display scale -----------------------------------------------------

    def set_state(self, player: str, rating: float, rd: float, vol: float) -> None:
        """Set a player's current state on the display scale. Their last rated date, if
        any, is unchanged; a player never rated by date is never RD-inflated."""
        self._states[player] = ((rating - DISPLAY_CENTER) / SCALE, rd / SCALE, vol)

    def rating(self, player: str) -> tuple[float, float, float]:
        """(rating, RD, volatility) on the display scale, as of the last update."""
        mu, phi, sigma = self._states.get(player, self._initial)
        return SCALE * mu + DISPLAY_CENTER, SCALE * phi, sigma

    def update_period(self, player: str, results: Sequence[tuple[str, float]]) -> None:
        """Glickman's procedure for one player over one rating period, against the
        opponents' current states. Opponents are not updated."""
        games = [(*self._states.get(opp, self._initial)[:2], score) for opp, score in results]
        self._states[player] = _update(self._states.get(player, self._initial), games, self.tau)

    # --- internal scale, by date -------------------------------------------

    def pre_fight(self, player: str, date: dt.date) -> State:
        """(mu, phi, sigma) on the internal scale as of the morning of ``date``, the RD
        inflated for every whole idle period since the player's last rated date.

        Raises:
            ValueError: if ``date`` is on or before that last rated date, whose state
                already holds that date's results.
        """
        mu, phi, sigma = self._states.get(player, self._initial)
        last = self._last.get(player)
        if last is None:
            return mu, phi, sigma
        if date <= last:
            raise ValueError(
                f"pre-fight state for {player!r} requested on {date}, on or before its last "
                f"rated date {last}: that state already holds the date's results"
            )
        idle = (date - last).days // self.period_days
        return mu, min(math.sqrt(phi * phi + idle * sigma * sigma), self._initial[1]), sigma

    def update_date(self, date: dt.date, results: Sequence[tuple[str, str, float]]) -> None:
        """Apply every result dated ``date`` at once, each participant from their
        pre-date state against the opponents' pre-date states.

        ``results`` holds ``(a, b, score_a)``. A player with several bouts gets one
        rating period holding all of them.

        Raises:
            ValueError: if ``date`` is not later than the previous update's date, a
                player meets themselves, or a score is outside [0, 1].
        """
        if self._clock is not None and date <= self._clock:
            raise ValueError(
                f"update_date dates must strictly increase: {date} after {self._clock}; a "
                "date's results are applied in one call so none of them sees another"
            )
        pre: dict[str, State] = {}
        games: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for a, b, score_a in results:
            if a == b:
                raise ValueError(f"{a!r} cannot fight themselves on {date}")
            if not 0.0 <= score_a <= 1.0:
                raise ValueError(f"score {score_a!r} for {a!r} v {b!r} on {date} not in [0, 1]")
            for player in (a, b):
                if player not in pre:
                    pre[player] = self.pre_fight(player, date)
            games[a].append((b, float(score_a)))
            games[b].append((a, 1.0 - float(score_a)))

        updated = {
            player: _update(
                pre[player], [(*pre[opp][:2], score) for opp, score in played], self.tau
            )
            for player, played in games.items()
        }
        for player, state in updated.items():
            self._states[player] = state
            self._last[player] = date
        self._clock = date


def build_glicko(fights: pd.DataFrame, params: Mapping[str, float]) -> pd.DataFrame:
    """Pre-fight Glicko-2 values for every fight.

    Args:
        fights: must contain fight_id, date, fighter_1, fighter_2, winner_id, is_decisive.
        params: the ``glicko2`` config block.

    Returns:
        fight_id and every ``GLICKO_COLS`` column except ``glicko_mu_diff``, which
        ``pit.py`` adds; all as of the morning of the fight's date.

    Raises:
        ValueError: on bad parameters, or a decisive bout whose winner is neither fighter.
    """
    engine = Glicko2(**check_params(params))
    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    rows: list[dict] = []
    for date, day in ordered.groupby("date", sort=True):
        # 1. Describe every fight on this date from the pre-date states.
        results: list[tuple[str, str, float]] = []
        for row in day.itertuples(index=False):
            mu_1, phi_1, _ = engine.pre_fight(row.fighter_1, date)
            mu_2, phi_2, _ = engine.pre_fight(row.fighter_2, date)
            rd_1, rd_2 = SCALE * phi_1, SCALE * phi_2
            rows.append(
                {
                    "fight_id": row.fight_id,
                    "glicko_mu_1": SCALE * mu_1 + DISPLAY_CENTER,
                    "glicko_mu_2": SCALE * mu_2 + DISPLAY_CENTER,
                    "glicko_rd_1": rd_1,
                    "glicko_rd_2": rd_2,
                    "glicko_expected_1": expected_score(mu_1, phi_1, mu_2, phi_2),
                    "glicko_rd_sum": rd_1 + rd_2,
                }
            )
            if not row.is_decisive or pd.isna(row.winner_id):
                continue  # draws and no contests carry no rating information here
            if row.winner_id not in (row.fighter_1, row.fighter_2):
                raise ValueError(
                    f"fight {row.fight_id}: winner {row.winner_id!r} is neither fighter"
                )
            results.append((row.fighter_1, row.fighter_2, float(row.winner_id == row.fighter_1)))
        # 2. Only now apply the date's results, all at once.
        if results:
            engine.update_date(date, results)

    out = pd.DataFrame(rows, columns=["fight_id", *RATED_COLS])
    if len(out) != len(fights):
        raise RuntimeError(f"glicko rated {len(out)} fights, expected {len(fights)}")
    return out
