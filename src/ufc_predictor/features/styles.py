"""Style-matchup interactions (the "style" family).

Every base rate describes one fighter alone. Whether a wrestler's takedowns matter depends
on whom he is shooting on: six takedowns per 15 minutes against a fighter who stops 80% of
attempts is a different threat from six against one who stops half. These columns cross
one fighter's offence with the other's defence or durability.

* ``td_threat_1 = td_landed_p15_career_1 * (1 - td_def_career_2)``, and mirrored.
* ``strike_exchange_1 = sig_str_landed_pm_career_1 * (1 - sig_str_def_career_2)``, and
  mirrored.
* ``ko_power_vs_chin_1 = kd_p15_career_1 * kd_absorbed_p15_career_2``, and mirrored. The
  second factor is the opponent's career knockdowns absorbed per 15 minutes, from
  ``adjusted.absorbed_rates``.
* ``grappler_vs_striker = (td_landed_p15_career_1 - td_landed_p15_career_2) *
  (sig_str_landed_pm_career_2 - sig_str_landed_pm_career_1)``: large when one side is the
  better wrestler and the other the busier striker. Swapping the sides flips the sign of
  both differences, so the product is orientation-invariant and has no ``_1``/``_2``.

Every input is a pre-fight career quantity, so the interactions inherit the base rates'
point-in-time guarantee. Any NaN input gives NaN: a fighter with no statistical history has
no style to cross with anything. ``pit.py`` adds the ``_diff`` columns.
"""

from __future__ import annotations

from collections.abc import Mapping

STYLE_COLS: list[str] = [
    "td_threat_1",
    "td_threat_2",
    "td_threat_diff",
    "strike_exchange_1",
    "strike_exchange_2",
    "strike_exchange_diff",
    "grappler_vs_striker",
    "ko_power_vs_chin_1",
    "ko_power_vs_chin_2",
    "ko_power_vs_chin_diff",
]


def matchup(one: Mapping[str, float], two: Mapping[str, float]) -> dict[str, float]:
    """The interactions for one bout, without the ``_diff`` columns.

    ``one`` and ``two`` hold each side's pre-fight career rates keyed by bare metric name,
    plus ``kd_absorbed_p15``.
    """
    out: dict[str, float] = {}
    for slot, me, opp in (("1", one, two), ("2", two, one)):
        out[f"td_threat_{slot}"] = me["td_landed_p15"] * (1.0 - opp["td_def"])
        out[f"strike_exchange_{slot}"] = me["sig_str_landed_pm"] * (1.0 - opp["sig_str_def"])
        out[f"ko_power_vs_chin_{slot}"] = me["kd_p15"] * opp["kd_absorbed_p15"]
    out["grappler_vs_striker"] = (one["td_landed_p15"] - two["td_landed_p15"]) * (
        two["sig_str_landed_pm"] - one["sig_str_landed_pm"]
    )
    return out
