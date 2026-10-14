"""Decline signals (the "decline" family).

The base features say how good a fighter has been; these ask whether that is still true.
Six per-fighter values, each ``_1``, ``_2`` and ``_diff``:

* ``years_since_ufc_debut`` -- (event date - date of the fighter's first bout in the
  fights table) / 365.25. A debutant gets 0.0, a true value (they debut tonight), and
  ``is_debut`` says so.
* ``finished_losses_last3`` -- KO/TKO and submission losses among the last 3 bouts (fewer
  if the fighter has fewer).
* ``kd_absorbed_last3`` -- knockdowns absorbed over those of the last 3 bouts that
  recorded round statistics.
* ``absorbed_trend`` -- ``sig_str_absorbed_pm_w3 - sig_str_absorbed_pm_career``: being hit
  more lately than over the career.
* ``landed_trend`` -- ``sig_str_landed_pm_w3 - sig_str_landed_pm_career``.
* ``days_since_last_finish_loss`` -- days from the most recent KO/TKO or submission loss.

"Last 3 bouts" is the base ``w3`` rule: the fighter's last 3 bouts dated strictly before the
event. A draw or no contest counts as a bout but never as a loss. With nothing to count --
no bout, no bout with statistics, no finish loss -- the value is NaN, never 0. ``pit.py``
owns the per-date loop that keeps every input point-in-time; this module only does the
arithmetic.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ufc_predictor.features.adjusted import SLOTS

LAST_N = 3
DAYS_PER_YEAR = 365.25
FINISH_CLASSES = ("KO/TKO", "SUB")

DECLINE_METRICS = (
    "years_since_ufc_debut",
    "finished_losses_last3",
    "kd_absorbed_last3",
    "absorbed_trend",
    "landed_trend",
    "days_since_last_finish_loss",
)
DECLINE_COLS: list[str] = [f"{m}_{s}" for m in DECLINE_METRICS for s in SLOTS]


def is_finish_loss(rec: Mapping) -> bool:
    """A decisive loss by KO/TKO or submission. Draws and no contests have ``won`` NaN."""
    return rec["won"] == 0.0 and rec["method_class"] in FINISH_CLASSES


def decline_snapshot(
    last: Sequence[Mapping],
    recent: Mapping[str, float],
    career: Mapping[str, float],
    event_date: dt.date,
    first_date: dt.date | None,
    last_finish_loss: dt.date | None,
) -> dict[str, float]:
    """One fighter's decline values on the morning of ``event_date``.

    ``last`` is the fighter's last ``LAST_N`` history records; ``recent`` and ``career`` are
    their rate metrics over those bouts and over the career, keyed by bare metric name.
    ``first_date`` and ``last_finish_loss`` are the dates of the first bout and of the most
    recent finish loss, or None. Every input must describe bouts before ``event_date``.
    """
    kds = [r["opp_kd"] for r in last if r["has_stats"] and pd.notna(r.get("opp_kd"))]
    return {
        "years_since_ufc_debut": (
            (event_date - first_date).days / DAYS_PER_YEAR if first_date is not None else 0.0
        ),
        "finished_losses_last3": float(sum(is_finish_loss(r) for r in last)) if last else np.nan,
        "kd_absorbed_last3": float(sum(kds)) if kds else np.nan,
        "absorbed_trend": recent["sig_str_absorbed_pm"] - career["sig_str_absorbed_pm"],
        "landed_trend": recent["sig_str_landed_pm"] - career["sig_str_landed_pm"],
        "days_since_last_finish_loss": (
            float((event_date - last_finish_loss).days) if last_finish_loss is not None else np.nan
        ),
    }
