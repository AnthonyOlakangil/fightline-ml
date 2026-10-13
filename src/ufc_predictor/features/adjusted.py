"""Opponent-adjusted and recency-weighted rate features (the "adjusted" and "recency" families).

A raw career rate does not say whom it was earned against: four significant strikes a
minute against opponents who usually concede three is better than four against opponents
who usually concede six. And a career mean weights a bout from six years ago exactly as
much as last month's. Both families answer that from the same per-bout record.

* ``adj_<metric>_career`` -- the mean, over a fighter F's prior bouts, of
  ``resid_b = F's rate in bout b - opponent O's pre-bout career quantity``, where the
  quantity is O's mirror of the metric: what O's own previous opponents had achieved
  against O (``MIRROR``). A bout is excluded when either side of the subtraction is NaN.
* ``ewm_<metric>`` -- an exponentially weighted mean of F's raw per-bout rate, most
  recent bout weight 1 and the k-th most recent ``2^(-k / half_life)``. Bouts whose rate
  is NaN are skipped and do not consume a step. Two bouts on one tournament night have
  no recorded order and are taken in ``fight_id`` order, which is deterministic.

**Point in time.** O's quantity is taken from O's state on the morning of bout b's date
and frozen into F's record of that bout, so nothing O does later -- nor anything else on
the same card, nor O's other bout on a tournament night -- can move it. ``pit.py`` owns
the per-date loop that guarantees this; this module only does the arithmetic.

Definitions and the half-life (``features.recency_half_life_fights``) were fixed a
priori, before any development-window experiment, not searched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

SLOTS = ("1", "2", "diff")

#: F's metric in a bout -> (O's pre-bout career quantity, whether to take 1 - it).
#: Keyed in ``pit.RATE_METRICS`` order (a test asserts the two never drift apart).
MIRROR: dict[str, tuple[str, bool]] = {
    "sig_str_landed_pm": ("sig_str_absorbed_pm", False),
    "sig_str_absorbed_pm": ("sig_str_landed_pm", False),
    "sig_str_acc": ("sig_str_def", True),  # 1 - def = accuracy O had allowed
    "sig_str_def": ("sig_str_acc", True),  # 1 - acc = defence O's opponents had shown
    "td_landed_p15": ("td_absorbed_p15", False),
    "td_acc": ("td_def", True),
    "td_def": ("td_acc", True),
    "sub_att_p15": ("sub_att_absorbed_p15", False),
    "kd_p15": ("kd_absorbed_p15", False),
    "ctrl_pm": ("ctrl_absorbed_pm", False),
}

ADJUSTED_COLS: list[str] = [f"adj_{m}_career_{s}" for m in MIRROR for s in SLOTS]
RECENCY_COLS: list[str] = [f"ewm_{m}_{s}" for m in MIRROR for s in SLOTS]


def absorbed_rates(records: Sequence[Mapping]) -> dict[str, float]:
    """Career rates of what a fighter's opponents did to them, as a ratio of sums.

    The four "absorbed" quantities ``pit._rate_block`` has no counterpart for:
    takedowns, submission attempts and knockdowns conceded per 15 minutes, and control
    time conceded per minute. Same denominator as ``_rate_block``: bouts with round
    statistics and a known duration. NaN when there are none, never 0.
    """
    with_stats = [r for r in records if r["has_stats"]]
    total_sec = sum(r["duration_sec"] for r in with_stats if pd.notna(r["duration_sec"]))
    minutes = total_sec / 60.0 if total_sec else 0.0
    if not with_stats or minutes <= 0:
        return dict.fromkeys(
            ("td_absorbed_p15", "sub_att_absorbed_p15", "kd_absorbed_p15", "ctrl_absorbed_pm"),
            np.nan,
        )

    def s(key: str) -> float:
        return float(sum(r[key] for r in with_stats if pd.notna(r.get(key))))

    return {
        "td_absorbed_p15": s("opp_td_landed") * 15.0 / minutes,
        "sub_att_absorbed_p15": s("opp_sub_att") * 15.0 / minutes,
        "kd_absorbed_p15": s("opp_kd") * 15.0 / minutes,
        "ctrl_absorbed_pm": s("opp_ctrl_sec") / minutes,
    }


def opponent_baseline(
    career: Mapping[str, float], absorbed: Mapping[str, float]
) -> dict[str, float]:
    """O's pre-bout quantity to subtract from each of F's metrics.

    ``career`` holds O's career rate metrics keyed by bare metric name; ``absorbed`` is
    ``absorbed_rates`` over the same bouts. Both must describe O before the bout's date.
    """
    out: dict[str, float] = {}
    for metric, (source, complement) in MIRROR.items():
        value = absorbed[source] if source in absorbed else career[source]
        out[metric] = 1.0 - value if complement else value
    return out


def family_snapshot(
    bouts: Sequence[tuple[Mapping[str, float], Mapping[str, float]]], half_life: float
) -> dict[str, float]:
    """The adjusted and recency values for one fighter, from their prior bouts.

    ``bouts`` is chronological: per prior bout, F's per-bout rates and O's frozen
    pre-bout baseline, both keyed by bare metric name.
    """
    # Plain Python rather than numpy: the lists are a handful of bouts long, and this
    # runs once per fighter per event date.
    decay = [2.0 ** (-k / half_life) for k in range(len(bouts))]
    out: dict[str, float] = {}
    for metric in MIRROR:
        resid_sum, resid_n = 0.0, 0
        num, den, k = 0.0, 0.0, 0
        for rates, base in reversed(bouts):  # most recent first
            rate = rates[metric]
            if math.isnan(rate):
                continue  # no rate: neither a residual nor a recency step
            num += decay[k] * rate
            den += decay[k]
            k += 1
            resid = rate - base[metric]
            if not math.isnan(resid):
                resid_sum += resid
                resid_n += 1
        out[f"adj_{metric}_career"] = resid_sum / resid_n if resid_n else np.nan
        out[f"ewm_{metric}"] = num / den if k else np.nan
    return out
