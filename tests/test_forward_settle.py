"""Settling a forward card: only the committed pair is ever scored."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.forward import settle
from ufc_predictor.forward.espn import Result


def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition_id": ["c1", "c2", "c3", "c4", "c5"],
            "fighter_a": [
                "Mike Malott",
                "Erin Blanchfield",
                "Kyle Nelson",
                "Nate Landwehr",
                "Tanner Boser",
            ],
            "fighter_b": [
                "Joaquin Buckley",
                "Jasmine Jasudavicius",
                "Cristian Pérez",
                "Mandel Nallo",
                "Jhonata Diniz",
            ],
            "p_model_a": [0.60, 0.70, 0.30, 0.40, 0.45],
            "kalshi_ticker_a": ["T1A", "T2A", "T3A", "T4A", "T5A"],
            "kalshi_ticker_b": ["T1B", "T2B", "T3B", "T4B", "T5B"],
            "kalshi_p_a": [0.55, 0.75, 0.25, 0.30, np.nan],
        }
    )


RESULTS = [
    Result("c1", ("Mike Malott", "Joaquin Buckley"), True, "Joaquin Buckley", "KO/TKO", 1),
    Result("c2", ("Jasmine Jasudavicius", "Erin Blanchfield"), True, "Erin Blanchfield", "DEC", 5),
    # c3: Pérez withdrew; ESPN kept the slot with a replacement opponent.
    Result("c3", ("Kyle Nelson", "Late Replacement"), True, "Kyle Nelson", "SUB", 2),
    Result("c4", ("Nate Landwehr", "Mandel Nallo"), True, None, None, 3),  # draw
    # c5 was cancelled: ESPN lists it as not completed.
    Result("c5", ("Tanner Boser", "Jhonata Diniz"), False, None, None, None),
]

QUOTES = {
    "T1A": (0.54, 0.56),
    "T1B": (0.44, 0.46),
    "T2A": (0.74, 0.76),
    "T2B": (0.25, 0.27),
    "T3A": (0.20, 0.22),
    "T3B": (0.78, 0.80),
    "T4A": None,
    "T4B": (0.69, 0.71),
    "T5A": None,
    "T5B": None,
}


def test_only_the_committed_pair_is_scored():
    out = settle.settle_bouts(predictions(), RESULTS, QUOTES).set_index("competition_id")
    assert out.loc["c1", "status"] == "scored" and out.loc["c1", "y_a"] == 0.0
    assert out.loc["c2", "status"] == "scored" and out.loc["c2", "y_a"] == 1.0  # names reordered
    assert out.loc["c3", "status"] == "void_replaced"
    assert out.loc["c4", "status"] == "void_draw_or_nc"
    assert out.loc["c5", "status"] == "void_not_held"
    assert out.loc[out["status"] != "scored", "y_a"].isna().all()


def test_a_bout_missing_from_the_results_is_not_held():
    out = settle.settle_bouts(predictions(), RESULTS[:2], QUOTES).set_index("competition_id")
    assert (out.loc[["c3", "c4", "c5"], "status"] == "void_not_held").all()


def test_close_price_is_the_normalized_mid_and_missing_stays_missing():
    out = settle.settle_bouts(predictions(), RESULTS, QUOTES).set_index("competition_id")
    assert out.loc["c1", "kalshi_close_p_a"] == pytest.approx(0.55 / (0.55 + 0.45))
    assert math.isnan(out.loc["c4", "kalshi_close_p_a"])  # one side unquoted: no 0.5 fill


def test_log_losses_use_the_outcome_of_fighter_a():
    out = settle.settle_bouts(predictions(), RESULTS, QUOTES).set_index("competition_id")
    assert out.loc["c1", "ll_model"] == pytest.approx(-math.log(0.40))
    assert out.loc["c2", "ll_model"] == pytest.approx(-math.log(0.70))
    assert out.loc["c1", "ll_kalshi_close"] == pytest.approx(-math.log(1 - 0.55))


def test_summary_counts_and_compares_on_common_bouts():
    out = settle.settle_bouts(predictions(), RESULTS, QUOTES)
    s = settle.summarize(out)
    assert s["n_predicted"] == 5 and s["n_scored"] == 2
    assert s["voids"] == {"void_draw_or_nc": 1, "void_not_held": 1, "void_replaced": 1}
    both = out.loc[out["status"] == "scored"]
    assert s["model_log_loss"] == pytest.approx(both["ll_model"].mean())
    assert s["n_with_kalshi_close"] == 2
    assert s["model_minus_kalshi_close"] == pytest.approx(
        (both["ll_model"] - both["ll_kalshi_close"]).mean()
    )
