"""The market baseline: the vig-free closing line.

This is the benchmark the project exists to measure against, and it is deliberately not
a learned model. ``fit`` is a no-op because the prediction *is* the input: the closing
price, de-vigged by two-sided normalization, expressed in canonical fighter order.

Keeping it behind the same interface as the learned models matters for fairness. The
backtest cannot accidentally score the market on a different set of fights, apply a
different clipping rule, or give it a calibration step the others do not get.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufc_predictor.models.base import Model

MARKET_COL = "p1_market"


class MarketModel(Model):
    """Predicts the de-vigged closing probability for fighter_1."""

    name = "market"
    requires_fit = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MarketModel:
        """No-op: the closing line is an observation, not something to learn."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if MARKET_COL not in X.columns:
            raise KeyError(
                f"{MARKET_COL!r} is missing. The market baseline can only be scored on "
                "fights that carry a closing line; the backtest must restrict every "
                "model to the same priced subset."
            )
        p = X[MARKET_COL].to_numpy(dtype=float)
        if np.isnan(p).any():
            n = int(np.isnan(p).sum())
            raise ValueError(
                f"market baseline received {n} rows with no closing price. Unpriced "
                "fights must be excluded before scoring, not imputed."
            )
        return self._clip(p)
