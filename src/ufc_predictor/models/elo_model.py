"""Elo as a probability model.

The raw Elo expectation ``1 / (1 + 10 ** (-gap / scale))`` is a reasonable forecast but
its sharpness is arbitrary -- it depends on the K-factor and scale chosen when the
ratings were built, not on how predictive the rating gap actually turns out to be. So
instead of using that curve directly, a one-feature logistic regression is fitted on the
rating gap, which lets the data decide how much a 100-point edge is worth.

That single coefficient is the only thing learned here, and it is learned on the
training slice the backtest supplies -- never on the evaluation period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ufc_predictor.models.base import Model

ELO_DIFF_COL = "elo_diff"


class EloModel(Model):
    """Logistic map from pre-fight Elo difference to P(fighter_1 wins)."""

    name = "elo"

    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self._clf: LogisticRegression | None = None
        #: Fallback used only if a fit is impossible (single-class training slice).
        self._scale = 400.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EloModel:
        gap = self._gap(X)
        target = np.asarray(y, dtype=int)

        if len(np.unique(target)) < 2:
            # A degenerate slice cannot fit a logistic. Fall back to the analytic Elo
            # curve rather than silently predicting a constant.
            self._clf = None
            return self

        # `penalty=` is deprecated in scikit-learn 1.8; L2 is the default and its
        # strength is set by C, so the argument is simply omitted.
        self._clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.seed,
        )
        self._clf.fit(gap, target)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        gap = self._gap(X)
        if self._clf is None:
            # Analytic Elo expectation.
            p = 1.0 / (1.0 + 10.0 ** (-gap[:, 0] / self._scale))
            return self._clip(p)
        return self._clip(self._clf.predict_proba(gap)[:, 1])

    @staticmethod
    def _gap(X: pd.DataFrame) -> np.ndarray:
        if ELO_DIFF_COL not in X.columns:
            raise KeyError(f"{ELO_DIFF_COL!r} is missing from the feature frame")
        # Every fight has an Elo difference: debutants start at the configured initial
        # rating, so the gap is defined even with no history. No imputation needed.
        gap = X[[ELO_DIFF_COL]].to_numpy(dtype=float)
        if np.isnan(gap).any():
            raise ValueError("elo_diff contains NaN; every fight should have a pre-fight rating")
        return gap
