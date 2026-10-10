"""The shared model interface.

Every model -- market baseline, Elo, LightGBM -- exposes the same two methods, so the
walk-forward backtest treats them identically and cannot accidentally give one of them a
different fitting regime or a different scoring set. That symmetry is what makes the
comparison against the market fair.

``predict_proba`` returns P(fighter_1 wins) for each row, where ``fighter_1`` is the
canonical (sorted-id) first fighter -- never a corner. Probabilities are clipped away
from 0 and 1 so log loss stays finite; an overconfident model should be penalised, not
produce an infinity that breaks the metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Log loss is unbounded at 0 and 1. Clipping at 1e-6 caps the per-fight penalty near
# 13.8 nats, which is severe but finite.
PROB_EPS = 1e-6


class Model(ABC):
    """Base class for every model in the backtest.

    Subclasses must be fittable repeatedly: the walk-forward engine calls ``fit`` once
    per retraining step with a strictly earlier slice of history, and a model that
    accumulated state across calls would leak across folds.
    """

    #: Human-readable name used in reports and metrics.json keys.
    name: str = "model"

    #: Whether this model needs ``fit`` at all. The market baseline does not: its
    #: prediction is the de-vigged closing line, which is an input, not a learned thing.
    requires_fit: bool = True

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model:
        """Fit on a training slice. Must be safe to call repeatedly."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fighter_1 wins) as a 1-D array aligned with ``X``'s rows."""

    @staticmethod
    def _clip(p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), PROB_EPS, 1.0 - PROB_EPS)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


def validate_predictions(p: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    """Check a model's output before it reaches a metric.

    A silent shape mismatch or a stray NaN would otherwise surface as a mysteriously
    bad score rather than as an error.
    """
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name}: predict_proba must return a 1-D array, got shape {arr.shape}")
    if arr.shape[0] != n_rows:
        raise ValueError(f"{name}: returned {arr.shape[0]} predictions for {n_rows} rows")
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name}: returned {n_bad} non-finite probabilities")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{name}: returned probabilities outside [0, 1]")
    return arr
