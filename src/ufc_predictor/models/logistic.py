"""A regularized logistic regression benchmark for the Phase 8 model stage.

The question it answers: does LightGBM's flexibility buy anything over a linear model
on the same point-in-time features? Everything is fitted on the training rows of the
fold and nothing else:

* **missing history** -- median imputation plus a missing-indicator column per feature
  that has gaps, because "no history" is information and must not become a fake value;
* **scale** -- standardization, so one L2 penalty treats every feature alike;
* **the penalty** -- ``C`` is chosen from a small fixed grid by log loss on the
  chronological tail of the training rows (ties go to the smaller ``C``, the stronger
  penalty), then the model is refitted on all training rows with that ``C``.

The model's own probabilities are used; no extra calibrator is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from ufc_predictor.models.base import Model


class LogisticModel(Model):
    name = "logistic"

    def __init__(
        self,
        *,
        features: list[str],
        seed: int,
        C_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
        validation_fraction: float = 0.2,
    ) -> None:
        if not C_grid:
            raise ValueError("C_grid must hold at least one value")
        self.features = list(features)
        self.seed = seed
        self.C_grid = tuple(float(c) for c in C_grid)
        self.validation_fraction = validation_fraction
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.clf_: LogisticRegression | None = None
        self.C_: float | None = None
        self.validation_scores_: dict[float, float] = {}

    def _fit_once(self, X: pd.DataFrame, y: np.ndarray, C: float):
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        scaler = StandardScaler()
        Z = scaler.fit_transform(imputer.fit_transform(X.loc[:, self.features].astype(float)))
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=self.seed)
        clf.fit(Z, y)
        return imputer, scaler, clf

    @staticmethod
    def _predict(parts, X: pd.DataFrame, features: list[str]) -> np.ndarray:
        imputer, scaler, clf = parts
        Z = scaler.transform(imputer.transform(X.loc[:, features].astype(float)))
        return clf.predict_proba(Z)[:, 1]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticModel:
        dates = X["date"].to_numpy()
        if len(dates) > 1 and any(dates[i] > dates[i + 1] for i in range(len(dates) - 1)):
            raise ValueError("training rows are not sorted by date; the C holdout must be the tail")
        target = np.asarray(y, dtype=int)

        self.validation_scores_ = {}
        if len(self.C_grid) == 1:
            self.C_ = self.C_grid[0]
        else:
            split = len(X) - round(len(X) * self.validation_fraction)
            head, tail = X.iloc[:split], X.iloc[split:]
            for C in self.C_grid:
                parts = self._fit_once(head, target[:split], C)
                p = self._predict(parts, tail, self.features)
                self.validation_scores_[C] = float(log_loss(target[split:], p, labels=[0, 1]))
            best = min(self.validation_scores_.values())
            self.C_ = min(C for C, s in self.validation_scores_.items() if s == best)

        self.imputer_, self.scaler_, self.clf_ = self._fit_once(X, target, self.C_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("model is not fitted")
        return self._clip(self._predict((self.imputer_, self.scaler_, self.clf_), X, self.features))
