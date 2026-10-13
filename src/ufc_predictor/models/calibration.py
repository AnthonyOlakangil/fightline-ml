"""Probability calibrators, fitted on a chronological tail of the training rows.

Two choices behind one interface, so the Phase 8 calibration stage compares them on
equal terms:

* :class:`BoundedIsotonic` -- the evaluation-run-3 calibrator: isotonic regression with
  its output bounded to ``[lo, hi]``, so a step holding only wins or only losses cannot
  claim certainty.
* :class:`Platt` -- a logistic fit on the logit of the raw score, with Platt's (1999)
  smoothed targets: positives are fitted to ``(N+ + 1) / (N+ + 2)`` and negatives to
  ``1 / (N- + 2)`` rather than to 1 and 0. Without the smoothing, a tail the raw score
  separates perfectly drives the slope to infinity and the output to certainty; with
  it, no output can pass those targets.

``"none"`` returns no calibrator; the model's raw probability is used as is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATIONS = ("isotonic_bounded", "platt", "none")


class Calibrator(ABC):
    @abstractmethod
    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        """Fit on raw scores and 0/1 outcomes from the chronological tail."""

    @abstractmethod
    def predict(self, raw: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities."""


class BoundedIsotonic(Calibrator):
    """Isotonic regression clipped to ``[lo, hi]`` (evaluation run 3)."""

    def __init__(self, lo: float, hi: float) -> None:
        if not 0 < lo < 0.5 < hi < 1:
            raise ValueError(f"bounds ({lo}, {hi}) must satisfy 0 < lo < 0.5 < hi < 1")
        self.lo, self.hi = float(lo), float(hi)
        self._iso: IsotonicRegression | None = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> BoundedIsotonic:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=self.lo, y_max=self.hi)
        self._iso.fit(np.asarray(raw, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("calibrator is not fitted")
        return self._iso.predict(np.asarray(raw, dtype=float))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt(Calibrator):
    """Platt scaling with smoothed targets (see the module docstring)."""

    def __init__(self) -> None:
        self._clf: LogisticRegression | None = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Platt:
        y = np.asarray(y, dtype=float)
        n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
        target = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        z = _logit(raw).reshape(-1, 1)
        # A soft target t is fitted as one row labelled 1 with weight t and one labelled
        # 0 with weight 1 - t. C is large: the smoothing, not a penalty, bounds the fit.
        X = np.vstack([z, z])
        labels = np.r_[np.ones(len(z)), np.zeros(len(z))]
        weights = np.r_[target, 1 - target]
        self._clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._clf.fit(X, labels, sample_weight=weights)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("calibrator is not fitted")
        return self._clf.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


def build_calibrator(name: str, bounds: tuple[float, float]) -> Calibrator | None:
    """A fresh, unfitted calibrator by name, or None for ``"none"``."""
    if name == "isotonic_bounded":
        return BoundedIsotonic(*bounds)
    if name == "platt":
        return Platt()
    if name == "none":
        return None
    raise ValueError(f"unknown calibration {name!r}; expected one of {CALIBRATIONS}")
