"""LightGBM on the point-in-time feature set, with time-ordered validation.

Three choices here exist to keep the eventual comparison against the market honest.

**No random cross-validation, anywhere.** Random K-fold would place future fights in the
training folds of past ones. Even though each *feature* is point-in-time correct, the
fitted model would have seen the era it is predicting, and the validation score would be
optimistic in a way that does not survive deployment. All validation is a chronological
holdout: the most recent slice of the training data, never a random subset.

**Early stopping on that chronological holdout.** The number of trees is the single most
overfit-prone hyperparameter, so it is not fixed by hand. The tail of the training window
selects it, and the selected count is recorded so the backtest can report it.

**Calibration fitted on a chronological holdout too.** Gradient boosting with log loss is
reasonably calibrated already, but not perfectly, and log loss and Brier both punish
miscalibration directly. Isotonic regression is fitted on the holdout tail -- never on
the data the trees were fitted to, which would be circular, and never on the evaluation
period. Its output is bounded to ``calibration_bounds``: isotonic regression is a step
function, and a step holding only wins or only losses would otherwise claim certainty --
evaluation run 2 had 51 such predictions, and each miss cost 13.8 nats.

Every seed comes from config; nothing in this module hardcodes one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufc_predictor.features.pit import FEATURE_COLS
from ufc_predictor.models.base import Model
from ufc_predictor.models.calibration import Calibrator, build_calibrator


class LightGBMModel(Model):
    """Gradient-boosted trees with chronological early stopping and calibration."""

    name = "lightgbm"

    def __init__(
        self,
        *,
        seed: int,
        params: dict | None = None,
        feature_cols: list[str] | None = None,
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = 50,
        calibrate: bool = True,
        min_validation_rows: int = 300,
        calibration_bounds: tuple[float, float] = (0.01, 0.99),
        calibration: str = "isotonic_bounded",
    ) -> None:
        lo, hi = calibration_bounds
        if not 0 < lo < 0.5 < hi < 1:
            raise ValueError(
                f"calibration_bounds {calibration_bounds} must satisfy 0 < lo < 0.5 < hi < 1"
            )
        self.seed = seed
        self.params = dict(params or {})
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.calibrate = calibrate
        self.min_validation_rows = min_validation_rows
        self.calibration_bounds = (float(lo), float(hi))
        # Phase 8 compares calibrators; "isotonic_bounded" is evaluation run 3's.
        build_calibrator(calibration, self.calibration_bounds)  # validates the name
        self.calibration = calibration

        self._booster = None
        self._calibrator: Calibrator | None = None
        #: Trees actually used after early stopping, for reporting.
        self.best_iteration_: int | None = None

    # -- internals ---------------------------------------------------------

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"feature frame is missing {len(missing)} columns, e.g. {missing[:5]}")
        return X.loc[:, self.feature_cols]

    @staticmethod
    def _assert_time_ordered(X: pd.DataFrame) -> None:
        """The chronological split is only meaningful if rows arrive in date order."""
        if "date" not in X.columns:
            raise KeyError(
                "LightGBMModel.fit needs a 'date' column to build a time-ordered "
                "validation split; random validation is not permitted here"
            )
        dates = X["date"].to_numpy()
        if len(dates) > 1 and not all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)):
            raise ValueError(
                "training rows are not sorted by date; a chronological holdout would "
                "silently mix eras"
            )

    # -- interface ---------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMModel:
        import lightgbm as lgb

        self._assert_time_ordered(X)
        target = np.asarray(y, dtype=int)
        n = len(X)

        params = dict(self.params)
        params.update({"random_state": self.seed, "verbosity": -1, "deterministic": True})
        n_estimators = int(params.pop("n_estimators", 400))

        n_val = round(n * self.validation_fraction)
        if n_val < self.min_validation_rows or n - n_val < self.min_validation_rows:
            # Fail loudly. The alternative -- fitting the full n_estimators ceiling with
            # no early stopping and no calibration -- is silently a different, much less
            # regularized model, and it would be reported in the same column as the
            # properly validated folds. The caller should raise min_train_fights instead.
            raise ValueError(
                f"training slice of {n} rows yields a {n_val}-row validation tail, below "
                f"min_validation_rows={self.min_validation_rows}. Refusing to fit without "
                "a chronological holdout, because that would mean no early stopping and "
                "no calibration while still being reported as the same model."
            )

        # Chronological: the validation slice is the most RECENT part of training.
        split = n - n_val
        X_tr, y_tr = self._matrix(X.iloc[:split]), target[:split]
        X_va, y_va = self._matrix(X.iloc[split:]), target[split:]

        if len(np.unique(y_tr)) < 2:
            raise ValueError("training slice contains a single outcome class")

        if len(np.unique(y_va)) < 2:
            raise ValueError(
                "the chronological validation tail contains a single outcome class, so "
                "early stopping and calibration are impossible"
            )

        model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        # eval_X / eval_y, not the deprecated eval_set (LightGBM >= 4.7).
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        self.best_iteration_ = int(getattr(model, "best_iteration_", n_estimators) or n_estimators)

        self._booster = model

        # Calibrate on the held-out tail only. Fitting isotonic on the trees' own
        # training rows would be circular and would look better than it is.
        self._calibrator = None
        if self.calibrate:
            calibrator = build_calibrator(self.calibration, self.calibration_bounds)
            if calibrator is not None:
                self._calibrator = calibrator.fit(model.predict_proba(X_va)[:, 1], y_va)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("model is not fitted")
        raw = self._booster.predict_proba(self._matrix(X))[:, 1]
        if self._calibrator is not None:
            raw = self._calibrator.predict(raw)
        return self._clip(raw)
