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
period.

Every seed comes from config; nothing in this module hardcodes one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ufc_predictor.features.pit import FEATURE_COLS
from ufc_predictor.models.base import Model


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
    ) -> None:
        self.seed = seed
        self.params = dict(params or {})
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.calibrate = calibrate
        self.min_validation_rows = min_validation_rows

        self._booster = None
        self._calibrator: IsotonicRegression | None = None
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
        use_holdout = n_val >= self.min_validation_rows and n - n_val >= self.min_validation_rows

        if use_holdout:
            # Chronological: the validation slice is the most RECENT part of training.
            split = n - n_val
            X_tr, y_tr = self._matrix(X.iloc[:split]), target[:split]
            X_va, y_va = self._matrix(X.iloc[split:]), target[split:]
        else:
            # Too little data to hold out; fit on everything with a fixed tree count.
            X_tr, y_tr = self._matrix(X), target
            X_va = y_va = None

        if len(np.unique(y_tr)) < 2:
            raise ValueError("training slice contains a single outcome class")

        model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        if X_va is not None and len(np.unique(y_va)) == 2:
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
            self.best_iteration_ = int(
                getattr(model, "best_iteration_", n_estimators) or n_estimators
            )
        else:
            model.fit(X_tr, y_tr)
            self.best_iteration_ = n_estimators

        self._booster = model

        # Calibrate on the held-out tail only. Fitting isotonic on the trees' own
        # training rows would be circular and would look better than it is.
        self._calibrator = None
        if self.calibrate and X_va is not None and len(np.unique(y_va)) == 2:
            raw_va = model.predict_proba(X_va)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_va, y_va)
            self._calibrator = iso

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("model is not fitted")
        raw = self._booster.predict_proba(self._matrix(X))[:, 1]
        if self._calibrator is not None:
            raw = self._calibrator.predict(raw)
        return self._clip(raw)
