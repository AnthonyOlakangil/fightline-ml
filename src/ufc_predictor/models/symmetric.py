"""Orientation symmetry: the probability that A beats B is one minus that of B beating A.

Each fight is stored once, with ``fighter_1`` the lower canonical id, so nothing forces
a model to give the same answer when the two sides are swapped. A tree model happily
learns slightly different splits for ``elo_1`` and ``elo_2``, and its two orientations
of one bout can disagree.

:class:`SymmetricModel` removes that freedom. It trains its inner model on every row and
on that row's **mirror** (sides swapped, label flipped), and predicts the average of the
two orientations:

    p(x) = (f(x) + 1 - f(mirror(x))) / 2

so ``p(x) + p(mirror(x)) = 1`` holds exactly, whatever the inner model.

:func:`mirror` swaps every ``_1``/``_2`` pair, negates every ``_diff``, maps the
one-sided probabilities in :data:`COMPLEMENT` to one minus themselves, and leaves the
columns in :data:`INVARIANT` alone. Any other column raises, so a new feature cannot
reach a symmetric model without a declared orientation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufc_predictor.models.base import Model

#: Columns that describe the bout rather than a side.
INVARIANT = frozenset(
    {
        "fight_id",
        "event_id",
        "date",
        "is_decisive",
        "is_priced",
        "scorable",
        "title_bout",
        "scheduled_rounds",
        "glicko_rd_sum",
        "grappler_vs_striker",
    }
)

#: Probabilities stated for fighter_1 only: the mirror is one minus the value.
COMPLEMENT = frozenset({"label", "glicko_expected_1", "p1_market"})


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    """The same bouts with the two sides swapped.

    Raises:
        KeyError: for a column with no declared orientation.
    """
    out = {}
    for col in X.columns:
        if col in INVARIANT:
            out[col] = X[col]
        elif col in COMPLEMENT:
            out[col] = 1.0 - X[col]
        elif col.endswith("_diff"):
            out[col] = -X[col]
        elif col.endswith("_1") and f"{col[:-2]}_2" in X.columns:
            out[col] = X[f"{col[:-2]}_2"]
        elif col.endswith("_2") and f"{col[:-2]}_1" in X.columns:
            out[col] = X[f"{col[:-2]}_1"]
        else:
            raise KeyError(f"no declared orientation for column {col!r}; cannot mirror it")
    return pd.DataFrame(out, index=X.index)[list(X.columns)]


def _feature_list(model: Model) -> list[str]:
    for attr in ("feature_cols", "features"):
        if hasattr(model, attr):
            return list(getattr(model, attr))
    raise TypeError(f"{type(model).__name__} does not expose its feature columns")


class SymmetricModel(Model):
    """Train on rows plus mirrors; predict the average of both orientations."""

    def __init__(self, inner: Model) -> None:
        self.inner = inner
        self.name = f"symmetric_{inner.name}"
        self.n_training_rows_: int | None = None

    def _view(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, ["date", *_feature_list(self.inner)]]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SymmetricModel:
        view = self._view(X)
        y = np.asarray(y, dtype=float)
        both = pd.concat(
            [view.assign(_y=y, _m=0), mirror(view).assign(_y=1.0 - y, _m=1)], ignore_index=True
        )
        # Date order is what the inner model's chronological holdout relies on.
        both = both.sort_values(["date", "_m"], kind="stable").reset_index(drop=True)
        self.n_training_rows_ = len(both)
        self.inner.fit(both.drop(columns=["_y", "_m"]), both["_y"].astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        view = self._view(X)
        p = self.inner.predict_proba(view)
        q = self.inner.predict_proba(mirror(view))
        return (p + 1.0 - q) / 2.0
