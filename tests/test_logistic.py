"""The L2 logistic benchmark: preprocessing and C chosen inside the training rows only."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.models.logistic import LogisticModel

DATES = [dt.date(2012, 1, 1) + dt.timedelta(days=7 * i) for i in range(400)]


def test_logistic_preprocessing_never_sees_scored_rows():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "date": DATES[:3], "label": [0, 1, 0]})
    scored = pd.DataFrame({"x": [100.0, np.nan], "date": DATES[3:5]})
    m = LogisticModel(features=["x"], C_grid=(1.0,), seed=0).fit(train, train["label"])
    assert m.imputer_.statistics_[0] == 2.0  # median of the training rows only
    assert m.scaler_.mean_[0] == pytest.approx(2.0)  # 100.0 never enters the fit
    m.predict_proba(scored)
    assert m.imputer_.statistics_[0] == 2.0  # prediction does not refit


def _synthetic(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    x_missing = x.copy()
    x_missing[rng.uniform(size=n) < 0.2] = np.nan
    return pd.DataFrame({"x": x_missing, "noise": noise, "date": DATES[:n], "label": y})


def test_logistic_learns_the_signal_and_chooses_c_on_the_chronological_tail():
    df = _synthetic()
    m = LogisticModel(
        features=["x", "noise"], C_grid=(0.01, 0.1, 1.0), seed=0, validation_fraction=0.25
    )
    m.fit(df, df["label"])
    assert m.C_ in (0.01, 0.1, 1.0)
    assert set(m.validation_scores_) == {0.01, 0.1, 1.0}
    p = m.predict_proba(df)
    assert p.shape == (len(df),) and np.isfinite(p).all()
    assert np.corrcoef(p, df["x"].fillna(0))[0, 1] > 0.8


def test_logistic_refuses_unsorted_training_rows():
    df = _synthetic().iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        LogisticModel(features=["x"], C_grid=(1.0,), seed=0).fit(df, df["label"])


def test_logistic_handles_a_column_missing_from_every_training_row():
    df = _synthetic()
    df["empty"] = np.nan
    m = LogisticModel(features=["x", "empty"], C_grid=(1.0,), seed=0).fit(df, df["label"])
    assert np.isfinite(m.predict_proba(df)).all()
