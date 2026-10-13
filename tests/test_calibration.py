"""Calibrators fitted on a chronological tail: Platt and the bounded isotonic of run 3."""

from __future__ import annotations

import numpy as np
import pytest

from ufc_predictor.models.calibration import BoundedIsotonic, Platt, build_calibrator


def test_platt_never_outputs_a_certainty():
    raw = np.r_[np.full(50, 0.001), np.full(50, 0.999)]
    y = np.r_[np.zeros(50), np.ones(50)]
    p = Platt().fit(raw, y).predict(raw)
    assert ((p > 1e-5) & (p < 1 - 1e-5)).all()


def test_platt_is_monotone_and_recovers_a_known_miscalibration():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=true.size) < true).astype(float)
    logit = np.log(true / (1 - true))
    overconfident = 1 / (1 + np.exp(-2.0 * logit))  # slope 2 on the logit scale
    cal = Platt().fit(overconfident, y)
    grid = np.linspace(0.02, 0.98, 50)
    out = cal.predict(grid)
    assert np.all(np.diff(out) > 0)
    np.testing.assert_allclose(cal.predict(overconfident), true, atol=0.03)


def test_bounded_isotonic_never_leaves_its_bounds():
    raw = np.linspace(0, 1, 200)
    y = (raw > 0.5).astype(float)  # perfectly separable: plain isotonic would output 0 and 1
    p = BoundedIsotonic(0.01, 0.99).fit(raw, y).predict(raw)
    assert p.min() == pytest.approx(0.01) and p.max() == pytest.approx(0.99)


def test_build_calibrator_by_name():
    assert isinstance(build_calibrator("platt", (0.01, 0.99)), Platt)
    assert isinstance(build_calibrator("isotonic_bounded", (0.01, 0.99)), BoundedIsotonic)
    assert build_calibrator("none", (0.01, 0.99)) is None
    with pytest.raises(ValueError, match="calibration"):
        build_calibrator("beta", (0.01, 0.99))


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="fitted"):
        Platt().predict(np.array([0.5]))
