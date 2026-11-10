"""Probability calibration (v2 step 4): exact recovery on synthetic data, monotone, measures."""

import numpy as np
import pytest

from surgiguard_d1.wound.prob_calibration import apply, brier, ece, fit_platt, log_loss


def _data(a_true=2.0, c_true=-1.0, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(a_true * z + c_true)))).astype(float)
    return z, y


def test_platt_recovers_the_true_slope_and_shift():
    z, y = _data()
    a, c = fit_platt(z, y)
    assert a == pytest.approx(2.0, abs=0.1) and c == pytest.approx(-1.0, abs=0.1)


def test_temperature_has_no_shift_and_cannot_fix_a_biased_model():
    z, y = _data(c_true=-1.0)
    a, c = fit_platt(z, y, with_bias=False)
    assert c == 0.0
    assert ece(apply(z, *fit_platt(z, y)), y) < ece(apply(z, a, c), y)


def test_calibration_is_monotone_so_the_ranking_is_unchanged():
    z, y = _data()
    p = apply(z, *fit_platt(z, y))
    assert np.array_equal(np.argsort(z), np.argsort(p))


def test_measures_on_known_cases():
    y = np.array([0, 0, 1, 1], float)
    assert ece(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.0
    assert ece(np.full(4, 0.9), y) == pytest.approx(0.4)          # predicts 0.9, observed 0.5
    assert brier(np.full(4, 0.5), y) == pytest.approx(0.25)


def test_calibrating_a_shifted_model_lowers_ece_and_log_loss():
    z, y = _data(a_true=1.0, c_true=-1.5)
    raw = apply(z, 1.0, 0.0)                                      # the model's own probabilities, wrongly centred
    cal = apply(z, *fit_platt(z, y))
    assert ece(cal, y) < ece(raw, y) and log_loss(cal, y) < log_loss(raw, y)
