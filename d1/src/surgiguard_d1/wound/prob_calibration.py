"""Make the classifier's probabilities mean what they say (v2 step 4). Not to be confused with calibration.py
(the camera calibration card).

Problem: the heads are logistic regressions trained with class_weight="balanced" (so rare positives are not ignored).
That re-weighting shifts every probability towards 0.5: "p = 0.7" does not mean 70 % of such photos are positive.
A clinician reading "70 %" would be misled, even though the RANKING of photos (AUROC) is fine.

Fix: post-hoc recalibration of the logit z = w.x + b, fitted on out-of-fold predictions (never on the test split):
* temperature scaling   p = sigmoid(a * z)          1 parameter: fixes over/under-confidence only;
* Platt scaling         p = sigmoid(a * z + c)      2 parameters: also fixes the shift caused by class weighting.
Both are monotone, so AUROC and the ranking cannot change; only the probability values do.

Measures: expected calibration error (ECE, 10 equal-width bins: mean |predicted - observed| weighted by bin size)
and the Brier score (mean squared error of the probability). numpy only, no scikit-learn.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_platt(z: np.ndarray, y: np.ndarray, with_bias: bool = True, iters: int = 100) -> tuple[float, float]:
    """Newton's method on the log loss of sigmoid(a*z + c) (with_bias=False: temperature, c = 0).

    A tiny L2 term (1e-6) keeps the Hessian invertible when a fold is perfectly separable."""
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
    theta = np.array([1.0, 0.0])
    X = np.stack([z, np.ones_like(z)], 1) if with_bias else z[:, None]
    theta = theta[: X.shape[1]]
    for _ in range(iters):
        p = _sigmoid(X @ theta)
        grad = X.T @ (p - y) / len(y) + 1e-6 * theta
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(theta[0]), float(theta[1]) if with_bias else 0.0


def apply(z: np.ndarray, a: float, c: float) -> np.ndarray:
    return _sigmoid(a * np.asarray(z, np.float64) + c)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.sum() * abs(p[sel].mean() - y[sel].mean())
    return float(total / len(p))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Per-bin mean prediction vs observed frequency (the data behind a reliability diagram)."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int((idx == b).sum()),
             "mean_predicted": round(float(p[idx == b].mean()), 3), "observed": round(float(y[idx == b].mean()), 3)}
            for b in range(bins) if (idx == b).any()]
