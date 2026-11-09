"""Portable model bundle (v2 step 1b): numpy-only heads that give the same numbers as the scikit-learn pickle."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from surgiguard_d1.wound import model_bundle as mb

MODELS = Path(__file__).resolve().parents[1] / "models"


def _toy_head(threshold=0.5, band=0.05):
    return mb.LinearHead("toy", mean=np.zeros(3), scale=np.ones(3), coef=np.array([1.0, 0.0, 0.0]), intercept=0.0,
                         threshold=threshold, band=band)


def test_logistic_head_is_plain_arithmetic():
    h = _toy_head()
    assert h.probability(np.array([[0.0, 5, 5]]))[0] == pytest.approx(0.5)
    assert h.probability(np.array([[2.0, 0, 0]]))[0] == pytest.approx(1 / (1 + np.exp(-2)))


def test_abstains_inside_the_band_and_on_unfamiliar_photos():
    h = _toy_head(threshold=0.3, band=0.05)
    assert h.decide(0.32) == mb.UNCERTAIN          # |p - thr| < band
    assert h.decide(0.40) == mb.YES and h.decide(0.20) == mb.NO
    assert h.decide(0.90, familiar=False) == mb.UNCERTAIN


def test_round_trip_through_json_and_npz(tmp_path):
    ood = mb.OODGate(np.eye(3, dtype=np.float16), k=1, threshold=0.2, info={"test_flagged_pct": 2.2})
    b = mb.ModelBundle("v", "enc", "data", {"toy": _toy_head()}, ood, {"trained_at": "now"})
    manifest = b.save(tmp_path / "heads.json")
    again = mb.load(manifest)
    x = np.random.default_rng(0).normal(size=(10, 3))
    assert np.array_equal(again.heads["toy"].probability(x), b.heads["toy"].probability(x))
    assert again.ood.familiar(np.array([1.0, 0, 0]))[0] and not again.ood.familiar(np.array([1.0, 1, 1]))[0]
    assert json.loads(manifest.read_text())["format_version"] == mb.FORMAT_VERSION


def test_refuses_an_unknown_format_version(tmp_path):
    manifest = mb.ModelBundle("v", "enc", "data", {"toy": _toy_head()}).save(tmp_path / "heads.json")
    data = json.loads(manifest.read_text())
    data["format_version"] = 99
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="format_version"):
        mb.load(manifest)


def test_shipped_portable_bundle_matches_the_legacy_pickle():
    """The exported files give the same probabilities as scikit-learn on real training features."""
    legacy_path, portable_path = MODELS / "wound_heads.joblib", MODELS / "wound_heads.json"
    if not (legacy_path.exists() and portable_path.exists()):
        pytest.skip("model files not present")
    joblib = pytest.importorskip("joblib")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = joblib.load(legacy_path)
    portable = mb.load(portable_path)
    x = np.asarray(legacy["ood"]["reference"], np.float64)[:100]
    for name, h in legacy["heads"].items():
        p_sklearn = h["model"].predict_proba(h["scaler"].transform(x))[:, 1]
        assert np.abs(portable.heads[name].probability(x) - p_sklearn).max() < 1e-9
        assert portable.heads[name].threshold == pytest.approx(h["threshold"])
    assert portable.ood.k == legacy["ood"]["k"] and portable.encoder == legacy["encoder"]


def test_portable_bundle_loads_without_scikit_learn(monkeypatch):
    """Inference must not need scikit-learn: block it and load the shipped bundle."""
    import builtins
    if not (MODELS / "wound_heads.json").exists():
        pytest.skip("portable bundle not exported")
    real_import = builtins.__import__

    def guarded(name, *a, **k):
        if name.startswith(("sklearn", "joblib")):
            raise ImportError(f"{name} blocked by the test")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)
    b = mb.load(MODELS / "wound_heads.json")
    assert set(b.heads) == {"erythema", "edema", "exudate", "not_healed", "elevated_risk", "needs_review"}
