"""Portable, versioned storage and numpy-only inference for the wound-attribute heads.

Problem it solves (journal V2): the heads were a scikit-learn pickle. A pickle stores the private attributes of
the classes that wrote it, so loading it with another scikit-learn version broke the model (AttributeError).

A logistic-regression head is only arithmetic:   p = sigmoid( w . (x - mean) / scale + b )
so the bundle stores those numbers as plain arrays (`.npz`) plus a JSON manifest (`format_version`, encoder,
thresholds, metrics). Inference then needs numpy only: no scikit-learn, no pickle, no arbitrary code execution
when loading a file, and the same numbers on every machine.

Patterns: value objects (`LinearHead`, `OODGate`, `ModelBundle`, all immutable); Factory Method (`load()` picks
the reader from the file type); Adapter (`from_sklearn_dict()` turns the legacy pickle into the portable form).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
YES, NO, UNCERTAIN = "yes", "no", "uncertain"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class LinearHead:
    """Standardisation + logistic regression + a sensitivity-first operating point with an abstention band."""
    name: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    band: float
    metrics: dict = field(default_factory=dict)

    def probability(self, x: np.ndarray) -> np.ndarray:
        """x: (n, d) features -> (n,) probability of the positive class."""
        z = ((np.asarray(x, np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        return _sigmoid(z)

    def decide(self, p: float, familiar: bool = True) -> str:
        """Abstain on unfamiliar photos and inside the band around the threshold, instead of guessing."""
        if not familiar or abs(p - self.threshold) < self.band:
            return UNCERTAIN
        return YES if p >= self.threshold else NO


@dataclass(frozen=True)
class OODGate:
    """Unfamiliar-photo gate: mean cosine distance to the k nearest training photos, above a percentile threshold."""
    reference: np.ndarray        # (n_train, d) L2-normalised training features
    k: int
    threshold: float
    info: dict = field(default_factory=dict)

    def distance(self, x: np.ndarray) -> float:
        u = np.asarray(x, np.float32).ravel()
        u = u / np.linalg.norm(u)
        sims = self.reference.astype(np.float32) @ u
        return float(1 - np.sort(sims)[-self.k:].mean())

    def familiar(self, x: np.ndarray) -> tuple[bool, float]:
        d = self.distance(x)
        return d <= self.threshold, d


@dataclass(frozen=True)
class ModelBundle:
    version: str
    encoder: str
    dataset: str
    heads: dict[str, LinearHead]
    ood: OODGate | None = None
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- portable format
    def save(self, manifest_path: str | Path) -> Path:
        """Writes <name>.json (manifest) + <name>.npz (arrays). Returns the manifest path."""
        manifest_path = Path(manifest_path).with_suffix(".json")
        arrays = {}
        heads_meta = {}
        for name, h in self.heads.items():
            arrays.update({f"{name}.mean": h.mean, f"{name}.scale": h.scale, f"{name}.coef": h.coef})
            heads_meta[name] = {"intercept": h.intercept, "threshold": h.threshold, "band": h.band, "metrics": h.metrics}
        if self.ood is not None:
            arrays["ood.reference"] = self.ood.reference
        np.savez_compressed(manifest_path.with_suffix(".npz"), **arrays)
        manifest = {"format_version": FORMAT_VERSION, "version": self.version, "encoder": self.encoder,
                    "dataset": self.dataset, "arrays": manifest_path.with_suffix(".npz").name, "heads": heads_meta,
                    "ood": None if self.ood is None else {"k": self.ood.k, "threshold": self.ood.threshold, **self.ood.info},
                    "meta": self.meta}
        manifest_path.write_text(json.dumps(manifest, indent=1, default=_json_default), encoding="utf-8")
        return manifest_path

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "ModelBundle":
        manifest_path = Path(manifest_path)
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{manifest_path.name}: format_version {m.get('format_version')} not supported "
                             f"(this code reads version {FORMAT_VERSION}); re-export the model")
        with np.load(manifest_path.with_name(m["arrays"]), allow_pickle=False) as z:   # no pickle: safe to load
            heads = {name: LinearHead(name, z[f"{name}.mean"], z[f"{name}.scale"], z[f"{name}.coef"],
                                      float(h["intercept"]), float(h["threshold"]), float(h["band"]), h.get("metrics", {}))
                     for name, h in m["heads"].items()}
            ood = None
            if m.get("ood"):
                info = {k: v for k, v in m["ood"].items() if k not in ("k", "threshold")}
                ood = OODGate(z["ood.reference"], int(m["ood"]["k"]), float(m["ood"]["threshold"]), info)
        return cls(m["version"], m["encoder"], m["dataset"], heads, ood, m.get("meta", {}))

    # ---------------------------------------------------------------- legacy adapter
    @classmethod
    def from_sklearn_dict(cls, bundle: dict) -> "ModelBundle":
        """Adapter: the v1 joblib dict {heads: {name: {model, scaler, threshold, band, metrics}}, ood, ...}."""
        heads = {}
        for name, h in bundle["heads"].items():
            model, scaler = h["model"], h["scaler"]
            heads[name] = LinearHead(name, np.asarray(scaler.mean_, np.float64), np.asarray(scaler.scale_, np.float64),
                                     np.asarray(model.coef_[0], np.float64), float(model.intercept_[0]),
                                     float(h["threshold"]), float(h["band"]), h.get("metrics", {}))
        ood = None
        if bundle.get("ood") is not None:
            o = bundle["ood"]
            ood = OODGate(np.asarray(o["reference"]), int(o["k"]), float(o["threshold"]),
                          {k: v for k, v in o.items() if k not in ("reference", "k", "threshold")})
        meta = {k: bundle[k] for k in ("trained_at", "encoder_cv_auroc") if k in bundle}
        return cls(bundle["version"], bundle.get("encoder", ""), bundle["dataset"], heads, ood, meta)


def load(path: str | Path) -> ModelBundle:
    """Factory Method: the portable manifest (.json) by default; the legacy pickle (.joblib) through the adapter."""
    path = Path(path)
    if path.suffix == ".json":
        return ModelBundle.from_manifest(path)
    if path.suffix == ".joblib":
        import joblib  # legacy only: needs the SAME scikit-learn version that wrote the file
        return ModelBundle.from_sklearn_dict(joblib.load(path))
    raise ValueError(f"unknown model file type: {path.name}")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
