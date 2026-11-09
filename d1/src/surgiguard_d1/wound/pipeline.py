"""D1-F wound-photo analysis: the Facade over the pipeline stages.

    load -> quality gate -> segmentation (Strategy) -> measurements -> [registration + comparison] -> overlay

Each stage lives in its own module with one responsibility (quality.py, segmentation.py, measurement.py,
registration.py, render.py; colours in colour.py; every number in config.py). `WoundAnalyzer` only orchestrates
them and receives its segmenter by injection, so a learned segmenter can replace the classical one without any
change here. `analyze()` keeps the original function API for the backend and the tests.

It reports what is visible. It never says infected / healed / safe. Swelling, discharge, depth and warmth are
NOT measurable from one 2-D photo and are listed as not measured.
"""

from __future__ import annotations

import cv2

from . import registration
from .colour import load
from .config import DEFAULT_KERNELS, DEFAULT_LIMITS, NOT_MEASURED, VERSION, Kernels, Limits
from .measurement import measure
from .quality import UNUSABLE, assess_quality
from .registration import compare, register
from .render import overlay
from .segmentation import ClassicalSegmenter, Segmentation, Segmenter

__all__ = ["VERSION", "Limits", "WoundAnalyzer", "analyze", "assess_quality", "compare", "load", "measure",
           "overlay", "register", "strip_private"]

METHOD = "OpenCV: local colour contrast in CIELAB + staple detection + ORB registration (classical vision, no learning)"


class WoundAnalyzer:
    """Orchestrates the stages. Stateless after construction, so one instance can serve concurrent requests."""

    def __init__(self, segmenter: Segmenter | None = None, limits: Limits = DEFAULT_LIMITS,
                 kernels: Kernels = DEFAULT_KERNELS):
        self.limits, self.kernels = limits, kernels
        self.segmenter = segmenter or ClassicalSegmenter(limits, kernels)

    def analyze(self, data: bytes, reference: bytes | None = None) -> dict:
        """Full analysis of one photo, optionally compared with the patient's reference photo."""
        try:
            bgr = load(data)
        except Exception as e:  # not an image / corrupt file
            return {"version": VERSION, "quality": {"status": UNUSABLE, "problems": [f"not a readable image ({type(e).__name__})"]},
                    "measurements": None, "segmentation": {"found": False, "reason": "unreadable image"},
                    "not_measured": list(NOT_MEASURED)}
        quality = assess_quality(bgr, self.limits)
        out = {"version": VERSION, "method": METHOD, "quality": quality, "not_measured": list(NOT_MEASURED),
               "measurements": None, "comparison": None, "segmentation": {"found": False, "reason": "photo quality too low"},
               "_mask": None, "_bgr": bgr}
        if quality["status"] == UNUSABLE:
            return out
        seg = self.segmenter.segment(bgr)
        if not seg.found:
            out["segmentation"] = {"found": False, "reason": seg.reason}
            return out
        m = measure(seg, self.limits, self.kernels)
        out.update(_mask=seg.mask, measurements=m, segmentation=self._public_segmentation(seg, m))
        if reference is not None:
            out["comparison"] = self._compare_with(reference, out)
        return out

    def _compare_with(self, reference: bytes, current: dict) -> dict:
        ref = self.analyze(reference)
        if ref.get("_bgr") is None:
            reg = {"comparable": False, "reason": "no reference"}
        else:
            try:
                reg = register(ref["_bgr"], current["_bgr"], self.limits)
            except cv2.error:
                reg = {"comparable": False, "reason": "registration failed"}
        return compare(ref, current, reg, self.limits)

    def _public_segmentation(self, seg: Segmentation, m: dict) -> dict:
        return {"found": True, "method": seg.method, "segmenter": self.segmenter.name,
                "area_pct_of_frame": m["wound_area_pct_of_frame"],
                "caution": "region may include normal tissue around the wound" if seg.too_large else None}


_DEFAULT = WoundAnalyzer()


def analyze(data: bytes, reference: bytes | None = None, limits: Limits = DEFAULT_LIMITS) -> dict:
    """Function API kept for callers of v1 (backend, tests)."""
    analyzer = _DEFAULT if limits == DEFAULT_LIMITS else WoundAnalyzer(limits=limits)
    return analyzer.analyze(data, reference)


def strip_private(result: dict) -> dict:
    """JSON-safe copy (drops the numpy arrays used for rendering)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


# v1 names, still importable from here
public_registration = registration.public_registration
