"""Learned wound segmentation (v2 step 3, part 3): a small U-Net exported to ONNX, run with onnxruntime.

Why ONNX: training needs GPU PyTorch (~2.5 GB), but inference must stay light and portable like the numpy model
bundle of step 1. onnxruntime is a ~15 MB CPU dependency already in the backend, its sessions are thread-safe,
and the model file is self-contained. The training script (scripts/train_segmenter.py) imports the SAME
`letterbox` and `normalise` functions below, so preprocessing can never differ between training and inference
(a classic silent bug in deployed CV models).

`LearnedSegmenter` implements the `Segmenter` Strategy, so `WoundAnalyzer(segmenter=LearnedSegmenter(...))`
swaps it in without touching quality, measurement, registration or the backend (Open/Closed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .colour import skin_mask, to_lab
from .config import DEFAULT_KERNELS, DEFAULT_LIMITS, Kernels, Limits
from .segmentation import Segmentation, finish_segmentation

MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet statistics (the encoder is ImageNet-pretrained)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class Letterbox:
    """How a photo was fitted into the square network input: scale, then padding (top, left)."""
    scale: float
    top: int
    left: int
    height: int          # original size
    width: int


def letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Fit the longest side to `size` and pad the rest with zeros. Keeps the aspect ratio, so a thin incision
    is not squeezed thinner (a plain resize of a 4:3 photo to a square changes widths by 33 %)."""
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    small = cv2.resize(bgr, (nw, nh), interpolation=interp).reshape((nh, nw) + bgr.shape[2:])  # keep 1-ch axis
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size) + bgr.shape[2:], bgr.dtype)
    canvas[top:top + nh, left:left + nw] = small
    return canvas, Letterbox(s, top, left, h, w)


def unletterbox(prob: np.ndarray, lb: Letterbox) -> np.ndarray:
    """Inverse of `letterbox` for a single-channel map: crop the padding, resize back to the photo size."""
    nh, nw = max(1, round(lb.height * lb.scale)), max(1, round(lb.width * lb.scale))
    crop = prob[lb.top:lb.top + nh, lb.left:lb.left + nw]
    return cv2.resize(crop.astype(np.float32), (lb.width, lb.height), interpolation=cv2.INTER_LINEAR)


def normalise(bgr_square: np.ndarray) -> np.ndarray:
    """uint8 BGR HxWx3 -> float32 1x3xHxW, RGB, ImageNet-normalised."""
    rgb = cv2.cvtColor(bgr_square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


@dataclass(frozen=True)
class ModelCard:
    """Metadata stored next to the .onnx file (<model>.json): what the model expects and how it scored."""
    input_size: int = 320
    threshold: float = 0.5
    version: str = "unknown"
    trained_on: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    @classmethod
    def load(cls, onnx_path: Path) -> "ModelCard":
        p = Path(onnx_path).with_suffix(".json")
        if not p.is_file():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# The shared 0.2 %-of-the-frame minimum exists because tiny CLASSICAL detections are mostly noise. A trained model's
# small detections are usually real (journal V19: it rejected 24 correctly found small ulcers, 17 % of FUSeg test
# wounds are below 0.2 %). So the learned segmenter gets its own floor: 0.02 % (~50 px on a 512 px photo).
LEARNED_LIMITS = replace(DEFAULT_LIMITS, min_wound_fraction=0.0002)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class LearnedSegmenter:
    """Segmenter Strategy backed by an ONNX U-Net. The onnxruntime session is created once (lazily)."""

    def __init__(self, model_path: str | Path, limits: Limits = LEARNED_LIMITS, kernels: Kernels = DEFAULT_KERNELS,
                 threshold: float | None = None, session=None):
        self.model_path = Path(model_path)
        self.card = ModelCard.load(self.model_path)
        self.threshold = self.card.threshold if threshold is None else threshold
        self.limits, self.k = limits, kernels
        self._session = session            # injectable for tests (any object with .run and .get_inputs)
        self.name = f"learned-unet:{self.card.version}"

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort      # lazy: importing the wound package must not require onnxruntime
            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        return self._session

    def probability(self, bgr: np.ndarray) -> np.ndarray:
        """Per-pixel wound probability at the photo's own resolution."""
        square, lb = letterbox(bgr, self.card.input_size)
        inp = self.session.get_inputs()[0].name
        logits = self.session.run(None, {inp: normalise(square)})[0]
        return unletterbox(_sigmoid(np.asarray(logits, np.float32)[0, 0]), lb)

    def segment(self, bgr: np.ndarray, ignore: np.ndarray | None = None,
                px_per_mm: float | None = None) -> Segmentation:
        k = self.k.scaled(px_per_mm) if px_per_mm else self.k
        lab = to_lab(bgr)
        mask = self.probability(bgr) > self.threshold
        if ignore is not None:
            mask &= ~ignore
        mask = _drop_specks(mask, min_fraction=self.limits.min_wound_fraction * 0.5)
        if not mask.any():
            return Segmentation(None, lab, bgr, reason="the model found no wound")
        skin = skin_mask(bgr, k)
        if ignore is not None:
            skin &= ~ignore
        return finish_segmentation(mask, lab, skin, bgr, "learned U-Net probability > %.2f" % self.threshold,
                                   self.limits, k)


def _drop_specks(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    """Remove isolated blobs smaller than `min_fraction` of the image (false positives on pores, hairs)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = max(1, int(min_fraction * mask.size))
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]
