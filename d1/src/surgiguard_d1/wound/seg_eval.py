"""How good is a segmenter? Metrics and a benchmark that work for ANY `Segmenter` (v2 step 3, part 1).

Before training a learned segmenter we need a fixed, honest yardstick, otherwise "the model is better" is an
opinion. This module gives:

* pixel metrics - Dice, IoU, precision, recall - on boolean masks;
* `width_ratio` - predicted / true mean width of an elongated wound (journal V14: the classical segmenter
  roughly doubles the width of thin incisions, which Dice alone hides when the incision is long);
* `SegmenterBenchmark` - runs any `Segmenter` (Strategy, depends only on the protocol: Dependency Inversion)
  over a list of `LabelledImage`s and returns a `BenchmarkReport` with per-image and aggregate scores.
  Classical and learned segmenters are therefore compared by the same code on the same data.

Conventions (standard in medical segmentation): two empty masks score Dice = IoU = 1 (correctly found nothing);
a segmenter that finds nothing on an image that has a wound scores 0 and counts as a miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Iterable

import cv2
import numpy as np

from .segmentation import Segmenter


@dataclass(frozen=True)
class MaskScores:
    """Overlap scores of one predicted mask against its ground truth."""
    dice: float
    iou: float
    precision: float
    recall: float
    width_ratio: float | None = None     # predicted / true mean width; None if either mask is empty


def _check(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if pred.shape != truth.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {truth.shape}")
    return pred, truth


def mean_width(mask: np.ndarray) -> float | None:
    """Mean width (px) of a blob = area / length.

    The length is the long side of the minimum-area rotated rectangle, which is exact for straight incisions
    and a slight under-estimate for curved ones (good enough to expose a 2x width error)."""
    mask = np.asarray(mask, bool)
    area = int(mask.sum())
    if area == 0:
        return None
    pts = cv2.findNonZero(mask.astype(np.uint8))
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    length = max(w, h, 1.0)
    return area / length


def score_masks(pred: np.ndarray | None, truth: np.ndarray) -> MaskScores:
    """Dice, IoU, precision, recall and width ratio. `pred=None` (nothing found) counts as an empty mask."""
    truth = np.asarray(truth, bool)
    pred = np.zeros_like(truth) if pred is None else pred
    pred, truth = _check(pred, truth)
    tp = int(np.logical_and(pred, truth).sum())
    p, t = int(pred.sum()), int(truth.sum())
    union = p + t - tp
    if p == 0 and t == 0:
        return MaskScores(1.0, 1.0, 1.0, 1.0, None)
    wp, wt = mean_width(pred), mean_width(truth)
    return MaskScores(
        dice=2 * tp / (p + t),
        iou=tp / union,
        precision=tp / p if p else 0.0,
        recall=tp / t if t else 0.0,
        width_ratio=(wp / wt) if (wp and wt) else None,
    )


@dataclass(frozen=True)
class LabelledImage:
    """One benchmark sample: a BGR photo, its ground-truth wound mask and optional known scale."""
    name: str
    bgr: np.ndarray
    truth: np.ndarray
    px_per_mm: float | None = None
    ignore: np.ndarray | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    segmenter: str
    per_image: dict[str, MaskScores] = field(default_factory=dict)
    misses: tuple[str, ...] = ()         # images with a wound where the segmenter found nothing

    def _agg(self, attr: str, how=mean) -> float | None:
        vals = [getattr(s, attr) for s in self.per_image.values() if getattr(s, attr) is not None]
        return how(vals) if vals else None

    def summary(self) -> dict:
        return {
            "segmenter": self.segmenter,
            "n_images": len(self.per_image),
            "dice_mean": self._agg("dice"),
            "dice_median": self._agg("dice", median),
            "iou_mean": self._agg("iou"),
            "precision_mean": self._agg("precision"),
            "recall_mean": self._agg("recall"),
            "width_ratio_median": self._agg("width_ratio", median),
            "misses": len(self.misses),
        }


class SegmenterBenchmark:
    """Runs any `Segmenter` over the same labelled images (Template Method: fixed loop, pluggable strategy)."""

    def __init__(self, samples: Iterable[LabelledImage]):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("a benchmark needs at least one labelled image")

    def run(self, segmenter: Segmenter) -> BenchmarkReport:
        per_image, misses = {}, []
        for s in self.samples:
            seg = segmenter.segment(s.bgr, ignore=s.ignore, px_per_mm=s.px_per_mm)
            if not seg.found and np.asarray(s.truth, bool).any():
                misses.append(s.name)
            per_image[s.name] = score_masks(seg.mask, s.truth)
        return BenchmarkReport(segmenter.name, per_image, tuple(misses))

    def compare(self, *segmenters: Segmenter) -> list[dict]:
        """Side-by-side summaries, e.g. compare(ClassicalSegmenter(), LearnedSegmenter(...))."""
        return [self.run(s).summary() for s in segmenters]
