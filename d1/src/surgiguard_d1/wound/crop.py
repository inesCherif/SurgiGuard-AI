"""Crop a photo around the wound before classifying it (v2 step 4).

Why: the attribute classifier embeds the WHOLE photo, so sheets, clothes, tattoos and background take most of the
encoder's attention while redness or discharge occupy a few percent of the pixels. Cropping around the wound found
by a `Segmenter` puts the evidence in the centre and at a usable resolution.

`WoundCropper` depends only on the `Segmenter` protocol (Dependency Inversion): classical or learned, injected.
It keeps context around the wound (the peri-wound skin is where redness and swelling are judged), makes the box
square (the encoder resizes to a square, so a long thin box would be distorted), and falls back to the whole photo
when no wound is found - the fallback is reported, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import Segmenter


@dataclass(frozen=True)
class Crop:
    """Pixel box (x0, y0, x1, y1) in the original photo and whether a wound was actually found."""
    box: tuple[int, int, int, int]
    found: bool
    reason: str | None = None

    def apply(self, rgb_or_bgr: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return rgb_or_bgr[y0:y1, x0:x1]


def square_box_around(mask: np.ndarray, margin: float, min_side_frac: float) -> tuple[int, int, int, int]:
    """Square box centred on the mask's bounding box, enlarged by `margin` x its long side on every side, at least
    `min_side_frac` of the photo's short side, shifted (not shrunk) to stay inside the photo when possible."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    long_side = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    side = min(max(long_side * (1 + 2 * margin), min_side_frac * min(h, w)), min(h, w))
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    s = int(round(side))
    return x0, y0, min(x0 + s, w), min(y0 + s, h)


class WoundCropper:
    def __init__(self, segmenter: Segmenter, margin: float = 0.35, min_side_frac: float = 0.25):
        self.segmenter, self.margin, self.min_side_frac = segmenter, margin, min_side_frac

    def crop(self, bgr: np.ndarray) -> Crop:
        h, w = bgr.shape[:2]
        seg = self.segmenter.segment(bgr)
        if not seg.found:
            return Crop((0, 0, w, h), False, seg.reason)
        return Crop(square_box_around(seg.mask, self.margin, self.min_side_frac), True)
