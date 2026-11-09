"""Image loading and colour spaces: the ONE implementation used by every wound module (and the backend).

Why CIELAB: a* is a direct green(-)/red(+) axis and Euclidean distances in Lab approximate perceived colour
differences, so "redder than the surrounding skin" becomes a simple, explainable subtraction.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import DEFAULT_KERNELS, MAX_SIDE, Kernels


def load(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """Bytes -> BGR uint8, EXIF-rotated (phones store portrait photos rotated), longest side <= max_side."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((max_side, max_side))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit Lab -> real CIELAB units (L 0-100, a/b about -128..127), float32."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def skin_mask(bgr: np.ndarray, kernels: Kernels = DEFAULT_KERNELS) -> np.ndarray:
    """Classic YCrCb skin box (Cr 133-180, Cb 77-135), cleaned by a small opening. Boolean mask."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (30, 133, 77), (255, 180, 135))
    k = np.ones((kernels.skin_open, kernels.skin_open), np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k) > 0


def ellipse(size: int) -> np.ndarray:
    """Elliptical structuring element (isotropic, unlike a square kernel)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Boolean dilation helper (OpenCV works on uint8)."""
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0
