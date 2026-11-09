"""Acquisition quality gate: a poor photo gets "retake" advice, never a measurement.

Garbage in, garbage out is the first rule of applied computer vision: every later stage trusts this gate.
"""

import cv2
import numpy as np

from .colour import skin_mask, to_lab
from .config import DEFAULT_LIMITS, Limits

GOOD, QUESTIONABLE, UNUSABLE = "good", "questionable", "unusable"


def _centre(h: int, w: int) -> tuple[slice, slice]:
    """Central 60 % of the frame, where the patient aims the wound."""
    return slice(int(h * .2), int(h * .8)), slice(int(w * .2), int(w * .8))


def assess_quality(bgr: np.ndarray, limits: Limits = DEFAULT_LIMITS) -> dict:
    h, w = bgr.shape[:2]
    rows, cols = _centre(h, w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    centre = gray[rows, cols]
    brightness = float(to_lab(bgr)[..., 0].mean())
    sharpness = float(cv2.Laplacian(centre, cv2.CV_64F).var())   # blur removes edges -> low Laplacian variance
    glare = float((centre > 245).mean())
    skin_fraction = float(skin_mask(bgr)[rows, cols].mean())

    problems = []
    if brightness < limits.min_brightness:
        problems.append("too dark: add light")
    elif brightness > limits.max_brightness:
        problems.append("overexposed: move away from direct light")
    if sharpness < limits.min_sharpness:
        problems.append("blurry: hold the phone still and let it focus")
    if glare > limits.max_glare_fraction:
        problems.append("glare or reflection on the skin: tilt the phone")
    if skin_fraction < limits.min_skin_fraction:
        problems.append("not enough skin in the frame: centre the wound")

    hard = (brightness < limits.min_brightness or skin_fraction < limits.min_skin_fraction
            or sharpness < limits.min_sharpness / 3)
    status = UNUSABLE if hard else QUESTIONABLE if problems else GOOD
    return {"status": status, "problems": problems, "brightness": round(brightness, 1), "sharpness": round(sharpness, 1),
            "glare_pct": round(glare * 100, 1), "skin_fraction": round(skin_fraction, 2)}
