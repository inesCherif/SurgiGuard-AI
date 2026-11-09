"""Draw the calibration card from its CardSpec: one source of truth for the printed card and the detector."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import ARUCO_DICT, DEFAULT_CARD, CardSpec


def render_card(spec: CardSpec = DEFAULT_CARD, px_per_mm: float = 12.0, label: bool = True) -> np.ndarray:
    """BGR image of the card at the given resolution (12 px/mm ~ 300 dpi)."""
    s = px_per_mm
    W, H = int(round(spec.width_mm * s)), int(round(spec.height_mm * s))
    img = np.full((H, W, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for mid, corners in spec.marker_corners_mm().items():
        x0, y0 = (corners[0] * s).round().astype(int)
        size = int(round(spec.marker_mm * s))
        img[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, mid, size), cv2.COLOR_GRAY2BGR)
    for p in spec.patches:
        x0, y0, x1, y1 = (int(round(v * s)) for v in (p.x, p.y, p.x + p.w, p.y + p.h))
        img[y0:y1, x0:x1] = p.rgb[::-1]
    # 50 mm ruler with 1 mm ticks (longer every 5 and 10 mm), centred horizontally
    x_start = (spec.width_mm - spec.ruler_mm) / 2
    y = spec.ruler_y_mm
    cv2.line(img, (int(x_start * s), int(y * s)), (int((x_start + spec.ruler_mm) * s), int(y * s)), (0, 0, 0), max(1, int(s * 0.15)))
    for mm in range(int(spec.ruler_mm) + 1):
        tick = 3.0 if mm % 10 == 0 else 2.0 if mm % 5 == 0 else 1.0
        x = int((x_start + mm) * s)
        cv2.line(img, (x, int(y * s)), (x, int((y + tick) * s)), (0, 0, 0), max(1, int(s * 0.1)))
    if label:
        _label(img, f"SurgiGuard calibration card - ruler {int(spec.ruler_mm)} mm", spec, s)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (200, 200, 200), max(1, int(s * 0.1)))   # cutting guide
    return img


def _label(img: np.ndarray, text: str, spec: CardSpec, s: float) -> None:
    """Text in the free band between the two top markers, shrunk until it fits (markers need a white quiet zone)."""
    left = spec.margin_mm + spec.marker_mm + 2.0
    width_mm = spec.width_mm - 2 * left
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(s * 0.08))
    scale = s * 0.08
    while cv2.getTextSize(text, font, scale, thick)[0][0] > width_mm * s and scale > 0.1:
        scale *= 0.92
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = int((spec.width_mm * s - tw) / 2)
    y = int((spec.margin_mm + spec.marker_mm / 2) * s + th / 2)
    cv2.putText(img, text, (x, y), font, scale, (60, 60, 60), thick, cv2.LINE_AA)
