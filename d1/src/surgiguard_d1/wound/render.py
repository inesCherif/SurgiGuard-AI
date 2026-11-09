"""Visual explanation: draw exactly what the code measured, so a human can check it."""

import cv2
import numpy as np


def overlay(bgr: np.ndarray, mask: np.ndarray | None, caption: str = "") -> bytes:
    """JPEG of the photo with the analysed region tinted and outlined."""
    out = bgr.copy()
    if mask is not None:
        out[mask] = (0.55 * out[mask] + 0.45 * np.array([60, 200, 90])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (40, 220, 255), 2)
    if caption:
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (30, 30, 30), -1)
        cv2.putText(out, caption[:70], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return encode_jpeg(out)


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
