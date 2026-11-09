"""Measurements of a segmented wound, always RELATIVE to the patient's own skin in the same photo."""

import cv2
import numpy as np

from .colour import dilate
from .config import DEFAULT_KERNELS, DEFAULT_LIMITS, Kernels, Limits
from .segmentation import Segmentation


def measure(seg: Segmentation, limits: Limits = DEFAULT_LIMITS, kernels: Kernels = DEFAULT_KERNELS) -> dict:
    mask, lab, ref = seg.mask, seg.lab, seg.reference_lab
    h, w = mask.shape
    pix = lab[mask]
    d_a = pix[:, 1] - ref[1]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    area, perim = float(cv2.contourArea(cnt)), float(cv2.arcLength(cnt, True))
    solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1.0)
    circularity = 4 * np.pi * area / max(perim * perim, 1.0)
    _, _, bw, bh = cv2.boundingRect(cnt)
    # Redness in a ring just outside the mask: does the redness spread beyond the visible wound?
    ring = dilate(mask, np.ones((kernels.spread_ring,) * 2, np.uint8)) & ~mask
    ring_a = float(np.median(lab[ring][:, 1] - ref[1])) if ring.any() else 0.0
    L, a, b = pix[:, 0], pix[:, 1], pix[:, 2]
    dark = float((L < ref[0] - 28).mean() * 100)                                           # dark / crust-coloured
    pale = float(((L > ref[0] + 8) & (b > ref[2] + 12) & (a < ref[1] + 6)).mean() * 100)   # pale-yellow regions
    return {
        "wound_area_pct_of_frame": round(area / (h * w) * 100, 2),
        "bbox_aspect": round(bw / max(bh, 1), 2),
        "redness_mean_delta_a": round(float(d_a.mean()), 1),
        "redness_area_pct": round(float((d_a > limits.redness_delta_a).mean() * 100), 1),
        "redness_spread_ring_delta_a": round(ring_a, 1),
        "colour_variability": round(float(pix.std(axis=0).mean()), 1),
        "boundary_solidity": round(float(solidity), 2),
        "boundary_circularity": round(float(circularity), 2),
        "dark_crust_coloured_pct": round(dark, 1),
        "pale_yellow_pct": round(pale, 1),
        "skin_reference_lab": [round(float(v), 1) for v in ref],
        "closure_material": closure_material(seg, kernels),
    }


def closure_material(seg: Segmentation, kernels: Kernels = DEFAULT_KERNELS) -> dict:
    """Staples counted by the segmenter when it used them; otherwise a rough count of bright, grey blobs nearby."""
    if seg.staple_count:
        return {"staple_like_blobs": seg.staple_count,
                "note": "bright metal-like marks along the incision; one staple can show as several marks"}
    hsv = cv2.cvtColor(seg.bgr, cv2.COLOR_BGR2HSV)
    near = dilate(seg.mask, np.ones((kernels.closure_search,) * 2, np.uint8))
    bright = ((hsv[..., 2] > 190) & (hsv[..., 1] < 60) & near).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), 8)
    staples = sum(1 for i in range(1, n) if 6 <= stats[i, cv2.CC_STAT_AREA] <= 400)
    return {"staple_like_blobs": int(staples), "note": "count of staple-like bright spots, indicative only"}
