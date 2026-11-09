"""Longitudinal comparison: do two photos show the same site, and what changed?

Rule: a comparison is only reported after GEOMETRIC proof that both photos show the same skin
(ORB features + RANSAC homography). Different patients, sites or extreme zoom are refused, never shown as change.
"""

import cv2
import numpy as np

from .config import DEFAULT_LIMITS, Limits
from .quality import QUESTIONABLE, UNUSABLE


def register(ref_bgr: np.ndarray, cur_bgr: np.ndarray, limits: Limits = DEFAULT_LIMITS) -> dict:
    """Map the current photo onto the reference. Returns comparable=False with a reason when it cannot be proven."""
    orb = cv2.ORB_create(1500)
    g1, g2 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (ref_bgr, cur_bgr))
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return _refused("too little texture to match the two photos")
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d2, d1), key=lambda m: m.distance)[:300]
    if len(matches) < limits.min_registration_inliers:
        return _refused("the two photos do not seem to show the same area")
    src = np.float32([k2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n_in = int(inl.sum()) if inl is not None else 0
    ratio = n_in / max(len(matches), 1)
    if H is None or n_in < limits.min_registration_inliers or ratio < limits.min_registration_ratio:
        return _refused(f"the two photos do not seem to show the same area ({n_in} consistent points)", inliers=n_in)
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    if not limits.min_registration_scale < scale < limits.max_registration_scale:   # wrong site matched on texture
        return _refused("the photos were taken from very different distances", inliers=n_in, scale=round(scale, 2))
    return {"comparable": True, "inliers": n_in, "inlier_ratio": round(ratio, 2), "scale": round(scale, 2), "homography": H}


def compare(ref: dict, cur: dict, reg: dict, limits: Limits = DEFAULT_LIMITS) -> dict:
    """ref/cur: results of analyze(). Unknown stays unknown; a different site is never reported as a change."""
    if not ref.get("measurements") or not cur.get("measurements"):
        return {"available": False, "reason": "one of the photos could not be measured: unknown"}
    statuses = (ref["quality"]["status"], cur["quality"]["status"])
    if UNUSABLE in statuses:
        return {"available": False, "reason": "photo quality too low to compare: unknown"}
    if not reg.get("comparable"):
        return {"available": False, "reason": reg.get("reason", "not the same site"), "registration": public_registration(reg)}
    r, c = ref["measurements"], cur["measurements"]
    if r.get("area_mm2") and c.get("area_mm2"):      # both calibrated: real areas, no camera-distance assumption
        area_ratio, area_basis = c["area_mm2"] / r["area_mm2"], "mm2 (calibration card)"
    else:                                            # frame share corrected by the registration scale
        area_ratio = c["wound_area_pct_of_frame"] / max(r["wound_area_pct_of_frame"], 1e-6) / reg["scale"] ** 2
        area_basis = "frame share, distance-corrected"
    d_area = round((area_ratio - 1) * 100)
    d_red = round(c["redness_mean_delta_a"] - r["redness_mean_delta_a"], 1)
    d_spread = round(c["redness_spread_ring_delta_a"] - r["redness_spread_ring_delta_a"], 1)
    t = limits.trend_delta_a
    notes = []
    if d_red >= t or d_spread >= t:
        trend = "more_red"
        notes.append("redder than the reference photo" + (", and the redness extends beyond the wound edge" if d_spread >= t else ""))
    elif d_red <= -t:
        trend = "less_red"
        notes.append("less red than the reference photo")
    else:
        trend = "stable_colour"
        notes.append("no clear change in redness")
    if d_area >= limits.area_change_pct:
        notes.append(f"the marked area is about {d_area}% larger (after correcting for distance)")
    elif d_area <= -limits.area_change_pct:
        notes.append(f"the marked area is about {-d_area}% smaller (after correcting for distance)")
    return {"available": True, "trend": trend, "summary": "; ".join(notes), "redness_change_delta_a": d_red,
            "redness_spread_change": d_spread, "area_change_pct": d_area, "area_basis": area_basis,
            "redness_area_ref_pct": r["redness_area_pct"], "redness_area_now_pct": c["redness_area_pct"],
            "registration": public_registration(reg),
            "quality": QUESTIONABLE if QUESTIONABLE in statuses else "good",
            "caution": "Computed from two photos: light, angle and distance change the numbers. Not a diagnosis."}


def public_registration(reg: dict) -> dict:
    """JSON-safe registration info (the homography matrix stays internal)."""
    return {k: v for k, v in reg.items() if k != "homography"}


def _refused(reason: str, **info) -> dict:
    return {"comparable": False, "reason": reason, "homography": None, **info}
