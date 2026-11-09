"""Adapter: the ONE wound analysis (d1 pipeline) -> the field layout the patient app and agents already read.

Before v2 the backend ran a SECOND analyser (app/wound_image.py, numpy-only, 512 px, other thresholds) for the
patient view, and it compared photos WITHOUT checking they showed the same wound: two photos of two different
patients were reported as "+35 % redness" to the patient while the clinician view refused the comparison.

Now there is one analysis; these pure functions only re-shape it (Adapter pattern), so the React Native screens
(`analysis.problems`, `comparison.trend` up/down/stable, `redness_area_*_pct`) keep working unchanged.
"""

from __future__ import annotations

NOT_MEASURED = ["swelling", "drainage", "depth", "warmth"]
METHOD = "code: OpenCV analysis vs. the patient's own surrounding skin (not AI)"
TREND = {"more_red": "up", "less_red": "down", "stable_colour": "stable"}
LABEL = {"up": "more redness than the reference photo", "down": "less redness than the reference photo",
         "stable": "no clear change in redness"}
FIRST_PHOTO = {"available": False, "reason": "first photo: it becomes the reference"}


def patient_analysis(d1: dict | None) -> dict:
    """Quality + headline measurement, in the v1 `analysis` shape."""
    if not d1:
        return {"quality": "unusable", "problems": ["the photo could not be analysed"], "method": METHOD,
                "not_measured": NOT_MEASURED}
    q = d1.get("quality") or {}
    out = {"quality": q.get("status", "unusable"), "problems": list(q.get("problems", [])), "method": METHOD,
           "not_measured": NOT_MEASURED}
    out.update({k: q[k] for k in ("brightness", "sharpness", "skin_fraction") if k in q})
    m = d1.get("measurements")
    if m:
        out["redness_area_pct"] = m["redness_area_pct"]
        out["redness_intensity"] = m["redness_mean_delta_a"]
        out.update({k: m[k] for k in ("area_cm2", "length_mm", "width_mm") if k in m})   # calibration card present
    elif not d1.get("segmentation", {}).get("found") and out["quality"] != "unusable":
        out["problems"].append(d1.get("segmentation", {}).get("reason") or "no wound region found: centre the wound")
    return out


def patient_comparison(d1_comparison: dict | None) -> dict:
    """The registered comparison (or its refusal reason), in the v1 `comparison` shape."""
    if not d1_comparison:
        return dict(FIRST_PHOTO)
    if not d1_comparison.get("available"):
        return {"available": False, "reason": d1_comparison.get("reason", "photos could not be compared: unknown"),
                "registration": d1_comparison.get("registration")}
    trend = TREND.get(d1_comparison["trend"], "stable")
    ref, now = d1_comparison["redness_area_ref_pct"], d1_comparison["redness_area_now_pct"]
    return {"available": True, "trend": trend, "summary": LABEL[trend],
            "redness_area_ref_pct": ref, "redness_area_now_pct": now,
            "redness_area_change_points": round(now - ref, 1),
            "redness_area_change_pct": round((now - ref) / ref * 100) if ref >= 0.5 else None,
            "redness_intensity_change": d1_comparison["redness_change_delta_a"],
            "area_change_pct": d1_comparison["area_change_pct"],
            "registration": d1_comparison.get("registration"),
            "not_measured": NOT_MEASURED, "quality": d1_comparison.get("quality", "good"),
            "caution": "Measured by code from two photos of the same site (geometrically matched); light and angle "
                       "still change the numbers. Not a diagnosis."}
