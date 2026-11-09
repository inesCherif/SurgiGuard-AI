"""Wound review rules (Strategy objects): which signals may ask a clinician to look at a photo."""

from app.integrations.wound_rules import AreaGrowthRule, LiveVisionRule, ModelSignRule, review_reasons

YES = {"probability": 0.9, "decision": "yes", "text": "visible fluid or discharge (exudate)"}
NO = {**YES, "decision": "no", "probability": 0.05}


def registered(**kw):
    return {"d1": {"comparison": {"available": True, "trend": "stable_colour", "area_change_pct": 0,
                                  "redness_change_delta_a": 0, "redness_spread_change": 0, **kw}}}


def test_chance_level_redness_trend_no_longer_raises_an_alert():
    """Mean / ring Delta a* scored AUROC 0.503 against surgeons' erythema labels: shown, never an alert reason."""
    assert review_reasons(registered(trend="more_red", redness_change_delta_a=9, redness_spread_change=7)) == []


def test_registered_area_growth_is_a_reason():
    assert AreaGrowthRule().reason(registered(area_change_pct=45)).startswith("code: marked area 45%")
    assert AreaGrowthRule().reason(registered(area_change_pct=10)) is None
    refused = {"d1": {"comparison": {"available": False, "reason": "not the same site", "area_change_pct": 99}}}
    assert AreaGrowthRule().reason(refused) is None          # an unproven comparison is never a reason


def test_model_yes_adds_a_reason_and_a_no_removes_nothing():
    value = {"model": {"available": True, "attributes": {"exudate": YES, "needs_review": NO}}, **registered(area_change_pct=40)}
    reasons = review_reasons(value)
    assert any(r.startswith("model:") for r in reasons) and any(r.startswith("code:") for r in reasons)
    assert ModelSignRule().reason({"model": {"available": True, "attributes": {"exudate": NO}}}) is None
    assert ModelSignRule().reason({"model": {"available": True, "attributes": {"erythema": YES}}}) is None  # sign only, not a review head


def test_vision_counts_only_when_live_and_confident():
    finding = {"aspect": "discharge_appearance", "finding": "Yellowish fluid appears", "confidence": "medium"}
    live = {"vision_review": {"available": True, "mode": "live", "review_suggested": True, "findings": [finding]}}
    assert LiveVisionRule().reason(live).startswith("vision reading:")
    stub = {"vision_review": {**live["vision_review"], "mode": "stub"}}
    low = {"vision_review": {**live["vision_review"], "findings": [{**finding, "confidence": "low"}]}}
    assert LiveVisionRule().reason(stub) is None and LiveVisionRule().reason(low) is None
