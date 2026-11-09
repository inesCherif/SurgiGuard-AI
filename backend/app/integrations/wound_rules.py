"""When should a clinician look at a wound photo? Each reason is a small rule object (Strategy pattern).

`decide()` in d1_wound.py asks every rule in REVIEW_RULES and raises one clinician-only "attention" alert with
all the reasons found. Adding or removing a reason means adding or removing a class here (Open/Closed); the alert
code never changes. Every rule only reads the stored observation, so each one is unit-testable on a plain dict.

Which signals may raise an alert (v2):
- the trained model saying "yes" on discharge / infection-risk rating / professional review: validated on the
  untouched SurgWound test split (AUROC 0.82-0.84, sensitivity-first thresholds);
- a large area growth between two photos PROVEN to show the same site (registration + distance correction);
- a LIVE vision-LLM reading that both suggests review and reports a medium/high-confidence concerning finding.

Removed in v2: "redder than the reference photo" (mean / ring Delta a*). Evaluated against the surgeons' erythema
labels, this OpenCV redness measure scored AUROC 0.503, i.e. chance level; alerting on it only added random alarms.
It is still measured and shown to the clinician as information, never used as an alert reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ReviewRule(Protocol):
    name: str

    def reason(self, value: dict) -> str | None:
        """value = the photo observation's stored value. Returns a short reason, or None."""


@dataclass(frozen=True)
class ModelSignRule:
    """The model can only ADD a reason (a "no" never removes another rule's reason)."""
    heads: tuple[str, ...] = ("exudate", "elevated_risk", "needs_review")
    name: str = "model"

    def reason(self, value: dict) -> str | None:
        md = value.get("model") or {}
        if not md.get("available"):
            return None
        hits = [f'{a["text"]} (p={a["probability"]})' for k, a in md.get("attributes", {}).items()
                if k in self.heads and a.get("decision") == "yes"]
        return "model: " + "; ".join(hits) if hits else None


@dataclass(frozen=True)
class AreaGrowthRule:
    """Registered, distance-corrected growth of the measured area vs the reference photo."""
    min_growth_pct: float = 30.0
    name: str = "area_growth"

    def reason(self, value: dict) -> str | None:
        comp = (value.get("d1") or {}).get("comparison") or {}
        if comp.get("available") and comp.get("area_change_pct", 0) >= self.min_growth_pct:
            return f"code: marked area {comp['area_change_pct']}% larger than the reference photo (same site, distance-corrected)"
        return None


@dataclass(frozen=True)
class LiveVisionRule:
    """A vision-LLM reading counts only when live (never the offline stub) and backed by a confident finding."""
    concerning: frozenset = frozenset({"discharge_appearance", "wound_edges", "redness"})
    confident: frozenset = frozenset({"medium", "high"})
    name: str = "vision"

    def reason(self, value: dict) -> str | None:
        vr = value.get("vision_review") or {}
        if not (vr.get("available") and vr.get("mode") == "live" and vr.get("review_suggested")):
            return None
        strong = [f["finding"] for f in vr.get("findings", [])
                  if f.get("aspect") in self.concerning and f.get("confidence") in self.confident]
        return ("vision reading: " + "; ".join(strong))[:300] if strong else None


REVIEW_RULES: tuple[ReviewRule, ...] = (ModelSignRule(), AreaGrowthRule(), LiveVisionRule())


def review_reasons(value: dict, rules: tuple[ReviewRule, ...] = REVIEW_RULES) -> list[str]:
    return [r for r in (rule.reason(value) for rule in rules) if r]
