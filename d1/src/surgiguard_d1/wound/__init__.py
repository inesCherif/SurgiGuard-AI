"""D1-F wound-photo analysis.

Public API: `analyze`, `overlay`, `strip_private` (function style) or `WoundAnalyzer` (object style, with an
injectable `Segmenter` strategy). See docs/wound/V2_ARCHITECTURE.md.
"""

from .config import VERSION, Kernels, Limits
from .pipeline import WoundAnalyzer, analyze, overlay, strip_private
from .segmentation import ClassicalSegmenter, Segmentation, Segmenter

__all__ = ["VERSION", "Kernels", "Limits", "WoundAnalyzer", "ClassicalSegmenter", "Segmentation", "Segmenter",
           "analyze", "overlay", "strip_private"]
