"""Every tunable number of the wound pipeline, in one place.

Keeping thresholds and kernel sizes here (not scattered as literals in the code) means one file to review,
one file to calibrate, and the analysis functions stay free of magic numbers.
"""

from dataclasses import dataclass

VERSION = "wound-cv-0.2"
MAX_SIDE = 640  # every photo is analysed at this resolution (longest side, px)

# What a single 2-D photo cannot show: always reported as "not measured", never as "absent".
NOT_MEASURED = ("swelling", "drainage", "depth", "warmth", "pain")


@dataclass(frozen=True)
class Limits:
    """Decision thresholds. Engineering defaults, not clinically validated."""
    min_brightness: float = 22.0       # mean L*
    max_brightness: float = 92.0
    min_sharpness: float = 25.0        # variance of the Laplacian on the centre, 640 px image
    max_glare_fraction: float = 0.08   # share of near-white pixels in the centre
    min_skin_fraction: float = 0.20    # share of the frame that looks like skin
    redness_delta_a: float = 8.0       # a* above the patient's own skin that counts as red
    min_wound_fraction: float = 0.002  # smaller than this of the frame: nothing reliable found
    max_wound_fraction: float = 0.60
    min_registration_inliers: int = 12
    min_registration_ratio: float = 0.25
    min_registration_scale: float = 0.4
    max_registration_scale: float = 2.5
    trend_delta_a: float = 4.0         # change in redness (a* units) that counts as a colour trend
    area_change_pct: float = 30.0      # change in area that is worth mentioning


@dataclass(frozen=True)
class Kernels:
    """Structuring-element and band sizes in pixels, for a MAX_SIDE image.

    Known limitation (v1 audit A5): these are fixed pixel sizes, so their physical size depends on the camera
    distance. The calibration card (v2 step 2) replaces them with sizes in millimetres.
    """
    skin_open: int = 3
    near_skin_dilate: int = 15
    staple_tophat: int = 11
    staple_along: int = 15
    staple_close: int = 9
    candidate_close: int = 13
    mask_dilate: int = 7
    reference_band_inner: int = 51     # peri-wound skin reference = band between these two dilations
    reference_band_outer: int = 121
    spread_ring: int = 25              # ring just outside the wound where spreading redness is measured
    closure_search: int = 31


DEFAULT_LIMITS = Limits()
DEFAULT_KERNELS = Kernels()
