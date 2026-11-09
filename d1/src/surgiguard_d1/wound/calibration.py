"""Calibration card: real size (mm, mm²) and corrected colour from a printed reference placed next to the wound.

Problem (v1 audit A4/A5): without a reference in the photo, the code cannot know
- the SIZE of the wound: v1 reported "% of the frame", which changes with every camera distance;
- the COLOUR of the light: a warm lamp shifts every pixel; v1 could only compare with the patient's own skin.

The card (printed from docs/wound/calibration_card.pdf) carries:
- 4 ArUco markers at known millimetre positions: their 16 corners give a homography image(px) -> card(mm), so any
  outline drawn on the same skin plane can be measured in mm and mm², perspective included;
- neutral grey patches: a per-channel gain/offset fitted in LINEAR RGB removes the colour cast of the light
  (white balance is a multiplication of light intensity, which is only valid before the sRGB gamma curve);
- a skin-tone and a red patch, used only to REPORT the remaining colour error (Delta E) as a quality check;
- a 50 mm ruler, to check the print was made at 100 % scale.

Assumption, stated in every result: the card lies flat on the skin, in the same plane as the wound.

Design: `CardSpec` (immutable value object: every size of the card), `CardCalibrator` (component injected into
`WoundAnalyzer`), `Calibration` (immutable result: homography, scale, colour model, card mask, residual error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass(frozen=True)
class Patch:
    name: str
    x: float          # top-left corner, mm from the card's top-left corner
    y: float
    w: float
    h: float
    rgb: tuple[int, int, int]  # printed sRGB value
    neutral: bool = False


@dataclass(frozen=True)
class CardSpec:
    width_mm: float = 80.0
    height_mm: float = 50.0
    marker_mm: float = 12.0
    margin_mm: float = 2.0
    marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3)   # top-left, top-right, bottom-right, bottom-left
    patches: tuple[Patch, ...] = (
        Patch("white", 16, 17, 8, 8, (245, 245, 245), True),
        Patch("light_grey", 24, 17, 8, 8, (190, 190, 190), True),
        Patch("mid_grey", 32, 17, 8, 8, (128, 128, 128), True),
        Patch("dark_grey", 40, 17, 8, 8, (64, 64, 64), True),
        Patch("skin", 48, 17, 8, 8, (224, 172, 150)),
        Patch("red", 56, 17, 8, 8, (200, 40, 40)),
    )
    ruler_mm: float = 50.0
    ruler_y_mm: float = 30.0

    def marker_corners_mm(self) -> dict[int, np.ndarray]:
        """Corners of each marker in card millimetres, in ArUco order (TL, TR, BR, BL)."""
        m, s, W, H = self.margin_mm, self.marker_mm, self.width_mm, self.height_mm
        origins = [(m, m), (W - m - s, m), (W - m - s, H - m - s), (m, H - m - s)]
        return {mid: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32)
                for mid, (x, y) in zip(self.marker_ids, origins)}


DEFAULT_CARD = CardSpec()


# ------------------------------------------------------------------ colour helpers (sRGB <-> linear)

def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055) * 255.0


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    px = np.clip(np.asarray(rgb, np.float64), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB).astype(np.float64).reshape(-1, 3)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    return lab


@dataclass(frozen=True)
class ColourModel:
    """Per-channel affine correction in linear RGB: lin_out = gain * lin_in + offset (fitted on neutral patches)."""
    gain: np.ndarray      # (3,) RGB
    offset: np.ndarray    # (3,) RGB

    def apply_bgr(self, bgr: np.ndarray) -> np.ndarray:
        lin = srgb_to_linear(bgr[..., ::-1])
        out = linear_to_srgb(lin * self.gain + self.offset)
        return np.round(out[..., ::-1]).astype(np.uint8)

    def apply_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return linear_to_srgb(srgb_to_linear(rgb) * self.gain + self.offset)


def fit_colour_model(measured_rgb: np.ndarray, reference_rgb: np.ndarray) -> ColourModel:
    """Least squares per channel on the neutral patches, in linear light."""
    m, r = srgb_to_linear(measured_rgb), srgb_to_linear(reference_rgb)
    gain, offset = np.ones(3), np.zeros(3)
    for c in range(3):
        a, b = np.polyfit(m[:, c], r[:, c], 1)
        gain[c], offset[c] = max(a, 1e-3), b
    return ColourModel(gain, offset)


# ------------------------------------------------------------------ result

@dataclass(frozen=True)
class Calibration:
    found: bool
    reason: str | None = None
    homography_px_to_mm: np.ndarray | None = None
    px_per_mm: float | None = None
    tilt_ratio: float | None = None         # 1 = camera parallel to the card; large = strongly oblique
    markers_found: int = 0
    card_mask: np.ndarray | None = None     # pixels covered by the card: excluded from skin and wound
    colour: ColourModel | None = None
    colour_error_before: float | None = None   # mean Delta E76 of all patches vs their printed value
    colour_error_after: float | None = None
    patch_rgb: dict = field(default_factory=dict)

    def to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(pts_px, np.float32).reshape(-1, 1, 2), self.homography_px_to_mm).reshape(-1, 2)

    def pixel_area_mm2(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Exact area (mm²) covered by each pixel: |det J| of the homography at the pixel centre.
        For H = [[h11 h12 h13], [h21 h22 h23], [h31 h32 h33]]: |det J(x, y)| = |det H| / (h31 x + h32 y + h33)^3."""
        H = self.homography_px_to_mm
        wgt = H[2, 0] * (xs + 0.5) + H[2, 1] * (ys + 0.5) + H[2, 2]
        return np.abs(np.linalg.det(H)) / np.abs(wgt) ** 3

    def metric_size(self, mask: np.ndarray) -> dict:
        """Area (mm²), length and width (mm) of a mask lying on the card plane.

        Area = sum of the true area of every mask pixel (journal V11: a polygon through boundary-pixel centres loses
        half a pixel on every edge, ~7 % of a 3 mm incision). Length / width = minimum-area rectangle of the pixel
        corners mapped to millimetres."""
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return {}
        area = float(self.pixel_area_mm2(ys, xs).sum())
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        corners = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
        corners = np.vstack([corners + d for d in ((0, 0), (1, 0), (0, 1), (1, 1))])   # outer pixel corners
        (_, _), (w, h), _ = cv2.minAreaRect(self.to_mm(corners).astype(np.float32))
        return {"area_mm2": round(area, 1), "area_cm2": round(area / 100, 2),
                "length_mm": round(float(max(w, h)), 1), "width_mm": round(float(min(w, h)), 1)}

    def public(self) -> dict:
        if not self.found:
            return {"found": False, "reason": self.reason}
        return {"found": True, "markers_found": self.markers_found, "px_per_mm": round(self.px_per_mm, 2),
                "tilt_ratio": round(self.tilt_ratio, 2),
                "colour_corrected": self.colour is not None,
                "colour_error_delta_e": {"before": self.colour_error_before, "after": self.colour_error_after},
                "assumption": "the card lies flat on the skin, in the same plane as the wound"}


NOT_FOUND = Calibration(False, "no calibration card in the photo: sizes are relative to the frame")


# ------------------------------------------------------------------ detector

class CardCalibrator:
    """Finds the card, fits geometry and colour. Stateless after construction (safe across threads)."""

    def __init__(self, spec: CardSpec = DEFAULT_CARD, min_markers: int = 2, max_tilt: float = 2.0,
                 max_colour_error_after: float = 12.0):
        self.spec, self.min_markers, self.max_tilt = spec, min_markers, max_tilt
        self.max_colour_error_after = max_colour_error_after
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners -> better scale
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
        self._corners_mm = spec.marker_corners_mm()

    def calibrate(self, bgr: np.ndarray) -> Calibration:
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return NOT_FOUND
        img_pts, card_pts = [], []
        for c, i in zip(corners, ids.ravel()):
            if int(i) in self._corners_mm:
                img_pts.append(c.reshape(4, 2))
                card_pts.append(self._corners_mm[int(i)])
        n = len(img_pts)
        if n < self.min_markers:
            return Calibration(False, f"only {n} card marker(s) visible: show the whole card", markers_found=n)
        H, _ = cv2.findHomography(np.vstack(img_pts), np.vstack(card_pts), cv2.RANSAC, 3.0)
        if H is None:
            return Calibration(False, "card markers found but inconsistent", markers_found=n)
        H_inv = np.linalg.inv(H)
        px_per_mm, tilt = self._scale_at_centre(H_inv)
        if tilt > self.max_tilt:
            return Calibration(False, f"photo too oblique (tilt ratio {tilt:.1f}): hold the phone parallel to the skin",
                               markers_found=n)
        card_mask = self._card_mask(H_inv, bgr.shape[:2])
        measured = {p.name: self._patch_rgb(bgr, H_inv, p) for p in self.spec.patches}
        colour, before, after = self._fit_colour(measured)
        return Calibration(True, None, H, px_per_mm, tilt, n, card_mask, colour, before, after,
                           {k: [round(float(v), 1) for v in rgb] for k, rgb in measured.items()})

    # ---------------------------------------------------------------- steps
    def _scale_at_centre(self, H_inv: np.ndarray) -> tuple[float, float]:
        """Local scale (px per mm) and anisotropy of the mm->px mapping at the card centre."""
        cx, cy = self.spec.width_mm / 2, self.spec.height_mm / 2
        p = cv2.perspectiveTransform(np.float32([[[cx, cy]], [[cx + 1, cy]], [[cx, cy + 1]]]), H_inv).reshape(3, 2)
        J = np.column_stack([p[1] - p[0], p[2] - p[0]])           # px per mm along card x and y
        sv = np.linalg.svd(J, compute_uv=False)
        return float(np.sqrt(abs(np.linalg.det(J)))), float(sv[0] / max(sv[1], 1e-9))

    def _card_mask(self, H_inv: np.ndarray, shape: tuple[int, int], pad_mm: float = 1.5) -> np.ndarray:
        W, Hh = self.spec.width_mm, self.spec.height_mm
        quad = np.float32([[-pad_mm, -pad_mm], [W + pad_mm, -pad_mm], [W + pad_mm, Hh + pad_mm], [-pad_mm, Hh + pad_mm]])
        px = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        return mask > 0

    def _patch_rgb(self, bgr: np.ndarray, H_inv: np.ndarray, p: Patch, inset_mm: float = 1.5) -> np.ndarray:
        """Median RGB inside the patch (inset to avoid its printed edges and blur)."""
        quad = np.float32([[p.x + inset_mm, p.y + inset_mm], [p.x + p.w - inset_mm, p.y + inset_mm],
                           [p.x + p.w - inset_mm, p.y + p.h - inset_mm], [p.x + inset_mm, p.y + p.h - inset_mm]])
        px = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        sel = bgr[mask > 0]
        return np.median(sel, axis=0)[::-1] if len(sel) else np.full(3, np.nan)

    def _fit_colour(self, measured: dict) -> tuple[ColourModel | None, float | None, float | None]:
        names = [p.name for p in self.spec.patches]
        meas = np.array([measured[n] for n in names])
        ref = np.array([p.rgb for p in self.spec.patches], np.float64)
        if np.isnan(meas).any():
            return None, None, None
        neutral = np.array([p.neutral for p in self.spec.patches])
        model = fit_colour_model(meas[neutral], ref[neutral])
        before = float(np.linalg.norm(_rgb_to_lab(meas) - _rgb_to_lab(ref), axis=1).mean())
        after = float(np.linalg.norm(_rgb_to_lab(model.apply_rgb(meas)) - _rgb_to_lab(ref), axis=1).mean())
        if after > self.max_colour_error_after:      # the fit did not explain the patches: do not trust it
            return None, round(before, 1), round(after, 1)
        return model, round(before, 1), round(after, 1)
