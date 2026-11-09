"""Where is the wound? Segmentation behind a Strategy interface.

`Segmenter` is the abstraction the pipeline depends on (Dependency Inversion). Today's implementation is the
classical, explainable `ClassicalSegmenter`; a learned model (v2 step 3) only has to implement the same
`segment()` method to replace it, without touching quality, measurement, registration or the backend
(Open/Closed). The result is an immutable `Segmentation` value object instead of an untyped dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from .colour import dilate, ellipse, skin_mask, to_lab
from .config import DEFAULT_KERNELS, DEFAULT_LIMITS, Kernels, Limits


@dataclass(frozen=True)
class Segmentation:
    """Outcome of a segmenter. `mask is None` means nothing reliable was found and `reason` says why."""
    mask: np.ndarray | None
    lab: np.ndarray
    bgr: np.ndarray
    reason: str | None = None
    method: str | None = None
    skin: np.ndarray | None = None
    reference_lab: np.ndarray | None = None      # median Lab of the patient's own skin around the wound
    reference_std: np.ndarray | None = None
    too_large: bool = False
    staple_count: int | None = None              # set when the incision was found from a row of staples
    extras: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.mask is not None


class Segmenter(Protocol):
    """Strategy interface: any wound segmenter (classical, learned, ...)."""
    name: str

    def segment(self, bgr: np.ndarray, ignore: np.ndarray | None = None,
                px_per_mm: float | None = None) -> Segmentation:
        """ignore: pixels that are not skin by construction (the calibration card).
        px_per_mm: known scale (calibration card); segmenters that use pixel sizes should convert them."""


@dataclass(frozen=True)
class _StapleChain:
    strip: np.ndarray
    count: int


def robust_z(x: np.ndarray, where: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD) over `where`: a z-score that outliers (the wound itself) cannot distort."""
    v = x[where]
    if not v.size:
        return x / 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-3
    return (x - med) / mad


class ClassicalSegmenter:
    """Local-contrast segmentation in CIELAB + staple-row detection. No learning, every step explainable.

    Idea: lighting is never uniform in a phone photo, so each pixel is compared with its OWN local background
    (a large Gaussian blur of the same channel) instead of a global threshold.
    """
    name = "classical-local-contrast"

    def __init__(self, limits: Limits = DEFAULT_LIMITS, kernels: Kernels = DEFAULT_KERNELS):
        self.limits, self.k = limits, kernels

    # ---------------------------------------------------------------- public
    def segment(self, bgr: np.ndarray, ignore: np.ndarray | None = None,
                px_per_mm: float | None = None) -> Segmentation:
        if px_per_mm:   # calibrated: same kernel sizes in millimetres whatever the camera distance
            return ClassicalSegmenter(self.limits, self.k.scaled(px_per_mm))._segment(bgr, ignore)
        return self._segment(bgr, ignore)

    def _segment(self, bgr: np.ndarray, ignore: np.ndarray | None) -> Segmentation:
        # No global white balance: when skin fills the frame, grey-world would turn the skin grey. Every measure
        # is relative to the surrounding skin of the same photo, which already cancels the colour of the light.
        lab = to_lab(bgr)
        skin = skin_mask(bgr, self.k)
        if ignore is not None:          # the card's white/black/patches must never count as skin or wound
            skin &= ~ignore
        if skin.mean() < self.limits.min_skin_fraction * 0.5:
            return Segmentation(None, lab, bgr, reason="no skin found around the site")
        near_skin = cv2.dilate(skin.astype(np.uint8), np.ones((self.k.near_skin_dilate,) * 2, np.uint8)) > 0
        if ignore is not None:
            near_skin &= ~ignore
        score = self._score(bgr, lab, skin, near_skin, ignore)
        thr = max(float(np.percentile(score[near_skin], 96)), 3.0)

        chain = self._staple_chain(bgr, near_skin)
        if chain is not None:
            # A row of staples marks the incision: keep the strip they outline + abnormal pixels right along it.
            along = dilate(chain.strip, ellipse(self.k.staple_along))
            mask = chain.strip | ((score > thr * 0.7) & along)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(self.k.staple_close)) > 0
            return self._finish(mask, lab, skin, bgr, "staple row + local redness/darkness along it", chain.count)

        mask = self._strongest_elongated(score, thr)
        if mask is None:
            return Segmentation(None, lab, bgr, reason="nothing stands out from the surrounding skin")
        return self._finish(mask, lab, skin, bgr, "local redness/darkness, strongest elongated structure")

    # ---------------------------------------------------------------- steps
    def _score(self, bgr, lab, skin, near_skin, ignore=None) -> np.ndarray:
        """Per-pixel "abnormality": local redness + 0.7 local darkness, ink excluded, plus bright metal."""
        h, w = lab.shape[:2]
        sigma = max(h, w) / 18.0
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        background = _background(sigma, ignore)
        a_loc = a - background(a)
        dark_loc = background(L) - L
        b_loc = b - background(b)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        metal = ((hsv[..., 2] > 200) & (hsv[..., 1] < 50)).astype(np.float32)
        # Surgical-marker ink and bruises are purple/blue (b* drops); wound redness is red-orange.
        not_ink = (b_loc > -6).astype(np.float32)
        score = (np.maximum(robust_z(a_loc, skin), 0) + 0.7 * np.maximum(robust_z(dark_loc, skin), 0)) * not_ink + 3.0 * metal
        score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 1.5)
        score[~near_skin] = 0
        border = np.zeros((h, w), bool)
        border[int(h * .04):int(h * .96), int(w * .04):int(w * .96)] = True
        score[~border] = 0
        return score

    def _staple_chain(self, bgr: np.ndarray, near_skin: np.ndarray) -> _StapleChain | None:
        """Metal staples: small bright, grey, LOCALLY brighter marks in a row (top-hat filter + chaining + curve fit)."""
        h, w = near_skin.shape
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, ellipse(self.k.staple_tophat))
        bright = ((tophat > 35) & (hsv[..., 1] < 90) & near_skin).astype(np.uint8)
        n, labels, st, cen = cv2.connectedComponentsWithStats(bright, 8)
        blobs = [i for i in range(1, n) if 3 <= st[i, cv2.CC_STAT_AREA] <= 250]
        if len(blobs) < 5:
            return None
        pts = cen[blobs]
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, np.inf)
        spacing = max(h, w) * 0.09
        chained = [k for k in range(len(blobs)) if (d[k] < spacing).sum() >= 1]
        if len(chained) < 5:
            return None
        best = _largest_component(d[np.ix_(chained, chained)] < spacing)
        if len(best) < 5:
            return None
        members = [chained[k] for k in best]
        chain_pts = pts[members]
        keep = _fit_smooth_curve(chain_pts, h, w)
        if keep is None:
            return None
        strip = np.zeros((h, w), np.uint8)
        horizontal = np.ptp(chain_pts[:, 0]) > np.ptp(chain_pts[:, 1])
        u = chain_pts[:, 0] if horizontal else chain_pts[:, 1]
        poly = chain_pts[keep][np.argsort(u[keep])].astype(np.int32)
        cv2.polylines(strip, [poly], False, 1, thickness=max(9, int(max(h, w) * 0.025)))
        for k, ok in zip(members, keep):
            if ok:
                strip[labels == blobs[k]] = 1
        return _StapleChain(strip > 0, int(keep.sum()))

    def _strongest_elongated(self, score: np.ndarray, thr: float) -> np.ndarray | None:
        """Rank connected components by area x elongation x strength x centrality; merge pieces on the same line."""
        h, w = score.shape
        cand = cv2.morphologyEx((score > thr).astype(np.uint8), cv2.MORPH_CLOSE, ellipse(self.k.candidate_close))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
        comps = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 30:
                continue
            pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.float32)
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(pts)
            elong = max(rw, rh) / max(min(rw, rh), 1.0)
            dist = np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2)
            strength = float(score[labels == i].mean())
            comps.append((area * min(elong, 8.0) * strength * (1.0 - 0.7 * dist), i, pts))
        if not comps:
            return None
        comps.sort(key=lambda c: -c[0])
        _, best, pts = comps[0]
        keep = labels == best
        # An incision often shows up broken: merge pieces lying on the line, refitting it as it grows.
        merged, rest = [pts], [c for c in comps[1:] if c[0] >= comps[0][0] * 0.02]
        changed = True
        while changed and rest:
            changed = False
            vx, vy, x0, y0 = cv2.fitLine(np.vstack(merged), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            for c in list(rest):
                if np.abs((c[2][:, 0] - x0) * vy - (c[2][:, 1] - y0) * vx).mean() < max(h, w) * 0.05:
                    keep |= labels == c[1]
                    merged.append(c[2])
                    rest.remove(c)
                    changed = True
        return dilate(keep, ellipse(self.k.mask_dilate))

    def _finish(self, mask, lab, skin, bgr, how: str, staple_count: int | None = None) -> Segmentation:
        return finish_segmentation(mask, lab, skin, bgr, how, self.limits, self.k, staple_count)


def finish_segmentation(mask, lab, skin, bgr, how: str, limits: Limits, kernels: Kernels,
                        staple_count: int | None = None) -> Segmentation:
    """Size sanity checks + the peri-wound skin reference (the patient's own skin, a band 25-60 px away).

    Shared by every segmenter (classical, learned) so the measurement stage always receives the same fields
    computed the same way, whatever drew the mask."""
    frac = float(mask.mean())
    if frac < limits.min_wound_fraction:
        return Segmentation(None, lab, bgr, reason="abnormal region too small to measure reliably")
    band = dilate(mask, ellipse(kernels.reference_band_outer)) & ~dilate(mask, ellipse(kernels.reference_band_inner)) & skin
    if band.sum() < 100:
        band = skin & ~mask
    if band.sum() == 0:                     # no skin at all around (e.g. a learned mask on a close-up)
        band = ~mask
    return Segmentation(mask, lab, bgr, method=how, skin=skin, reference_lab=np.median(lab[band], axis=0),
                        reference_std=lab[band].std(axis=0), too_large=frac > limits.max_wound_fraction,
                        staple_count=staple_count)


# ------------------------------------------------------------------ small pure helpers

def _background(sigma: float, ignore: np.ndarray | None):
    """Local background estimator: a Gaussian blur, or a NORMALIZED CONVOLUTION when some pixels must be ignored.

    blur(x * w) / blur(w) with w = 0 on ignored pixels (e.g. the white calibration card): without it, the bright
    card leaks into the background of the skin next to it, which then looks "darker than its surroundings" and is
    segmented as a wound (journal V12).
    """
    if ignore is None or not ignore.any():
        return lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    weight = (~ignore).astype(np.float32)
    norm = np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)
    return lambda x: cv2.GaussianBlur(x * weight, (0, 0), sigma) / norm


def _largest_component(adj: np.ndarray) -> list[int]:
    """Largest connected group in a boolean adjacency matrix (iterative DFS)."""
    seen, best = set(), []
    for s0 in range(len(adj)):
        if s0 in seen:
            continue
        stack, group = [s0], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            group.append(u)
            stack.extend(np.nonzero(adj[u])[0].tolist())
        if len(group) > len(best):
            best = group
    return best


def _fit_smooth_curve(chain_pts: np.ndarray, h: int, w: int) -> np.ndarray | None:
    """An incision is a smooth curve: fit a quadratic 3 times, dropping off-curve highlights each time."""
    horizontal = np.ptp(chain_pts[:, 0]) > np.ptp(chain_pts[:, 1])
    u, v = (chain_pts[:, 0], chain_pts[:, 1]) if horizontal else (chain_pts[:, 1], chain_pts[:, 0])
    keep = np.ones(len(u), bool)
    for _ in range(3):
        coef = np.polyfit(u[keep], v[keep], 2)
        res = np.abs(np.polyval(coef, u) - v)
        keep = res <= max(2.5 * np.median(res[keep]), max(h, w) * 0.015)
        if keep.sum() < 5:
            return None
    return keep
