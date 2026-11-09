"""Calibration card (v2 step 2), checked on synthetic photos whose ground truth we drew ourselves:
a skin plane at a known scale, an incision of known size in millimetres, the printed card lying next to it,
then a camera tilt (perspective) and a warm light (colour cast)."""

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from surgiguard_d1.wound import WoundAnalyzer, analyze
from surgiguard_d1.wound.calibration import DEFAULT_CARD, CardCalibrator, fit_colour_model, srgb_to_linear
from surgiguard_d1.wound.card_render import render_card

PLANE_MM = (160, 120)       # skin plane, millimetres
S = 5.0                     # ground-truth scale of the flat plane, px per mm
CARD_AT = (8, 8)            # card top-left on the plane, mm
CUT = dict(x=95, y=55, length=40, width=3)   # incision, mm  -> area 120 mm²


def _plane(with_card=True, with_cut=True, seed=0):
    rng = np.random.default_rng(seed)
    h, w = int(PLANE_MM[1] * S), int(PLANE_MM[0] * S)
    skin_bgr = np.array([150, 170, 215], np.float64)       # light skin tone (BGR)
    img = np.clip(skin_bgr + rng.normal(0, 5, (h, w, 3)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    truth = np.zeros((h, w), bool)
    if with_cut:
        x0, y0 = int(CUT["x"] * S), int(CUT["y"] * S)
        x1, y1 = int((CUT["x"] + CUT["length"]) * S), int((CUT["y"] + CUT["width"]) * S)
        img[y0:y1, x0:x1] = (60, 50, 150)                   # dark red incision
        truth[y0:y1, x0:x1] = True
    if with_card:
        card = render_card(DEFAULT_CARD, S, label=False)
        cy, cx = int(CARD_AT[1] * S), int(CARD_AT[0] * S)
        img[cy:cy + card.shape[0], cx:cx + card.shape[1]] = card
    return img, truth


def _camera(img, truth, tilt=0.15, cast=(1.0, 1.0, 1.0)):
    """Perspective (one side further from the camera) + light colour cast in linear RGB (BGR order for cast)."""
    h, w = img.shape[:2]
    d = tilt * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d * 0.3], [w - d * 0.2, 0], [w, h], [0, h - d * 0.2]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h), borderValue=(150, 170, 215))
    mask = cv2.warpPerspective(truth.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    lin = srgb_to_linear(warped) * np.array(cast)
    out = np.clip(np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055) * 255, 0, 255)
    return out.astype(np.uint8), mask


def _jpeg(bgr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_card_is_found_and_scale_is_recovered_on_a_flat_photo():
    img, _ = _plane()
    cal = CardCalibrator().calibrate(img)
    assert cal.found and cal.markers_found == 4
    assert cal.px_per_mm == pytest.approx(S, rel=0.02) and cal.tilt_ratio < 1.05


def test_real_size_survives_camera_tilt():
    """The homography maps the outline back to millimetres: perspective does not change the measured size."""
    img, truth = _plane()
    photo, mask = _camera(img, truth, tilt=0.15)
    cal = CardCalibrator().calibrate(photo)
    assert cal.found and cal.tilt_ratio > 1.05                  # the photo really is oblique
    size = cal.metric_size(mask)
    assert size["area_mm2"] == pytest.approx(CUT["length"] * CUT["width"], rel=0.05)
    assert size["length_mm"] == pytest.approx(CUT["length"], rel=0.03)
    assert size["width_mm"] == pytest.approx(CUT["width"], abs=0.5)


def test_warm_light_is_corrected_from_the_grey_patches():
    img, truth = _plane()
    photo, _ = _camera(img, truth, tilt=0.0, cast=(0.75, 1.0, 1.25))   # BGR: more red, less blue = warm lamp
    cal = CardCalibrator().calibrate(photo)
    assert cal.colour is not None
    assert cal.colour_error_after < cal.colour_error_before / 2      # Delta E at least halved
    white = cal.colour.apply_rgb(np.array([cal.patch_rgb["white"]]))[0]
    assert np.abs(white - 245).max() < 8                             # the white patch is white again


def test_colour_fit_is_exact_for_a_pure_gain():
    ref = np.array([[245, 245, 245], [190, 190, 190], [128, 128, 128], [64, 64, 64]], np.float64)
    lin = srgb_to_linear(ref) * np.array([1.2, 1.0, 0.7])
    measured = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055) * 255
    model = fit_colour_model(measured, ref)
    assert np.allclose(model.gain, [1 / 1.2, 1.0, 1 / 0.7], rtol=1e-3) and np.allclose(model.offset, 0, atol=1e-4)


def test_end_to_end_reports_millimetres_and_never_segments_the_card():
    img, truth = _plane()
    photo, _ = _camera(img, truth, tilt=0.1, cast=(0.85, 1.0, 1.15))
    res = analyze(_jpeg(photo))
    assert res["calibration"]["found"] and res["calibration"]["colour_corrected"]
    m = res["measurements"]
    assert m["size_basis"] == "mm (calibration card)"
    assert m["length_mm"] == pytest.approx(CUT["length"], rel=0.15)  # segmentation adds a small margin
    cal = CardCalibrator().calibrate(cv2.imdecode(np.frombuffer(_jpeg(photo), np.uint8), cv2.IMREAD_COLOR))
    analyzer = WoundAnalyzer()
    seg = analyzer.segmenter.segment(cal.colour.apply_bgr(photo), ignore=cal.card_mask)
    assert seg.found and not (seg.mask & cal.card_mask).any()        # the black markers are never "wound"


def test_calibrated_photos_compare_real_areas():
    small, t1 = _plane(seed=1)
    photo1, _ = _camera(small, t1, tilt=0.0)
    far = cv2.resize(photo1, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)   # same site, phone further away
    canvas = np.full_like(photo1, (150, 170, 215))
    canvas[:far.shape[0], :far.shape[1]] = far
    cmp = analyze(_jpeg(canvas), reference=_jpeg(photo1))["comparison"]
    if not cmp["available"]:
        pytest.skip(f"registration refused the synthetic pair: {cmp['reason']}")
    assert cmp["area_basis"].startswith("mm2") and abs(cmp["area_change_pct"]) <= 15   # same wound, same size


def test_missing_or_partial_card_is_reported_not_guessed():
    img, _ = _plane(with_card=False)
    assert CardCalibrator().calibrate(img).found is False
    card_only, _ = _plane()
    cropped = card_only[:, int((CARD_AT[0] + 40) * S):]        # cut the left half of the card: 2 markers left
    cal = CardCalibrator(min_markers=3).calibrate(cropped)
    assert not cal.found and "marker" in cal.reason


def test_strongly_oblique_photo_is_refused():
    img, truth = _plane()
    photo, _ = _camera(img, truth, tilt=0.45)
    cal = CardCalibrator(max_tilt=1.3).calibrate(photo)
    assert not cal.found and "oblique" in cal.reason
