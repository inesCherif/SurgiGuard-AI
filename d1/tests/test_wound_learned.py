"""LearnedSegmenter (v2 step 3, part 3): preprocessing contract, post-processing, pipeline injection, real ONNX."""

import json

import numpy as np
import pytest

from surgiguard_d1.wound import WoundAnalyzer
from surgiguard_d1.wound.learned import LearnedSegmenter, letterbox, normalise, unletterbox
from surgiguard_d1.wound.seg_eval import score_masks


def _photo(h=300, w=400):
    rng = np.random.default_rng(1)
    img = np.clip(np.array([150, 170, 215]) + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    truth = np.zeros((h, w), bool)
    truth[140:155, 80:320] = True
    img[truth] = (60, 50, 150)
    return img, truth


class _RednessSession:
    """Stands in for onnxruntime: 'logits' = how red the (normalised) input pixel is (skin ~ -2.5, cut ~ +1.4). Same .run/.get_inputs API."""
    class _In:
        name = "image"

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        x = feeds["image"]                       # 1x3xSxS, RGB normalised
        r, g = x[0, 0], x[0, 1]
        return [((r - g) * 4.0 - 5.0)[None, None].astype(np.float32)]


def test_letterbox_keeps_aspect_ratio_and_inverts_exactly():
    img, truth = _photo(300, 400)
    sq, lb = letterbox(img, 320)
    assert sq.shape == (320, 320, 3) and lb.top == 40 and lb.left == 0      # 400->320, 300->240, 40 px pads
    back = unletterbox(letterbox(truth.astype(np.uint8)[..., None], 320)[0][..., 0].astype(np.float32), lb) > 0.5
    assert score_masks(back, truth).dice > 0.95


def test_normalise_layout():
    x = normalise(np.zeros((8, 8, 3), np.uint8))
    assert x.shape == (1, 3, 8, 8) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx(-0.485 / 0.229, rel=1e-5)


def test_learned_segmenter_finds_the_cut_and_fills_the_measurement_fields(tmp_path):
    img, truth = _photo()
    seg = LearnedSegmenter(tmp_path / "m.onnx", session=_RednessSession()).segment(img)
    assert seg.found and score_masks(seg.mask, truth).dice > 0.9
    assert seg.reference_lab is not None and seg.skin is not None


def test_ignored_pixels_and_specks_are_removed(tmp_path):
    img, truth = _photo()
    img[10:13, 10:13] = (60, 50, 150)                       # a 3x3 red speck
    ignore = np.zeros(truth.shape, bool)
    ignore[:, 200:] = True
    seg = LearnedSegmenter(tmp_path / "m.onnx", session=_RednessSession()).segment(img, ignore=ignore)
    assert not seg.mask[:, 200:].any() and not seg.mask[10:13, 10:13].any()


def test_model_card_is_read_and_nothing_found_is_reported(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"input_size": 256, "threshold": 0.99, "version": "t1"}))
    s = LearnedSegmenter(tmp_path / "m.onnx", session=_RednessSession())
    assert s.name == "learned-unet:t1" and s.card.input_size == 256
    blank = np.full((200, 200, 3), 180, np.uint8)
    seg = s.segment(blank)
    assert not seg.found and seg.reason == "the model found no wound"


def test_injected_into_the_analyser_without_touching_it(tmp_path):
    import cv2
    img, _ = _photo()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = WoundAnalyzer(segmenter=LearnedSegmenter(tmp_path / "m.onnx", session=_RednessSession())).analyze(buf.tobytes())
    assert out["segmentation"]["found"], out["segmentation"]
    assert out["segmentation"]["segmenter"].startswith("learned-unet")
    assert out["measurements"] is not None


def test_real_onnx_model_runs_through_onnxruntime(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")                 # needed by torch.onnx.export; present in the training venv

    class Red(torch.nn.Module):                 # tiny fixed "network": logits = 4*(R-G) - 5, as the fake above
        def forward(self, x):
            return (x[:, 0:1] - x[:, 1:2]) * 4.0 - 5.0

    path = tmp_path / "red.onnx"
    torch.onnx.export(Red(), torch.zeros(1, 3, 320, 320), str(path), input_names=["image"], output_names=["logits"])
    img, truth = _photo()
    real = LearnedSegmenter(path).segment(img)
    fake = LearnedSegmenter(path, session=_RednessSession()).segment(img)
    assert np.array_equal(real.mask, fake.mask)
    assert score_masks(real.mask, truth).dice > 0.9


def test_small_wound_found_by_the_model_is_kept_journal_v19(tmp_path):
    """A 0.05 %-of-the-frame ulcer: the classical floor (0.2 %) would discard it, the learned floor keeps it."""
    from surgiguard_d1.wound.learned import LEARNED_LIMITS
    from surgiguard_d1.wound.config import DEFAULT_LIMITS
    img, _ = _photo()
    img[:] = (150, 170, 215)
    truth = np.zeros(img.shape[:2], bool)
    truth[150:156, 200:210] = True                  # 60 px of 120 000
    img[truth] = (60, 50, 150)
    assert DEFAULT_LIMITS.min_wound_fraction > truth.mean() > LEARNED_LIMITS.min_wound_fraction
    seg = LearnedSegmenter(tmp_path / "m.onnx", session=_RednessSession()).segment(img)
    assert seg.found and score_masks(seg.mask, truth).dice > 0.8
    strict = LearnedSegmenter(tmp_path / "m.onnx", limits=DEFAULT_LIMITS, session=_RednessSession()).segment(img)
    assert not strict.found
