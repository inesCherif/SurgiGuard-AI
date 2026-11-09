"""Architecture guarantees of the wound package (v2 step 1): Strategy injection, no shared state, one config."""

import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from surgiguard_d1.wound import ClassicalSegmenter, Segmentation, WoundAnalyzer, analyze
from surgiguard_d1.wound.colour import to_lab

PHOTOS = Path(__file__).resolve().parents[2] / "backend" / "seed" / "photos"


def _skin_jpeg(size=320) -> bytes:
    rng = np.random.default_rng(0)
    img = np.clip(np.array([200, 160, 140]) + rng.normal(0, 6, (size, size, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=95)
    return buf.getvalue()


class _FixedSquareSegmenter:
    """A test double: proves the pipeline only depends on the Segmenter interface (Dependency Inversion)."""
    name = "fixed-square"

    def segment(self, bgr, ignore=None, px_per_mm=None):
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), bool)
        mask[h // 3:2 * h // 3, w // 3:2 * w // 3] = True
        lab = to_lab(bgr)
        return Segmentation(mask, lab, bgr, method="fixed square", skin=~mask, reference_lab=np.median(lab[~mask], axis=0))


def test_any_segmenter_can_be_injected_without_touching_the_pipeline():
    res = WoundAnalyzer(segmenter=_FixedSquareSegmenter()).analyze(_skin_jpeg())
    assert res["segmentation"]["found"] and res["segmentation"]["segmenter"] == "fixed-square"
    assert 10 < res["measurements"]["wound_area_pct_of_frame"] < 12      # the 1/3 x 1/3 square
    assert abs(res["measurements"]["redness_mean_delta_a"]) < 2          # uniform skin: no redness


def test_default_segmenter_is_the_classical_one():
    assert isinstance(WoundAnalyzer().segmenter, ClassicalSegmenter)


def test_staple_count_is_returned_not_kept_in_global_state():
    """v1 stored the count on the function object (_staple_chain.last_count): concurrent requests could mix it up."""
    staples = analyze((PHOTOS / "day1.jpg").read_bytes())["measurements"]["closure_material"]["staple_like_blobs"]
    plain = analyze(_skin_jpeg())
    assert staples > 5
    assert plain["measurements"] is None or plain["measurements"]["closure_material"]["staple_like_blobs"] != staples


def test_concurrent_analyses_give_the_same_results_as_sequential_ones():
    photos = [(PHOTOS / f"day{i}.jpg").read_bytes() for i in (1, 2, 3)]
    expected = [analyze(p)["measurements"] for p in photos]
    got = [None] * 3

    def run(i):
        got[i] = analyze(photos[i])["measurements"]
    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == expected


def test_comparison_reports_both_red_areas_for_the_patient_view():
    day1 = (PHOTOS / "day1.jpg").read_bytes()
    cmp = analyze(day1, reference=day1)["comparison"]       # same photo: registration succeeds
    assert cmp["available"] and cmp["trend"] == "stable_colour"
    assert cmp["redness_area_ref_pct"] == cmp["redness_area_now_pct"]
