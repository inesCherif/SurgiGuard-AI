"""Segmentation metrics and benchmark (v2 step 3, part 1): exact values on hand-built masks."""

import numpy as np
import pytest

from surgiguard_d1.wound import ClassicalSegmenter, Segmentation
from surgiguard_d1.wound.colour import to_lab
from surgiguard_d1.wound.seg_eval import LabelledImage, SegmenterBenchmark, mean_width, score_masks


def _bar(h=200, w=300, y=95, thick=10, x0=50, x1=250):
    m = np.zeros((h, w), bool)
    m[y:y + thick, x0:x1] = True
    return m


def test_perfect_prediction_scores_one():
    t = _bar()
    s = score_masks(t, t)
    assert (s.dice, s.iou, s.precision, s.recall) == (1.0, 1.0, 1.0, 1.0)
    assert s.width_ratio == pytest.approx(1.0, abs=0.02)


def test_half_overlap_gives_known_dice_and_iou():
    t = _bar(x0=0, x1=100)
    p = _bar(x0=50, x1=150)                  # 50 % of each overlaps
    s = score_masks(p, t)
    assert s.dice == pytest.approx(0.5)
    assert s.iou == pytest.approx(1 / 3)


def test_doubled_width_is_exposed_even_when_dice_looks_fine():
    """Journal V14: a long incision segmented twice too wide still has Dice 0.67 - width_ratio shows 2x."""
    t = _bar(thick=10)
    p = _bar(y=90, thick=20)
    s = score_masks(p, t)
    assert s.recall == 1.0 and s.precision == pytest.approx(0.5)
    assert s.dice == pytest.approx(2 / 3)
    assert s.width_ratio == pytest.approx(2.0, rel=0.05)


def test_nothing_found_on_a_wound_scores_zero_and_empty_on_empty_scores_one():
    t = _bar()
    assert score_masks(None, t).dice == 0.0
    empty = np.zeros_like(t)
    assert score_masks(None, empty).dice == 1.0


def test_shape_mismatch_is_an_error_not_a_silent_score():
    with pytest.raises(ValueError):
        score_masks(np.zeros((10, 10), bool), np.zeros((10, 11), bool))


def test_mean_width_of_a_rotated_bar():
    import cv2
    m = np.zeros((300, 300), np.uint8)
    cv2.line(m, (40, 40), (260, 260), 1, thickness=8)
    assert mean_width(m) == pytest.approx(8, rel=0.15)


class _Oracle:
    """Fake segmenter that returns a fixed mask - lets the benchmark loop be tested without any model."""
    def __init__(self, mask, name="oracle"):
        self.mask, self.name = mask, name

    def segment(self, bgr, ignore=None, px_per_mm=None):
        return Segmentation(mask=self.mask, lab=to_lab(bgr), bgr=bgr, method=self.name)


def test_benchmark_compares_any_segmenters_on_the_same_data():
    t = _bar()
    img = np.full(t.shape + (3,), 180, np.uint8)
    bench = SegmenterBenchmark([LabelledImage("a", img, t), LabelledImage("b", img, t)])
    perfect, blind = _Oracle(t, "perfect"), _Oracle(None, "blind")
    good, bad = bench.compare(perfect, blind)
    assert good["dice_mean"] == 1.0 and good["misses"] == 0
    assert bad["dice_mean"] == 0.0 and bad["misses"] == 2


def test_classical_segmenter_runs_through_the_benchmark_on_a_synthetic_incision():
    rng = np.random.default_rng(0)
    h, w = 400, 600
    img = np.clip(np.array([150, 170, 215]) + rng.normal(0, 5, (h, w, 3)), 0, 255).astype(np.uint8)
    t = np.zeros((h, w), bool)
    t[190:205, 150:450] = True
    img[t] = (60, 50, 150)
    report = SegmenterBenchmark([LabelledImage("cut", img, t, px_per_mm=5.0)]).run(ClassicalSegmenter())
    s = report.summary()
    assert s["segmenter"] == "classical-local-contrast"
    assert s["misses"] == 0
    assert s["dice_mean"] > 0.5            # finds the cut; exact quality is what step 3 must beat
