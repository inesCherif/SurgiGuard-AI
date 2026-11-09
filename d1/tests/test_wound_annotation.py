"""Click-to-outline logic and the shared mask-folder layout (v2 step 3, part 2)."""

import numpy as np
import pytest

from surgiguard_d1.wound.annotation import Annotation, Polygon, Stroke
from surgiguard_d1.wound.seg_data import MaskFolder, write_png
from surgiguard_d1.wound.seg_eval import mean_width


def test_stroke_gives_a_bar_of_the_chosen_width():
    a = Annotation(200, 300, tool="stroke", stroke_width=10)
    a.click(50, 100)
    a.click(250, 100)
    assert a.finish_shape()
    m = a.mask()
    assert m[100, 150] and not m[120, 150]
    assert mean_width(m) == pytest.approx(10, rel=0.15)


def test_polygon_fills_its_inside():
    a = Annotation(100, 100, tool="polygon")
    for p in [(10, 10), (60, 10), (60, 60), (10, 60)]:
        a.click(*p)
    a.finish_shape()
    m = a.mask()
    assert m[30, 30] and not m[80, 80]
    assert m.sum() == pytest.approx(51 * 51, rel=0.02)


def test_mask_is_the_union_of_several_shapes():
    a = Annotation(100, 200, tool="stroke", stroke_width=6)
    a.click(10, 20); a.click(90, 20); a.finish_shape()
    a.click(110, 80); a.click(190, 80); a.finish_shape()
    m = a.mask()
    assert m[20, 50] and m[80, 150] and not m[50, 100]


def test_undo_removes_points_then_reopens_the_last_shape():
    a = Annotation(100, 100)
    a.click(10, 10); a.click(50, 50); a.finish_shape()
    assert len(a.done) == 1
    a.undo()                                   # reopens the stroke and drops its last point
    assert not a.done and a.current.points == [(10, 10)]
    a.undo()
    assert a.is_empty()


def test_too_short_shapes_are_dropped_and_clicks_are_clamped():
    a = Annotation(50, 50, tool="polygon")
    a.click(-5, 999); a.click(10, 10)
    assert a.current.points[0] == (0, 49)
    assert not a.finish_shape() and a.is_empty()


def test_width_change_applies_to_the_stroke_being_drawn_and_unknown_tool_is_rejected():
    a = Annotation(50, 50, stroke_width=10)
    a.click(5, 5)
    a.change_width(+4)
    assert isinstance(a.current, Stroke) and a.current.width == 14
    with pytest.raises(ValueError):
        a.set_tool("lasso")
    a.set_tool("polygon")
    a.click(1, 1)
    assert isinstance(a.current, Polygon)


def test_mask_folder_round_trip_and_resume(tmp_path):
    store = MaskFolder(tmp_path / "set")
    img = np.full((40, 60, 3), 128, np.uint8)
    for name in ("a", "b"):
        write_png(store.images_dir / f"{name}.png", img)
    assert len(store) == 0 and len(store.unlabelled()) == 2
    truth = np.zeros((40, 60), bool); truth[10:20, 5:50] = True
    store.write_mask(store.image_paths()[0], truth)
    assert [p.stem for p in store.unlabelled()] == ["b"]
    (sample,) = list(store)
    assert sample.name == "a" and np.array_equal(sample.truth, truth)


def test_mask_of_the_wrong_size_is_an_error(tmp_path):
    store = MaskFolder(tmp_path)
    write_png(store.images_dir / "x.png", np.zeros((40, 60, 3), np.uint8))
    store.write_mask(store.images_dir / "x.png", np.zeros((10, 10), bool))
    with pytest.raises(ValueError):
        list(store)
