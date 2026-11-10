"""WoundCropper (v2 step 4): square box with context, kept inside the photo, honest fallback."""

import numpy as np

from surgiguard_d1.wound import Segmentation
from surgiguard_d1.wound.colour import to_lab
from surgiguard_d1.wound.crop import WoundCropper, square_box_around


class _Fixed:
    name = "fixed"

    def __init__(self, mask):
        self.mask = mask

    def segment(self, bgr, ignore=None, px_per_mm=None):
        return Segmentation(mask=self.mask, lab=to_lab(bgr), bgr=bgr, reason=None if self.mask is not None else "none")


def test_box_is_square_centred_with_margin():
    m = np.zeros((400, 600), bool)
    m[190:210, 250:350] = True                       # 100 x 20 incision centred on (300, 200)
    x0, y0, x1, y1 = square_box_around(m, margin=0.35, min_side_frac=0.1)
    assert x1 - x0 == y1 - y0 == 170                 # 100 * (1 + 2 * 0.35)
    assert abs((x0 + x1) / 2 - 300) <= 1 and abs((y0 + y1) / 2 - 200) <= 1


def test_box_near_the_border_is_shifted_inside_not_shrunk():
    m = np.zeros((400, 600), bool)
    m[5:15, 2:40] = True
    x0, y0, x1, y1 = square_box_around(m, margin=0.35, min_side_frac=0.25)
    assert (x0, y0) == (0, 0) and x1 - x0 == y1 - y0 == 100      # min side = 0.25 * 400


def test_tiny_wound_gets_at_least_the_minimum_context():
    m = np.zeros((400, 600), bool)
    m[200:203, 300:303] = True
    x0, y0, x1, y1 = square_box_around(m, margin=0.35, min_side_frac=0.25)
    assert x1 - x0 == 100


def test_cropper_falls_back_to_the_whole_photo_and_says_so():
    img = np.zeros((120, 160, 3), np.uint8)
    c = WoundCropper(_Fixed(None)).crop(img)
    assert not c.found and c.box == (0, 0, 160, 120) and c.apply(img).shape == img.shape


def test_cropper_uses_the_injected_segmenter():
    img = np.zeros((300, 300, 3), np.uint8)
    m = np.zeros((300, 300), bool)
    m[100:120, 100:200] = True
    c = WoundCropper(_Fixed(m), margin=0.25, min_side_frac=0.1).crop(img)
    assert c.found and c.apply(img).shape[:2] == (150, 150)
