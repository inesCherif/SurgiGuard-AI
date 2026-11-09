"""Training building blocks (v2 step 3, part 3). Skipped when PyTorch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from surgiguard_d1.wound.seg_eval import LabelledImage  # noqa: E402
from surgiguard_d1.wound.training.data import SegDataset, augment, prepare, split  # noqa: E402
from surgiguard_d1.wound.training.unet import ResNetUNet, dice_bce_loss  # noqa: E402


def _sample(name="a", h=120, w=160):
    img = np.full((h, w, 3), (150, 170, 215), np.uint8)
    truth = np.zeros((h, w), bool)
    truth[55:65, 20:140] = True
    img[truth] = (60, 50, 150)
    return LabelledImage(name, img, truth)


def test_unet_output_has_the_input_resolution():
    m = ResNetUNet(pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 1, 64, 64)


def test_loss_is_low_for_a_perfect_prediction_and_high_for_the_inverse():
    y = torch.zeros(1, 1, 8, 8)
    y[..., 2:5, :] = 1
    good = dice_bce_loss((y * 2 - 1) * 20, y)
    bad = dice_bce_loss((1 - 2 * y) * 20, y)
    assert good.item() < 0.01 and bad.item() > 1.5


def test_augment_moves_photo_and_mask_together_and_keeps_the_mask_binary():
    (img, mask), = prepare([_sample()], 96)
    rng = np.random.default_rng(3)
    for _ in range(5):
        a_img, a_mask = augment(img, mask, rng)
        assert set(np.unique(a_mask)) <= {0, 1}
        if a_mask.any():               # the red cut must still lie under the mask after the same warp
            red_under = (a_img[..., 2].astype(int) - a_img[..., 1])[a_mask > 0].mean()
            assert red_under > 60


def test_split_is_deterministic_and_disjoint():
    s = [_sample(str(i)) for i in range(20)]
    tr1, va1 = split(s, 0.1, seed=0)
    tr2, va2 = split(s, 0.1, seed=0)
    assert [x.name for x in va1] == [x.name for x in va2] and len(va1) == 2
    assert not {x.name for x in tr1} & {x.name for x in va1}


def test_a_few_steps_overfit_one_image():
    """Sanity check every training pipeline should pass: loss must fall on a single repeated sample."""
    torch.manual_seed(0)
    ds = SegDataset(prepare([_sample()], 64), train=False)
    x, y = ds[0]
    x, y = x[None], y[None]
    m = ResNetUNet(pretrained=False)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = None
    for _ in range(60):          # from random weights: 1.49 -> 0.71 measured
        opt.zero_grad()
        loss = dice_bce_loss(m(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.6
