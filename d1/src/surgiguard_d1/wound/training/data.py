"""Training samples: letterboxed with the SAME function as inference, augmented on the fly.

Augmentations reflect what changes between home phone photos: framing (scale, rotation, shift, flips) and light
(brightness, contrast, small hue shift). Geometry is applied identically to photo and mask; the mask is warped
with nearest-neighbour so it stays binary.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..learned import letterbox, normalise
from ..seg_eval import LabelledImage


def prepare(samples, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Letterbox once, keep in RAM (1 000 photos at 320 px ~ 300 MB)."""
    out = []
    for s in samples:
        img, _ = letterbox(s.bgr, size)
        m, _ = letterbox(s.truth.astype(np.uint8)[..., None], size)
        out.append((img, m[..., 0]))
    return out


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    angle = rng.uniform(-25, 25)
    scale = rng.uniform(0.75, 1.3)
    tx, ty = rng.uniform(-0.1, 0.1, 2) * (w, h)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180           # small hue shift only: colour is a signal
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-15, 15)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, np.ascontiguousarray(mask)


class SegDataset(Dataset):
    def __init__(self, prepared: list[tuple[np.ndarray, np.ndarray]], train: bool, seed: int = 0):
        self.items, self.train = prepared, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.items[i]
        if self.train:
            img, mask = augment(img, mask, self.rng)
        x = torch.from_numpy(normalise(img)[0])
        y = torch.from_numpy((mask > 0).astype(np.float32)[None])
        return x, y


def split(samples: list[LabelledImage], val_fraction: float, seed: int = 0):
    """Deterministic shuffle-split (same split every run, so validation scores are comparable)."""
    idx = np.random.default_rng(seed).permutation(len(samples))
    n_val = max(1, int(round(len(samples) * val_fraction)))
    val = [samples[i] for i in idx[:n_val]]
    train = [samples[i] for i in idx[n_val:]]
    return train, val
