"""Labelled wound images on disk: one folder layout shared by public data, our own outlines and training.

Layout (the FUSeg challenge layout, so the public set needs no conversion):

    <root>/images/<name>.<jpg|png>      photo
    <root>/labels/<name>.png            mask, 0 = background, >0 = wound

`MaskFolder` is the single place that knows this layout (DRY): the benchmark reads it, the click-to-outline
tool writes it, the training script reads it. It yields `LabelledImage` value objects from `seg_eval`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .seg_eval import LabelledImage

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def read_bgr(path: Path) -> np.ndarray:
    # imdecode instead of imread: imread fails on non-ASCII Windows paths (OneDrive folders with accents)
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"could not encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


@dataclass(frozen=True)
class MaskFolder:
    root: Path

    @property
    def images_dir(self) -> Path:
        return Path(self.root) / "images"

    @property
    def labels_dir(self) -> Path:
        return Path(self.root) / "labels"

    def image_paths(self) -> list[Path]:
        if not self.images_dir.is_dir():
            return []
        return sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)

    def label_path(self, image_path: Path) -> Path:
        return self.labels_dir / (Path(image_path).stem + ".png")

    def is_labelled(self, image_path: Path) -> bool:
        return self.label_path(image_path).is_file()

    def unlabelled(self) -> list[Path]:
        return [p for p in self.image_paths() if not self.is_labelled(p)]

    def read_mask(self, image_path: Path) -> np.ndarray:
        m = cv2.imdecode(np.fromfile(str(self.label_path(image_path)), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"unreadable mask for {image_path.name}")
        return m > 0

    def write_mask(self, image_path: Path, mask: np.ndarray) -> Path:
        out = self.label_path(image_path)
        write_png(out, np.asarray(mask, bool).astype(np.uint8) * 255)
        return out

    def __iter__(self) -> Iterator[LabelledImage]:
        """Labelled pairs only; a mask whose size differs from its photo is an error, not silently resized."""
        for p in self.image_paths():
            if not self.is_labelled(p):
                continue
            bgr, truth = read_bgr(p), self.read_mask(p)
            if truth.shape != bgr.shape[:2]:
                raise ValueError(f"{p.name}: mask {truth.shape} vs image {bgr.shape[:2]}")
            yield LabelledImage(p.stem, bgr, truth)

    def __len__(self) -> int:
        return sum(1 for p in self.image_paths() if self.is_labelled(p))

