"""Pick SurgWound photos to outline by hand, balanced by closure method, without train/test leakage.

    python scripts/make_outline_set.py <prepared surgwound folder> <out folder> [--train 120] [--test 40]

Input: the output of prepare_surgwound.py (labels.csv + images/<split>/...).
Output: <out>/train/images and <out>/test/images, ready for outline_wounds.py (MaskFolder layout).
Train photos come only from SurgWound's train split and test photos only from its test split, so a photo of the
same wound can never be on both sides. Within each, photos are drawn round-robin over closure methods (sutures,
staples, glue, ...) so a rare closure type is not left out.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def balanced_pick(rows: list[dict], n: int, key: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "unknown"].append(r)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    picked = []
    while len(picked) < n and any(groups.values()):
        for g in order:
            if groups[g] and len(picked) < n:
                picked.append(groups[g].pop())
    return picked


def main(src: Path, out: Path, n_train: int, n_test: int, key: str, seed: int) -> None:
    with open(src / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for split, n, from_split in (("train", n_train, "train"), ("test", n_test, "test")):
        chosen = balanced_pick([r for r in rows if r["split"] == from_split], n, key, seed)
        dst = out / split / "images"
        dst.mkdir(parents=True, exist_ok=True)
        counts = defaultdict(int)
        for r in chosen:
            shutil.copy2(src / r["image"], dst / Path(r["image"]).name)
            counts[r.get(key) or "unknown"] += 1
        print(split, len(chosen), dict(sorted(counts.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--by", default="closure")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.src, a.out, a.train, a.test, a.by, a.seed)
