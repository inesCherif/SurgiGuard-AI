"""Turn outlines written as shapes (JSON) into masks, with the same Annotation code as the click tool.

    python scripts/masks_from_shapes.py <shapes.json> <MaskFolder root> [--snap 3]

Used for outlines drafted without the GUI (e.g. read off a coordinate grid). JSON format, one entry per photo:

    {"1000": {"conf": "high", "shapes": [["s", 1.2, [[1.9, 0.9], [3.5, 4.3], [5.3, 9.0]]],
                                          ["p", [[4.0, 2.9], [5.6, 3.6], [4.3, 4.0]]]]},
     "1007": {"conf": "low", "shapes": []}}

* coordinates are on a 0-10 grid: x = 10 * column / width, y = 10 * row / height (independent of photo size);
* "s" = Stroke (centre line of an incision), its width in % of the photo's long side;
* "p" = Polygon (border of an open wound); an empty list = no wound visible (empty mask);
* "conf" = how sure the annotator was ("high" / "low"); benchmarks can report both subsets.

--snap P moves each stroke point sideways onto the cut line, at most P % of the long side (annotation.snap_stroke):
the annotator decides where the incision is, snapping only fixes click precision. Polygons are never changed.

The JSON is the provenance record: it says exactly what was drawn, by whom (see "_meta"), and can be redrawn.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surgiguard_d1.wound.annotation import Annotation, ridge_map, snap_stroke  # noqa: E402
from surgiguard_d1.wound.seg_data import MaskFolder, read_bgr  # noqa: E402


def to_mask(entry: dict, height: int, width: int, bgr=None, snap_pct: float = 0.0):
    ann = Annotation(height, width)
    ridge = ridge_map(bgr) if (bgr is not None and snap_pct > 0) else None
    for shape in entry["shapes"]:
        kind = shape[0]
        if kind == "s":
            ann.set_tool("stroke")
            ann.stroke_width = max(1, round(shape[1] / 100 * max(height, width)))
            pts = shape[2]
        elif kind == "p":
            ann.set_tool("polygon")
            pts = shape[1]
        else:
            raise ValueError(f"unknown shape kind {kind!r}")
        px = [(round(x / 10 * (width - 1)), round(y / 10 * (height - 1))) for x, y in pts]
        if kind == "s" and ridge is not None:
            px = snap_stroke(bgr, px, radius=round(snap_pct / 100 * max(height, width)), ridge=ridge)
        for x, y in px:
            ann.click(x, y)
        if not ann.finish_shape():
            raise ValueError(f"shape with too few points: {shape}")
    return ann.mask()


def main(shapes_json: Path, root: Path, snap_pct: float = 0.0) -> None:
    shapes = json.loads(shapes_json.read_text(encoding="utf-8"))
    store = MaskFolder(root)
    by_stem = {p.stem: p for p in store.image_paths()}
    done = 0
    for name, entry in shapes.items():
        if name.startswith("_"):
            continue
        path = by_stem[name]
        bgr = read_bgr(path)
        h, w = bgr.shape[:2]
        store.write_mask(path, to_mask(entry, h, w, bgr, snap_pct))
        done += 1
    missing = sorted(set(by_stem) - set(shapes))
    print(f"{done} masks written to {store.labels_dir}; photos without shapes: {len(missing)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shapes", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--snap", type=float, default=0.0, help="max sideways stroke correction, %% of the long side")
    a = ap.parse_args()
    main(a.shapes, a.root, a.snap)
