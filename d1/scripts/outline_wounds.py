"""Click-to-outline tool: draw wound masks on photos, one after the other, resumable.

    python scripts/outline_wounds.py <folder>          # <folder>/images/*.jpg -> <folder>/labels/*.png

Needs OpenCV WITH windows (`opencv-python`, not `-headless`): run it with the training venv.

Mouse   left click = add point          right click = undo last point
Keys    l = stroke tool (incision: click along the cut)      p = polygon tool (click around the border)
        + / - (or mouse wheel) = stroke width                 ENTER / SPACE = finish the current shape
        s = save mask and next     e = save EMPTY mask (no wound visible)     n = skip     b = back
        r = reset this image       q / ESC = quit (unsaved shapes on this image are lost)

All drawing logic lives in surgiguard_d1.wound.annotation (unit-tested); this file only maps events to it.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surgiguard_d1.wound.annotation import Annotation, preview  # noqa: E402
from surgiguard_d1.wound.seg_data import MaskFolder, read_bgr  # noqa: E402

WINDOW = "SurgiGuard outline (h = help in terminal)"
MAX_SIDE = 900          # display size; clicks are mapped back to full resolution


def _banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate(folder: Path) -> None:
    store = MaskFolder(folder)
    paths = store.image_paths()
    if not paths:
        sys.exit(f"no images in {store.images_dir}")
    todo = [i for i, p in enumerate(paths) if not store.is_labelled(p)]
    idx = todo[0] if todo else 0
    print(__doc__)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    state = {}

    def load(i):
        bgr = read_bgr(paths[i])
        scale = min(1.0, MAX_SIDE / max(bgr.shape[:2]))
        state.update(bgr=bgr, scale=scale, ann=Annotation(*bgr.shape[:2], stroke_width=max(4, round(12 / scale))))

    def on_mouse(event, x, y, flags, _):
        ann, s = state["ann"], state["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            ann.click(round(x / s), round(y / s))
        elif event == cv2.EVENT_RBUTTONDOWN:
            ann.undo()
        elif event == cv2.EVENT_MOUSEWHEEL:
            ann.change_width(2 if flags > 0 else -2)

    cv2.setMouseCallback(WINDOW, on_mouse)
    load(idx)
    while True:
        ann, s = state["ann"], state["scale"]
        view = preview(state["bgr"], ann)
        if s < 1:
            view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        done = sum(store.is_labelled(p) for p in paths)
        tag = "LABELLED" if store.is_labelled(paths[idx]) else "new"
        _banner(view, f"{idx + 1}/{len(paths)} [{tag}] done {done} | tool {ann.tool} w={ann.stroke_width}px | "
                      f"s save  e empty  n skip  b back  q quit")
        cv2.imshow(WINDOW, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        key = chr(k).lower() if k < 128 else ""
        if k in (13, 32):
            ann.finish_shape()
        elif key in ("l", "p"):
            ann.set_tool("stroke" if key == "l" else "polygon")
        elif key in ("+", "="):
            ann.change_width(2)
        elif key == "-":
            ann.change_width(-2)
        elif key == "r":
            load(idx)
        elif key in ("s", "e"):
            ann.finish_shape()
            if key == "s" and ann.is_empty():
                print("nothing drawn: use 'e' to save an empty mask on purpose")
                continue
            mask = ann.mask() if key == "s" else np.zeros(state["bgr"].shape[:2], bool)
            print("saved", store.write_mask(paths[idx], mask))
            idx = min(idx + 1, len(paths) - 1)
            load(idx)
        elif key == "n":
            idx = min(idx + 1, len(paths) - 1)
            load(idx)
        elif key == "b":
            idx = max(idx - 1, 0)
            load(idx)
        elif key == "h":
            print(__doc__)
        elif key == "q" or k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    annotate(Path(sys.argv[1]))
