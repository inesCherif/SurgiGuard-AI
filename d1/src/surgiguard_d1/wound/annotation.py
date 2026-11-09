"""Click-to-outline annotation: the logic behind scripts/outline_wounds.py, without any window code.

Why two shape tools (Strategy pattern, `Shape`):
* `Polygon`  - click around the border; for open wounds, ulcers, dehiscence.
* `Stroke`   - click ALONG a closed incision and set its width; a 3 mm cut is 10-20 px wide, so tracing both
  borders by hand is slow and noisy, while a centre line + width is fast and repeatable.

An `Annotation` is a list of finished shapes plus one shape being drawn (Composite: the mask is the union of
all shapes). Every edit is undoable. The GUI only translates mouse/keys into these calls, so all of it is
unit-tested without a screen (Humble Object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

Point = tuple[int, int]


class Shape(Protocol):
    points: list[Point]

    def is_complete(self) -> bool: ...

    def draw(self, mask: np.ndarray) -> None:
        """Paint this shape (value 1) into a uint8 mask, in place."""


@dataclass
class Polygon:
    points: list[Point] = field(default_factory=list)

    def is_complete(self) -> bool:
        return len(self.points) >= 3

    def draw(self, mask: np.ndarray) -> None:
        if self.is_complete():
            cv2.fillPoly(mask, [np.array(self.points, np.int32)], 1)


@dataclass
class Stroke:
    width: int = 12
    points: list[Point] = field(default_factory=list)

    def is_complete(self) -> bool:
        return len(self.points) >= 2

    def draw(self, mask: np.ndarray) -> None:
        if self.is_complete():
            cv2.polylines(mask, [np.array(self.points, np.int32)], False, 1, thickness=max(1, int(self.width)),
                          lineType=cv2.LINE_8)
            for p in (self.points[0], self.points[-1]):      # round caps so the ends are not cut square
                cv2.circle(mask, p, max(1, int(self.width)) // 2, 1, -1)


SHAPES = {"polygon": Polygon, "stroke": Stroke}


@dataclass
class Annotation:
    """Shapes on one image. `tool` decides what the next click starts."""
    height: int
    width: int
    tool: str = "stroke"
    stroke_width: int = 12
    done: list = field(default_factory=list)
    current: object | None = None

    def _new_shape(self):
        return Stroke(self.stroke_width) if self.tool == "stroke" else Polygon()

    def click(self, x: int, y: int) -> None:
        x, y = int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))
        if self.current is None:
            self.current = self._new_shape()
        self.current.points.append((x, y))

    def finish_shape(self) -> bool:
        """Close the shape being drawn. Returns False (and drops it) if it has too few points."""
        cur, self.current = self.current, None
        if cur is not None and cur.is_complete():
            self.done.append(cur)
            return True
        return False

    def undo(self) -> None:
        """Remove the last point; if no shape is being drawn, reopen the last finished one."""
        if self.current is None and self.done:
            self.current = self.done.pop()
        if self.current is not None:
            self.current.points.pop()
            if not self.current.points:
                self.current = None

    def set_tool(self, tool: str) -> None:
        if tool not in SHAPES:
            raise ValueError(f"unknown tool {tool!r}, expected one of {sorted(SHAPES)}")
        self.finish_shape()
        self.tool = tool

    def change_width(self, delta: int) -> None:
        self.stroke_width = int(np.clip(self.stroke_width + delta, 1, 200))
        if isinstance(self.current, Stroke):
            self.current.width = self.stroke_width

    def mask(self, include_current: bool = True) -> np.ndarray:
        m = np.zeros((self.height, self.width), np.uint8)
        for s in self.done + ([self.current] if include_current and self.current is not None else []):
            s.draw(m)
        return m.astype(bool)

    def is_empty(self) -> bool:
        return not self.done and self.current is None


def preview(bgr: np.ndarray, ann: Annotation, alpha: float = 0.45) -> np.ndarray:
    """Photo with the current mask tinted green and the clicked points shown (used by the GUI)."""
    out = bgr.copy()
    m = ann.mask()
    out[m] = (out[m] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
    if ann.current is not None:
        pts = ann.current.points
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 255), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return out
