"""Score the classical segmenter and any number of ONNX models on one labelled test set, same code for all.

    python scripts/compare_segmenters.py --test <MaskFolder root> --models a.onnx b.onnx [--shapes shapes.json]
                                         [--out results.json]

With --shapes (the provenance file of drafted outlines), scores are also reported on the photos whose outline is
marked "high" confidence, so label noise on faint / ambiguous photos cannot hide or fake a difference.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surgiguard_d1.wound import ClassicalSegmenter  # noqa: E402
from surgiguard_d1.wound.learned import LearnedSegmenter  # noqa: E402
from surgiguard_d1.wound.seg_data import MaskFolder  # noqa: E402
from surgiguard_d1.wound.seg_eval import BenchmarkReport, SegmenterBenchmark  # noqa: E402

COLUMNS = ("dice_mean", "dice_median", "iou_mean", "precision_mean", "recall_mean", "width_ratio_median", "misses")


def subset(report: BenchmarkReport, names: set[str]) -> dict:
    sub = BenchmarkReport(report.segmenter, {k: v for k, v in report.per_image.items() if k in names},
                          tuple(m for m in report.misses if m in names))
    return sub.summary()


def table(title: str, rows: list[dict]) -> str:
    head = f"\n{title}\n| segmenter | n | " + " | ".join(COLUMNS) + " |\n|---|---|" + "---|" * len(COLUMNS)
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    return head + "".join(f"\n| {r['segmenter']} | {r['n_images']} | " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |"
                          for r in rows)


def main(a) -> None:
    samples = list(MaskFolder(Path(a.test)))
    bench = SegmenterBenchmark(samples)
    segmenters = [ClassicalSegmenter()] + [LearnedSegmenter(m) for m in a.models]
    reports = [bench.run(s) for s in segmenters]
    out = {"test": str(a.test), "all": [r.summary() for r in reports]}
    print(table(f"All {len(samples)} photos", out["all"]))
    if a.shapes:
        conf = json.loads(Path(a.shapes).read_text(encoding="utf-8"))
        high = {k for k, v in conf.items() if not k.startswith("_") and v.get("conf") == "high"}
        out["high_confidence"] = [subset(r, high) for r in reports]
        print(table(f"High-confidence outlines only ({len(high & {s.name for s in samples})} photos)",
                    out["high_confidence"]))
    out["per_image"] = {r.segmenter: {k: v.dice for k, v in r.per_image.items()} for r in reports}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--shapes")
    ap.add_argument("--out")
    main(ap.parse_args())
