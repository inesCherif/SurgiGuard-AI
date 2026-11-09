# 05. Wound v2 step 3: learned segmentation

> Status: **part 1 of 3 done** (the yardstick). Part 2 (data + annotation tool) and part 3 (the learned
> `Segmenter`) are next. Previous: [04_CALIBRATION.md](04_CALIBRATION.md). Journal: [02_PROBLEMS_AND_SOLUTIONS.md](02_PROBLEMS_AND_SOLUTIONS.md) (V15).

## Why this step

Calibration (step 2) made the segmentation error measurable in millimetres, and it showed the real weakness
(journal V14): the classical segmenter grows its mask by ~1.4 mm on each side, so a 3 mm incision is reported
about 6 mm wide. A learned segmenter can draw a tight outline, but "the model is better" is only a claim until
both are scored by the **same code on the same images**. So the first deliverable is the yardstick, not the model.

## Plan

| Part | What | Status |
|---|---|---|
| 1 | `wound/seg_eval.py`: Dice, IoU, precision, recall, **width ratio**; `SegmenterBenchmark` for any `Segmenter` | ✅ done |
| 2 | Data: public wound-segmentation masks + ~100–150 incision outlines drawn with a click-to-outline tool (SurgWound has no masks) | ⏭ next |
| 3 | `LearnedSegmenter` (small U-Net, GPU torch), injected into `WoundAnalyzer`; benchmark vs `ClassicalSegmenter` | later |

## Part 1: the evaluation module

`d1/src/surgiguard_d1/wound/seg_eval.py`

- **`score_masks(pred, truth) -> MaskScores`**: Dice `2|P∩T| / (|P|+|T|)`, IoU `|P∩T| / |P∪T|`, precision, recall.
  Conventions used in medical segmentation: empty prediction on empty truth = 1 (correctly found nothing);
  nothing found on a real wound = 0. Masks of different shapes raise an error instead of returning a silent score.
- **`width_ratio`**: predicted mean width / true mean width, with mean width = area / length of the
  minimum-area rotated rectangle. Why a separate metric: for a long thin incision drawn twice too wide,
  recall is 1, precision is 0.5 and **Dice is still 0.67**, which looks acceptable. The width ratio says
  **2.0** directly, which is the clinically relevant error (V14).
- **`SegmenterBenchmark`** takes a list of `LabelledImage` (photo, true mask, optional scale and ignore mask),
  runs any segmenter and returns a `BenchmarkReport` (per-image scores, misses, `summary()`); `compare(a, b, ...)`
  gives side-by-side summaries.

### Design

- **Dependency Inversion / Strategy**: the benchmark depends only on the `Segmenter` protocol from
  `segmentation.py`. Classical, learned or a test fake all go through the same loop, so a comparison can never
  be biased by different evaluation code.
- **Template Method** (in spirit): the loop "segment → score → collect misses" is fixed; the segmenter is the
  pluggable step.
- **Value objects**: `MaskScores`, `LabelledImage` and `BenchmarkReport` are frozen dataclasses.
- **Single Responsibility**: metrics know nothing about segmenters; the benchmark knows nothing about how a
  segmenter works.

### First baseline (synthetic incision, 300 × 15 px, scale 5 px/mm)

| Segmenter | Dice | IoU | Precision | Recall | Width ratio |
|---|---|---|---|---|---|
| classical-local-contrast | 0.74 | 0.59 | 0.59 | 1.00 | **1.65** |

Recall 1.0 with precision 0.59 is exactly the V14 behaviour (never misses the edge, but over-draws the margin).
This is the number part 3 has to beat; real photos will be scored the same way once part 2 provides outlines.

### Tests

`d1/tests/test_wound_seg_eval.py` (8 tests): exact Dice/IoU on hand-built masks (perfect, half overlap),
the doubled-width case (Dice 0.67 but width ratio 2.0), empty-mask conventions, shape mismatch error, width of a
rotated bar, benchmark with fake perfect/blind segmenters, and the classical segmenter end to end.

## Honest limits

- The width estimate assumes an elongated, roughly straight wound; for curved or round wounds it is only a rough
  indication (Dice/IoU remain valid).
- The baseline above is one synthetic image. It proves the tool works; it is **not** a claim about real photos.
