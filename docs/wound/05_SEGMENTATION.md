# 05. Wound v2 step 3: learned segmentation

> Status: **parts 1–4 done, incision results PRELIMINARY.** At human review (10 Nov 2026) the AI-drafted incision
> outlines used below were judged wrong (the lines did not follow the wounds), so the incision numbers in this page are
> not reliable and v2 was partly trained on wrong examples (journal V25). Final step: outlines drawn by a human, then
> one command re-trains and re-grades everything: **[07_HUMAN_OUTLINES.md](07_HUMAN_OUTLINES.md)**. The FUSeg (ulcer)
> results are not affected (expert outlines from the dataset).
> Previous: [04_CALIBRATION.md](04_CALIBRATION.md). Journal: [02_PROBLEMS_AND_SOLUTIONS.md](02_PROBLEMS_AND_SOLUTIONS.md) (V15–V22).

## Results in two tables

**Surgical incisions — preliminary, graded against AI-drafted outlines later rejected at human review (V25)**
(40 SurgWound test photos, outlines drafted by an AI assistant; "high" = the 19 photos
where the cut is clearly visible). `scripts/compare_segmenters.py`, results in `d1/models/wound_seg/compare_incisions.json`:

| Segmenter | Dice (high, 19) | Dice (all, 40) | Precision (high) | Recall (high) | Width ratio (all) | Nothing found (all) |
|---|---|---|---|---|---|---|
| classical-local-contrast | 0.18 | 0.14 | 0.19 | 0.24 | **2.21** | 0 |
| learned-unet:fuseg-v1 (ulcers only) | 0.26 | 0.26 | 0.49 | 0.19 | 0.82 | 13 |
| **learned-unet:fuseg+surgwound-v2** | **0.46** | **0.31** | **0.56** | **0.43** | 0.85 | 2 |

Paired, photo by photo (high-confidence): v2 beats classical on **16 / 19** (loses 3), mean Dice gain **+0.28,
95 % bootstrap CI [0.17, 0.39]**; v2 beats v1 on 16 / 19. On all 40: 24 wins, 8 losses, gain +0.18 [0.10, 0.25].

**Foot ulcers** (200 FUSeg test photos), `compare_fuseg.json`: classical 0.17, v1 0.82, **v2 0.83** Dice
(fine-tuning on incisions did not make the model forget ulcers).

The first FUSeg-only result, kept for the record (200 FUSeg photos, same `SegmenterBenchmark` code):

| Segmenter | Dice mean | Dice median | IoU | Precision | Recall | Width ratio | Nothing found | CPU time / photo |
|---|---|---|---|---|---|---|---|---|
| classical-local-contrast | 0.17 | 0.01 | 0.11 | 0.15 | 0.31 | 1.11 | 0 / 200 | 0.12 s |
| **learned-unet:fuseg-v1** | **0.82** | **0.91** | **0.74** | **0.82** | **0.86** | **1.02** | 2 / 200 | 0.18 s |

Read it honestly:
- These are **foot ulcers** (FUSeg), not surgical incisions. The classical segmenter was designed for incisions
  on skin, so 0.17 is partly an unfair test; the learned model has not yet been tested on incisions at all (V20).
  This is why the classical segmenter stays the **default** until part 4.
- The size floor of the learned segmenter was corrected after looking at test failures (V19). Before the fix the
  same model scored Dice 0.74 mean with 26 misses; the raw network (no post-processing) scores 0.82. Both numbers
  are kept in `C:\Users\juba839\surgiguard_data\unet_benchmark_before_V19.json` and in this doc.
- Width ratio 1.02: on these photos the learned outline has the right width, which is exactly the classical
  weakness found in V14 (×1.65 on incisions). It must be confirmed on incisions.

## Plan

| Part | What | Status |
|---|---|---|
| 1 | `seg_eval.py`: Dice, IoU, precision, recall, width ratio; `SegmenterBenchmark` for any `Segmenter` | ✅ |
| 2 | Data layer (`seg_data.py`), click-to-outline tool, outline set of 160 SurgWound photos | ✅ |
| 3 | `LearnedSegmenter` (ONNX), training code, model trained on FUSeg, benchmark | ✅ |
| 4 | 160 incision outlines, fine-tune (v2), three-way benchmark on incisions and ulcers | ✅ default still classical (labels not human-verified) |

## Part 1: the yardstick

`d1/src/surgiguard_d1/wound/seg_eval.py`

- **`score_masks(pred, truth)`**: Dice `2|P∩T| / (|P|+|T|)`, IoU `|P∩T| / |P∪T|`, precision, recall. Empty on empty = 1,
  nothing found on a real wound = 0, masks of different shapes raise an error.
- **`width_ratio`**: predicted / true mean width (area / long side of the rotated bounding rectangle). For a long incision
  drawn twice too wide, Dice is still 0.67 but the width ratio says 2.0, the clinically relevant error (V15).
- **`SegmenterBenchmark`** runs any segmenter over `LabelledImage`s, returns per-image scores, misses and a summary.

Design: depends only on the `Segmenter` protocol (**Dependency Inversion / Strategy**), fixed loop with a pluggable step
(**Template Method** in spirit), frozen value objects.

## Part 2: data and the outline tool

**Data**, stored outside OneDrive in `C:\Users\juba839\surgiguard_data` (V16):

| Set | Content | Masks | Use |
|---|---|---|---|
| FUSeg 2021 (MICCAI challenge, UW-Milwaukee) | 1 010 foot-ulcer photos, 512 × 512, de-identified | yes | train (810, split 729 / 81) + test (official validation, 200) |
| SurgWound (Hugging Face `Sonulee/SurgWound`, CC BY-SA 4.0) | 686 surgical-wound photos with surgeon labels | **no** | 160 photos to outline (`outline/train` 120, `outline/test` 40) |

FUSeg citation: C. Wang et al., "Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks",
Scientific Reports 10:21897, 2020.

**`seg_data.MaskFolder`** is the only code that knows the layout `<root>/images/*` + `<root>/labels/*.png` (the FUSeg
layout, so no conversion). The benchmark and training read it, the outline tool writes it (**DRY**). It refuses a mask
whose size differs from its photo instead of resizing silently.

**`annotation.py` + `scripts/outline_wounds.py`**: two **Strategy** shapes, `Polygon` (click around an open wound) and
`Stroke` (click *along* an incision + set its width: a 3 mm cut is 10–20 px wide, tracing both borders is slow and
noisy). The mask is the union of shapes (**Composite**), every edit undoable. All logic is window-free and unit-tested;
the script only maps mouse/keys to it (**Humble Object**).

**`scripts/make_outline_set.py`**: draws photos round-robin over closure methods so rare types are included (all 14
adhesive and all 7 staple photos in train), train only from SurgWound's train split, test only from its test split.

How to outline (about 1 minute per photo, resumable):

```powershell
C:\Users\juba839\surgiguard_data\venv-train\Scripts\python d1\scripts\outline_wounds.py C:\Users\juba839\surgiguard_data\outline\train
C:\Users\juba839\surgiguard_data\venv-train\Scripts\python d1\scripts\outline_wounds.py C:\Users\juba839\surgiguard_data\outline\test
```
`l` stroke / `p` polygon, left click = point, right click = undo, mouse wheel = width, Enter = finish shape,
`s` = save + next, `e` = save "no wound visible", `n` skip, `b` back, `q` quit.

## Part 3: the learned segmenter

### Inference: `wound/learned.py`

- **ONNX + onnxruntime**: training needs GPU PyTorch (~2.5 GB), inference must stay light and portable like the
  numpy model bundle of step 1. onnxruntime is already in the backend, CPU-only, thread-safe; the model is one file
  (`d1/models/wound_seg/unet.onnx`, 58 MB) plus its model card (`unet.json`: input size, threshold, data, scores,
  training history).
- **`LearnedSegmenter`** implements the `Segmenter` protocol, so
  `WoundAnalyzer(segmenter=LearnedSegmenter("d1/models/wound_seg/unet.onnx"))` swaps it in without touching
  quality, measurement, registration or the backend (**Open/Closed**). The onnxruntime session is created lazily
  and can be injected (tests use a fake session).
- **Letterbox, not resize**: the photo's long side is fitted to 320 px and the rest padded, so a thin incision is not
  squeezed (a plain resize of a 4:3 photo to a square changes widths by 33 %). `letterbox`, `unletterbox` and
  `normalise` are imported by the training code too: preprocessing **cannot** differ between training and inference.
- **Shared finishing step**: the peri-wound skin reference and size checks moved from `ClassicalSegmenter._finish` to
  `finish_segmentation()`, used by both segmenters, so measurement always receives the same fields computed the same
  way. Proven safe: 80/80 FUSeg photos give a bit-identical classical mask and reference colour before/after.
- **Per-segmenter limits** (V19): `LEARNED_LIMITS` lowers the minimum wound size from 0.2 % to 0.02 % of the frame;
  isolated blobs below half of that are dropped as specks.

### Training: `wound/training/` + `scripts/train_segmenter.py`

| Choice | Why |
|---|---|
| U-Net, ResNet18 encoder pretrained on ImageNet (~14 M parameters) | skip connections bring back the fine detail thin wounds need; pretrained features matter with only ~800 photos; fits a 4 GB GPU |
| Loss = BCE + soft Dice | a wound can be ~1 % of the pixels; Dice stops the model from predicting "all background" |
| Augmentations: rotation ±25°, scale 0.75–1.3, shift, flips, brightness / contrast / saturation, hue ±4 only | home phone photos vary in framing and light; hue is kept almost fixed because colour is a clinical signal |
| AdamW + OneCycle LR, mixed precision, batch 8 at 320 px | ~25 s per epoch on the RTX 2050, 1.7 GB of GPU memory |
| Best epoch chosen on the internal validation split (81 photos), test set used once at the end | no test-set selection |
| Full checkpoint every epoch + `--resume` | the PC ran out of memory mid-training (V17) |

Training curve (validation Dice at network resolution): epoch 1: 0.39, 10: 0.72, 20: 0.81, best 0.86 at epoch 38 of 40.

Re-train or re-score:
```powershell
$F = "C:\Users\juba839\surgiguard_data\raw\wound-segmentation\data\Foot Ulcer Segmentation Challenge"
$env:OPENBLAS_NUM_THREADS = 1
C:\Users\juba839\surgiguard_data\venv-train\Scripts\python d1\scripts\train_segmenter.py --train "$F\train" --test "$F\validation" --out d1\models\wound_seg\unet --version fuseg-v1 --resume
# add --benchmark-only to re-score the existing model
```

The training environment (`C:\Users\juba839\surgiguard_data\venv-train`, Python 3.11) is separate from the backend
venv: torch 2.5.1+cu121, torchvision 0.20.1, onnx, onnxruntime, opencv-python (with windows, for the outline tool),
numpy, pillow, pydantic, scikit-learn, joblib, pytest. The backend needs nothing new.

### Tests

| File | Tests | What they prove |
|---|---|---|
| `test_wound_seg_eval.py` | 8 | exact Dice / IoU on hand-built masks, doubled width exposed, empty conventions, benchmark loop |
| `test_wound_annotation.py` | 8 | stroke width, polygon fill, union, undo, clamping, folder round trip, size mismatch refused |
| `test_wound_learned.py` | 8 | letterbox round trip, normalisation layout, post-processing, model card, injection into `WoundAnalyzer`, small wound kept (V19), **a real ONNX file through onnxruntime gives the same mask as the reference** |
| `test_wound_training.py` | 5 | U-Net output shape, loss sanity, augmentation keeps photo and mask aligned, deterministic split, one-image overfit |

Backend venv: d1 229 passed, 3 skipped (the torch-training tests and the ONNX export, which need the training venv).
Training venv: all 30 wound-segmentation tests pass (incl. stroke snapping).

## Part 4: incision outlines and fine-tuning

### Who drew the outlines, and how

The 160 SurgWound photos (120 train, 40 test) were outlined by **Claude, an AI assistant**, at Ines's request, not by
a clinician. Method, so it can be checked and repeated:

1. Each photo shown on a contact sheet with a 0–10 grid; the cut written as a **stroke** (centre line + width in %
   of the long side) or an open wound as a **polygon**, plus a confidence flag: `high` = clearly visible,
   `low` = faint, partly hidden or ambiguous (83 / 37 in train, 19 / 21 in test; 10 photos "no wound visible").
2. Definition used everywhere: the incision line itself (edges + scab) or the open wound bed; surrounding redness,
   suture threads, dressings and pen/marker drawings excluded.
3. `scripts/masks_from_shapes.py` turns the shapes into masks with the **same `Annotation` code** as the click tool.
   `--snap 3` moves each stroke point sideways (≤ 3 % of the long side) onto the darkest/reddest local ridge
   (`annotation.snap_stroke`): reading coordinates off a grid is only precise to 2–5 % of the photo, which is the
   whole width of a thin incision (V22). Polygons are never changed.
4. Every mask drawn back on its photo and checked visually; 2 strokes re-placed by hand (130, 108).
5. The shapes are the provenance record: `d1/data/wound_outlines/shapes_train.json`, `shapes_test.json`
   (with a `_meta` block saying who, how and when).

### Fine-tuning

`train_segmenter.py --train <FUSeg train> <outline train> --weights 1 4 --init models/wound_seg/unet.pt --lr 3e-4
--epochs 30`. Each root is split 90/10 **before** oversampling (×4 for the 120 incisions), so no photo can be in
both train and validation. Best epoch 29, validation Dice 0.77 (mixed ulcers + incisions). Model:
`d1/models/wound_seg/unet_v2.onnx` + `unet_v2.json`.

Re-run the comparison:
```powershell
$O = "C:\Users\juba839\surgiguard_data\outline"
backend\.venv\Scripts\python d1\scripts\compare_segmenters.py --test "$O\test" --models d1\models\wound_seg\unet.onnx d1\models\wound_seg\unet_v2.onnx --shapes d1\data\wound_outlines\shapes_test.json
```

### Why the default is still the classical segmenter

The evidence favours v2 on every metric and both datasets, but the incision labels (train **and** test) come from one
non-clinical annotator. Switching the product default should wait until a human, ideally a clinician, has checked
at least the 40 test outlines (`outline_wounds.py` can open and redraw them). To use v2 now:
`WoundAnalyzer(segmenter=LearnedSegmenter("d1/models/wound_seg/unet_v2.onnx"))`.

## Honest limits

- **Incision labels are AI-drafted**, not clinician-verified; the same annotator drew train and test, so a systematic
  bias in the definition would be shared by both. Snapping uses local darkness/redness, an idea close to the classical
  segmenter, which if anything favours the classical baseline.
- Absolute Dice on incisions (0.46 on clear photos) is much lower than on ulcers: a 1 %-wide structure loses most of
  its overlap with a few pixels of misalignment, in the label or the prediction. Width ratio and paired wins are the
  more robust readings.
- 40 test photos (19 high-confidence): the bootstrap interval is wide, though clearly above zero.
- The V19 size-floor correction was found by looking at test failures; reported with before/after numbers.
- The width ratio assumes an elongated, roughly straight wound; for round ulcers it is only indicative.
- One training run per model, one seed.
