# 06. Wound v2 step 4: classifier on the wound crop, calibrated probabilities

> Status: **done.** The wound-attribute model (`wound-attr-0.4`) now looks at the whole photo **and** a crop around
> the wound found by the step-3 segmenter, and its probabilities are calibrated.
> Previous: [05_SEGMENTATION.md](05_SEGMENTATION.md). Model card: [`d1/models/WOUND_MODEL_CARD.md`](../../d1/models/WOUND_MODEL_CARD.md).
> Journal: [02_PROBLEMS_AND_SOLUTIONS.md](02_PROBLEMS_AND_SOLUTIONS.md) (V23–V24).

## The two problems

1. **The classifier looked at the whole photo.** Sheets, clothes, tattoos and background fill most of the image;
   redness or discharge are a few percent of the pixels. A frozen encoder (CLIP) summarises the whole scene.
2. **Its probabilities did not mean what they said.** The heads are logistic regressions trained with
   `class_weight="balanced"` (otherwise rare positives are ignored). That re-weighting pushes every probability
   towards 0.5: on the test photos, the model said "discharge 30 %" for photos where discharge was present far less
   often. A clinician reading a percentage would be misled, even though the ranking (AUROC) was fine.

## Part 1: whole photo vs wound crop (pre-registered)

`d1/src/surgiguard_d1/wound/crop.py` — `WoundCropper(segmenter)`: square box around the wound mask, enlarged by
0.35 × its long side (peri-wound skin is where redness and swelling are judged), at least 25 % of the photo's short
side, shifted inside the photo; falls back to the whole photo when no wound is found (4.2 % of SurgWound photos),
and says so. Depends only on the `Segmenter` protocol (**Strategy / Dependency Inversion**), here the learned v2 segmenter.

`d1/scripts/wound_view_experiment.py` — protocol fixed before looking at any test number:

1. Three feature views: **whole** (as before), **crop**, **whole+crop** (both vectors concatenated).
2. Same CV and regularisation grid as the production training, on train+val only.
3. View used from now on = best **mean CV AUROC** over the six attributes; then every view is scored on test (nothing
   hidden), with a paired bootstrap of the difference.

No leakage: the segmenter was trained on SurgWound **train** photos only; the classifier's test split was never seen
by either model.

| Mean CV AUROC (6 attributes) | whole 0.818 | crop 0.830 | **whole+crop 0.836 → chosen** |
|---|---|---|---|

Test AUROC (137 photos, untouched), whole → whole+crop, paired 95 % CI of the difference:

| Attribute | whole | whole+crop | difference CI |
|---|---|---|---|
| Swelling (edema) | 0.767 | **0.814** | **[+0.009, +0.090]** |
| Discharge (exudate) | 0.818 | **0.873** | **[+0.014, +0.110]** |
| Not yet healed | 0.771 | **0.803** | [0.000, +0.070] |
| Redness (erythema) | 0.759 | 0.766 | [−0.034, +0.047] |
| Infection-risk rating | 0.843 | 0.853 | [−0.034, +0.060] |
| Needs review (urgency) | 0.823 | 0.809 | [−0.062, +0.037] |

Reading: clear gains where the evidence is **on** the wound (swelling, discharge, healing); no change where the
judgement uses the whole scene (urgency). Crop alone was even better on discharge (0.909) but worse on urgency (0.774):
combining both views keeps the best of each, which is why CV chose it.

## Part 2: calibration

`d1/src/surgiguard_d1/wound/prob_calibration.py` (numpy only): temperature scaling `sigmoid(a·z)` and Platt scaling
`sigmoid(a·z + c)`, ECE (10 bins) and Brier score. Fitted on **out-of-fold logits** of the training split, never on
test. Platt won the second-level CV on every attribute (temperature cannot undo the shift caused by class weighting:
it has no intercept). Calibration is monotone, so **AUROC cannot change**.

| Attribute | ECE raw → calibrated (test) | Brier raw → calibrated |
|---|---|---|
| Redness | 0.124 → **0.077** | 0.182 → 0.171 |
| Swelling | 0.191 → **0.048** | 0.158 → 0.123 |
| Discharge | 0.217 → **0.064** | 0.130 → 0.076 |
| Not yet healed | 0.077 → 0.090 (worse) | 0.180 → 0.178 |
| Infection-risk rating | 0.151 → **0.072** | 0.127 → 0.095 |
| Needs review | 0.157 → **0.096** | 0.128 → 0.090 |

Brier improves on all six; ECE on five. "Not yet healed" (40 % positives) was already almost calibrated: Platt cannot
help much there and the small ECE increase is within noise. Sanity check on test: the mean calibrated probability
matches the real positive rate (e.g. swelling 0.15 vs 0.18, discharge 0.16 vs 0.13).

**Decisions do not change because of calibration** (V24): the sensitivity-first threshold and the "uncertain" band are
still chosen on the raw scale, then mapped through the same monotone calibration (`band_low`, `band_high`). A test
loads the shipped bundle and checks that its yes / no / uncertain decisions are identical to the training metrics.

## Final model (`wound-attr-0.4`, `d1/models/wound_heads.json`)

| Attribute | Test n (+) | AUROC before | **AUROC now** [95 % CI] | Sens / Spec | Abstained (before) |
|---|---|---|---|---|---|
| Redness | 131 (40+) | 0.759 | **0.766** [0.67, 0.85] | 0.92 / 0.34 | 12.2 % (17.6 %) |
| Swelling | 115 (21+) | 0.767 | **0.814** [0.71, 0.90] | 0.90 / 0.61 | 11.3 % (20.0 %) |
| Discharge | 125 (16+) | 0.818 | **0.873** [0.78, 0.95] | 0.75 / 0.75 | 10.4 % (13.6 %) |
| Not yet healed | 137 (55+) | 0.771 | **0.803** [0.73, 0.87] | 0.83 / 0.58 | 5.8 % (10.9 %) |
| Infection-risk rating | 137 (20+) | 0.843 | **0.853** [0.76, 0.94] | 0.90 / 0.76 | 8.8 % (11.7 %) |
| Needs review | 137 (16+) | 0.823 | **0.809** [0.69, 0.91] | 0.75 / 0.74 | 8.0 % (7.3 %) |

Unfamiliar-photo gate on the new features: 0.7 % of test photos flagged (was 2.2 %).

## Code changes

| File | Change | Pattern / principle |
|---|---|---|
| `wound/crop.py` | `WoundCropper`, `square_box_around`, `Crop` value object | Strategy (injected segmenter), honest fallback |
| `wound/prob_calibration.py` | Platt / temperature, ECE, Brier, reliability bins | numpy only, pure functions |
| `wound/model_bundle.py` | format v2: `calib_a`, `calib_c`, `band_low`, `band_high`; v1 files still load | backward-compatible versioned format |
| `wound/model.py` | `photo_features(im, bundle)`: the bundle declares its view; crop at inference with the same cropper; occlusion map recomputes both halves | the model file decides, the code adapts (no silent mismatch) |
| `scripts/train_wound_model.py` | `--view`, Platt calibration, `--encoder`; crop + crop features shared with the experiment | DRY |
| `scripts/wound_view_experiment.py` | the pre-registered comparison | |

Inference cost: ~20 s per photo on the laptop CPU with the evidence map, of which the segmenter is ~0.2 s; without the
evidence map (default), two CLIP passes instead of one.

Tests: `test_wound_crop.py` (5), `test_wound_prob_calibration.py` (5), 2 new bundle tests (calibrated round trip,
v1 still loads), shipped-bundle test updated. d1: 241 passed, 3 skipped; backend: 194 passed.

Reproduce:
```powershell
$P = "C:\Users\juba839\surgiguard_data\surgwound"     # output of scripts/prepare_surgwound.py
backend\.venv\Scripts\python d1\scripts\wound_view_experiment.py $P
backend\.venv\Scripts\python d1\scripts\train_wound_model.py $P --encoder openai/clip-vit-base-patch16
```

## Honest limits

- Small test split (137 photos, 16–55 positives per attribute): CIs are wide; only swelling and discharge gains are
  statistically clear.
- The crop depends on the segmenter, whose incision labels were AI-drafted (05, part 4). If it misses the wound, the
  model falls back to whole-photo features for the crop half (4.2 % of photos), which it has seen in training.
- Calibration is measured on SurgWound; on our own patients' phone photos it must be re-checked (and refitted if needed).
- Sensitivity on discharge and urgency is 0.75 on test, below the 0.85 target set on CV: reported, not tuned on test.
