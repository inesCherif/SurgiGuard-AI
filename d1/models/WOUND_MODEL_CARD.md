# Wound-attribute model (D1-F) — model card

**What it does.** From one photo of a surgical wound, estimates six attributes that surgeons annotate: redness, swelling,
discharge, healing status, infection-risk rating and urgency. Each answer is *yes / no / uncertain* with a probability,
plus an evidence map showing which parts of the photo pushed the redness score up.

**What it is not.** Not a diagnosis, not a medical device. It never closes or downgrades a case: a "yes" on discharge,
risk or urgency can only ADD a reason for the clinician to look; a "no" removes nothing. Urgent symptoms go through the
rule-based urgent lane, which never waits for this model.

## Current version: `wound-attr-0.4` (v2 step 4, details in docs/wound/06_CLASSIFIER_V2.md)
- **Features = whole photo + square crop around the wound** (learned segmenter `wound_seg/unet_v2.onnx`,
  `WoundCropper`, 35 % context, fallback to the whole photo on 4.2 % of photos). Chosen by mean CV AUROC on
  train+val (whole 0.818, crop 0.830, whole+crop 0.836) before looking at the test split.
- **Calibrated probabilities**: Platt scaling fitted on out-of-fold logits; the operating point is chosen on the raw
  scale and mapped through the calibration, so decisions are unchanged and the percentage shown is honest.

| Attribute | Test n (+) | AUROC 0.3 → **0.4** [95 % CI] | Sens / Spec | Abstained | ECE raw → calibrated |
|---|---|---|---|---|---|
| Redness | 131 (40+) | 0.759 → **0.766** [0.67, 0.85] | 0.92 / 0.34 | 12.2 % | 0.124 → 0.077 |
| Swelling | 115 (21+) | 0.767 → **0.814** [0.71, 0.90] | 0.90 / 0.61 | 11.3 % | 0.191 → 0.048 |
| Discharge | 125 (16+) | 0.818 → **0.873** [0.78, 0.95] | 0.75 / 0.75 | 10.4 % | 0.217 → 0.064 |
| Not yet healed | 137 (55+) | 0.771 → **0.803** [0.73, 0.87] | 0.83 / 0.58 | 5.8 % | 0.077 → 0.090 |
| Infection-risk rating | 137 (20+) | 0.843 → **0.853** [0.76, 0.94] | 0.90 / 0.76 | 8.8 % | 0.151 → 0.072 |
| Needs review | 137 (16+) | 0.823 → **0.809** [0.69, 0.91] | 0.75 / 0.74 | 8.0 % | 0.157 → 0.096 |

Paired test differences (0.4 − 0.3) are clearly positive for swelling [+0.009, +0.090] and discharge [+0.014, +0.110];
the others are within noise. Unfamiliar-photo gate: 0.7 % of test photos flagged.

*The sections below describe the data and method shared by both versions; the results table further down is the
historical `wound-attr-0.3` (whole photo, uncalibrated).*

## Data
- **SurgWound** (Hugging Face `Sonulee/SurgWound`, licence **CC BY-SA 4.0**; SurgWound-Bench, arXiv 2508.15189):
  686 usable surgical-wound photos, each labelled by surgical experts at The Ohio State University Wexner Medical Center.
- Official split kept: **train 480 + val 69** for every training decision, **test 137 used once** at the end.
- Labels marked *Uncertain* by the annotators are left out of that attribute (never guessed).
- Caveat: images were collected from public online sources (variable cameras, light, framing, skin tones); performance
  on our patients' phone photos is **not yet measured**.

## Method (CPU, ~15 min, no fine-tuning of the encoder)
1. Frozen image encoder; features = [CLS token, mean of patch tokens], averaged with the mirror image.
   Encoder chosen by **cross-validation on train+val only**: CLIP ViT-B/16 (mean CV AUROC 0.817)
   vs DINOv2-small (0.75). Encoder in use: `openai/clip-vit-base-patch16`.
2. One class-balanced, L2-regularised logistic regression per attribute (regularisation chosen by 5-fold CV AUROC).
3. Operating threshold chosen on out-of-fold predictions for **sensitivity ≥ 0.85** (missing a sign is worse than a
   false alarm), and an **uncertain band** (±0.05 around the threshold) where the model abstains.
4. Explanation: **occlusion sensitivity** — each cell of a 5×5 grid is greyed out and the drop in the redness score is
   measured; the heat map shows which regions the decision depended on. (Raw CLIP patch-token attributions were tried
   first and rejected: they highlighted the background, a known artefact.)

Why this design: 549 labelled photos are too few to fine-tune a large network without over-fitting; a frozen
pre-trained encoder + linear heads is the standard, reproducible approach for small medical datasets, trains on a laptop,
and stays explainable.

5. **Unfamiliar-photo gate** (out-of-distribution): cosine distance from the new photo to its 5 nearest training photos
   in encoder space; above the 97.5th percentile of the training photos' own distances, the model abstains on every
   attribute. On the untouched test split it flags 2.2 % of photos. Found necessary in practice: close-up phone photos
   with dark backgrounds otherwise produced confident answers.

## Results on the untouched test split (historical: wound-attr-0.3)
| Attribute | Test n (positives) | AUROC [95 % CI, bootstrap] | CV AUROC | Sens / Spec (all photos) | Sens / Spec (after abstaining) | Abstained |
|---|---|---|---|---|---|---|
| Redness (erythema) | 131 (40+) | **0.759** [0.665–0.839] | 0.748 | 0.9 / 0.462 | 0.914 / 0.452 | 17.6 % |
| Swelling (edema) | 115 (21+) | **0.767** [0.65–0.865] | 0.77 | 0.81 / 0.574 | 0.85 / 0.556 | 20.0 % |
| Discharge (exudate: serous, sanguineous, purulent) | 125 (16+) | **0.818** [0.692–0.922] | 0.855 | 0.75 / 0.633 | 0.857 / 0.649 | 13.6 % |
| Not yet healed | 137 (55+) | **0.771** [0.687–0.844] | 0.835 | 0.818 / 0.512 | 0.824 / 0.521 | 10.9 % |
| Infection-risk rating medium/high | 137 (20+) | **0.843** [0.746–0.929] | 0.856 | 0.9 / 0.675 | 0.889 / 0.68 | 11.7 % |
| Urgency: clinic or emergency (vs home care) | 137 (16+) | **0.823** [0.703–0.917] | 0.841 | 0.812 / 0.653 | 0.812 / 0.649 | 7.3 % |

- Baseline for redness: the OpenCV redness measure we used before this model scores **AUROC 0.503**
  on the same test photos (chance level); the trained model scores **0.759**.
- A majority-class predictor has balanced accuracy 0.5 on every attribute.
- Small positive counts (e.g. 16 photos with discharge in the test split) give wide confidence intervals: they are
  reported, not hidden.

## Known limits / what we would do next
- Validate on our own consented patient photos (same phone, same lighting guidance) before any clinical use.
- More labelled surgical wounds, especially discharge and high-risk cases; per-skin-tone performance analysis.
- The attributes are what *surgeons labelled from a photo*; warmth, pain, smell and depth cannot be seen.

Reproduce: `python scripts/prepare_surgwound.py <hf json files> <prep>`, then `python scripts/wound_view_experiment.py <prep>` (view choice) and `python scripts/train_wound_model.py <prep> --encoder openai/clip-vit-base-patch16`
(metrics in `models/wound_model_metrics.json`).

## Files and inference (v2, Nov 2026)
- `models/wound_heads.json` + `models/wound_heads.npz` (~1.7 MB): **portable bundle** read by `model_bundle.py`. Plain arrays (scaler mean/std, logistic weights and bias, thresholds, OOD reference) + a versioned JSON manifest; inference is numpy only, no scikit-learn, no pickle.
- `models/wound_heads.joblib` (~1.8 MB): the original scikit-learn 1.9.1 pickle, kept for provenance. `scripts/export_wound_heads.py` converts it and refuses to write if any probability differs by more than 1e-9 (measured: 1.1e-16 over 549 photos x 6 heads).
