# Wound-photo analysis v1: how it works

This is the version built at the hackathon (October 2026). It is the baseline that Wound CV v2 will be compared against.

## 1. The problem

**Surgical site infection** (SSI) is one of the most common complications after surgery. It affects roughly 2–5% of surgical patients and many more after some procedures. Most cases appear **after discharge, typically 5–10 days after the operation**, when the patient is at home. At home, nobody trained is looking at the wound.

The early visible signs are:
- redness spreading around the incision;
- swelling;
- discharge or pus;
- edges opening.

If a care team saw a photo at every dressing change, they could catch these early. But:
- clinicians cannot look at every photo of every patient every day;
- photos taken by patients are inconsistent: different light, distance, angle, blur and skin tones;
- an AI that says "infected" or "healed" would be a medical device making a diagnosis, which is unsafe and not allowed here.

**The goal:** turn a phone photo into **objective, explainable measurements**, a **trend** compared with the first photo, and a decision on **whether a clinician should look**. The system never gives a diagnosis.

## 2. The whole pipeline at a glance

```
 Phone (React Native / Android)                       AI backend (FastAPI)
 ┌───────────────────────────┐   POST /patients/{id}/wound-photos   ┌──────────────────────────────────────────────┐
 │ WoundScanScreen.tsx       │ ───────────────────────────────────► │ routers/wounds.py  upload_wound_photo        │
 │ camera, ~2 Mpx capture    │   (only after "dressing change       │  ├ wound_photos.py eligibility: is a photo   │
 │ "dressing changed" ✔      │    confirmed"; max 12 MB)            │  │  window open in the care plan?            │
 └───────────────────────────┘                                      │  └ wounds.py save_photo → uploads/           │
                                                                    │                                              │
       ┌────────────────────────── synchronous (~0.2 s) ────────────┤ integrations/d1_wound.py  process()          │
       ▼                                                            │                                              │
 STAGE A  d1/.../wound/pipeline.py   analyze()                      │                                              │
   1 load + quality check  →  "retake" if unusable                  │                                              │
   2 skin mask (YCrCb)                                              │                                              │
   3 segmentation: local redness / darkness / staples               │                                              │
   4 measurements vs the patient's OWN skin (CIELAB)                │                                              │
   5 ORB registration + comparison with the reference (day-1) photo │                                              │
   6 overlay JPEG (what the code looked at)                         │                                              │
       │                                                            │                                              │
       └─────────────── background job "wound_review" ──────────────┤                                              │
 STAGE B  d1/.../wound/model.py   predict()                         │                                              │
   CLIP ViT-B/16 (frozen) → 6 logistic-regression heads             │                                              │
   + unfamiliar-photo gate + abstention band (+ optional heat map)  │                                              │
 STAGE C  d1_wound.py  vision_review()                              │                                              │
   vision LLM (Groq Qwen 3.8 / Mistral Pixtral) describes      │                                              │
   what is visible, JSON, diagnosis words filtered                  │                                              │
 STAGE D  d1_wound.py  decide()                                     │                                              │
   code rule → "attention" alert for the clinician, or nothing      │                                              │
                                                                    └──────────────────────────────────────────────┘
 Display: clinician  backend/web/index.html renderWoundClinic()  |  patient  mobile/src/components/WoundAnalysis.tsx
```

## 3. Capture and upload

| Where | What it does |
|---|---|
| `mobile/src/screens/patient/WoundScanScreen.tsx` | In-app camera (react-native-vision-camera). The photo is sent only after the patient ticks "dressing change confirmed". |
| `mobile/src/services/api.ts` `uploadWoundPhoto` | Multipart upload with a 60 s timeout |
| `backend/app/routers/wounds.py:12` `upload_wound_photo` | Accepts the photo if a care-plan photo window is open. Outside a window it still accepts it, flagged `outside_planned_window`, if the patient confirmed a dressing change; otherwise it returns 409. Maximum 12 MB. |
| `backend/app/wound_photos.py:17` `eligibility` | Checks the dressing-change windows from the care plan. Photos are only taken when the dressing is off anyway, so the wound is never exposed just for the app. |
| `backend/app/wounds.py:23` `save_photo` | Stores the file and calls the analysis. `reference()` (line 17) is the patient's **first** photo, used as the baseline. |

**Why tie photos to dressing changes?** Clinically, the dressing should not be removed just to take a picture, because that adds infection risk. The care plan says when dressings are changed, and the app asks for a photo at those moments.

## 4. Stage A: classical computer vision (`d1/src/surgiguard_d1/wound/pipeline.py`)

The pipeline has no learning. Every number can be explained and tested on synthetic images where the true answer is known.

### A1. Loading and quality check: `load()` line 51, `assess_quality()` line 74
- Load the image, apply the EXIF rotation (phones store portrait photos rotated), and resize so the longest side is at most **640 px**.
- Measure four quality signals on the centre of the image (20–80% of width and height):

| Check | How | Limit |
|---|---|---|
| Brightness | mean of CIELAB **L\*** (0–100) | 22 – 92 |
| Sharpness | **variance of the Laplacian**: a blurry image has few sharp edges, so the variance is low | ≥ 25 |
| Glare | share of near-white pixels (> 245) | ≤ 8% |
| Skin in view | share of skin pixels (YCrCb box, see A2) | ≥ 20% |

- The result is `good`, `questionable` (shown with advice such as "hold the phone still") or `unusable` (no measurement at all, "retake"). **A bad photo never produces a number.**

### A2. Skin mask: `_skin_mask()` line 66
- Convert to **YCrCb** and keep pixels where 133 ≤ Cr ≤ 180 and 77 ≤ Cb ≤ 135. This box is a classic skin-detection rule.
- A small morphological opening removes isolated pixels.

### A3. Colour space: `_lab()` line 58
- Everything colour-related uses **CIELAB**:
  - **L\*** is lightness;
  - **a\*** runs from green (−) to red (+);
  - **b\*** runs from blue (−) to yellow (+).
- a\* is a direct "redness" axis, and CIELAB distances are close to how humans perceive colour differences. OpenCV's 8-bit Lab is rescaled to real units.

### A4. Segmentation (finding the incision): `segment()` line 170
The key idea is that **every pixel is compared with its own local background**, not with a global threshold. Lighting is never uniform in a phone photo.

1. **Local background:** a large Gaussian blur (σ = image size / 18) of each channel.
2. **Local redness:** `a_loc = a* − blur(a*)`. **Local darkness:** `dark_loc = blur(L*) − L*` (scabs and incision lines are darker).
3. **Robust z-scores** relative to skin pixels: (x − median) / (1.4826 · MAD), from `_z()` line 104. Median and MAD are not thrown off by outliers the way mean and standard deviation are.
4. **Score** = max(z(a_loc), 0) + 0.7·max(z(dark_loc), 0), multiplied by a **"not ink"** mask (`b_loc > −6`), plus 3 × bright metal.
   - Surgical marker ink and bruises are purple or blue, so b\* drops; inflammation is red-orange.
   - Bright metal pixels are staples.
5. **Threshold:** keep pixels above the 96th percentile of the score, with a floor of 3.0. The image border is ignored.
6. **Two ways to pick the wound:**
   - **Staple path** (`_staple_chain()` line 111):
     - Staples are small, bright, grey spots that stand out locally; a **top-hat** filter finds them.
     - If at least 5 of them form a chain (each within 9% of the image size of another), the chain is kept.
     - A **quadratic curve** is fitted three times, dropping outliers each time, because an incision is a smooth curve.
     - A strip along the curve becomes the wound mask.
   - **Fallback:** connected components of the thresholded score, ranked by area × elongation × strength × closeness to the centre. An incision is long, thin and usually centred in the photo. Pieces lying on the same line (`cv2.fitLine`) are merged, because incisions often show up broken.
7. `_finish()` line 249 builds the **peri-wound skin reference**: the median Lab of a band 25–60 px around the mask. This is the patient's own skin, so no fixed skin tone is assumed.

### A5. Measurements: `measure()` line 266
| Measurement | Meaning |
|---|---|
| `redness_mean_delta_a` | mean a\* of the wound minus a\* of the patient's own skin |
| `redness_area_pct` | % of wound pixels with Δa\* > 8 |
| `redness_spread_ring_delta_a` | Δa\* in a 25 px ring **outside** the wound: is redness spreading? |
| `wound_area_pct_of_frame` | area as a % of the image. **No real size**, because the camera distance is unknown. |
| `boundary_solidity`, `boundary_circularity` | shape regularity |
| `dark_crust_coloured_pct`, `pale_yellow_pct` | % of dark/crust-coloured pixels and pale-yellow (possible slough or fibrin) pixels |
| `closure_material` | number of staple-like spots |

`NOT_MEASURED = swelling, drainage, depth, warmth, pain`: the system says plainly what one 2-D photo cannot show.

### A6. Comparison with the reference photo: `register()` line 313, `compare()` line 341
- **ORB** keypoints (1500) are matched with a brute-force Hamming matcher, then a **RANSAC homography** maps the new photo onto the reference.
- The comparison is **refused** if:
  - there are fewer than 12 inliers;
  - the inlier ratio is below 0.25;
  - the scale is outside 0.4–2.5.
  When that happens, it reports "not the same area" instead of showing a fake change.
- The area change is corrected by **scale²**, because a photo taken closer makes the wound look bigger.
- The trend is `more_red` when Δredness or Δring redness is at least +4 a\* units, and `less_red` when Δredness is −4 or below.

### A7. Overlay: `overlay()` line 380
- A green tint and contour show exactly which region was measured. A person can check what the code looked at.

## 5. Stage B: trained model (`d1/src/surgiguard_d1/wound/model.py`)

**Data:** **SurgWound** (Hugging Face `Sonulee/SurgWound`, CC BY-SA 4.0). 686 surgical-wound photos labelled by surgeons at Ohio State University. The official split is kept: train 480 + val 69; **test 137, used once**.

**Method**, which is transfer learning with a frozen encoder:
1. **Encoder:** `openai/clip-vit-base-patch16`, a Vision Transformer, **frozen** (not fine-tuned).
   - Features are the [CLS token, mean of patch tokens] (1536 numbers per image), averaged with the mirrored image (test-time augmentation).
   - CLIP was chosen over DINOv2 by cross-validation: AUROC 0.817 vs 0.75.
2. **Heads:** one **logistic regression** per attribute (StandardScaler, class-balanced, L2 strength chosen by 5-fold CV).
   - The six attributes are erythema (redness), edema (swelling), exudate (discharge), not_healed, elevated_risk and needs_review.
   - Trained by `d1/scripts/train_wound_model.py`. The heads are saved in `d1/models/wound_heads.joblib`.
3. **Sensitivity-first threshold:** the threshold is the highest value that keeps **sensitivity ≥ 0.85** on out-of-fold predictions. Missing a sign is worse than a false alarm.
4. **Abstention band:** within ±0.05 of the threshold the answer is "uncertain" instead of a guess.
5. **Unfamiliar-photo gate (OOD):**
   - Compute the cosine distance from the photo to its 5 nearest training photos.
   - If it is above the 97.5th percentile of the training photos' own distances, **every** attribute becomes "uncertain".
   - It was added after close-up phone photos with dark backgrounds produced confident answers.
6. **Explanation:** occlusion sensitivity (`_heatmap()` line 119).
   - Grey out each cell of a 5×5 grid and measure how much the score drops.
   - Off by default (`WOUND_EVIDENCE_MAP=on`), because it costs about 25 extra forward passes.

**Results on the untouched test set** (`d1/models/WOUND_MODEL_CARD.md`):

| Attribute | Test AUROC [95% CI] |
|---|---|
| Erythema | 0.759 [0.665–0.839] |
| Edema | 0.767 [0.65–0.865] |
| Exudate | 0.818 [0.692–0.922] |
| Not healed | 0.771 [0.687–0.844] |
| Elevated risk | 0.843 [0.746–0.929] |
| Needs review | 0.823 [0.703–0.917] |

The key baseline: **the OpenCV redness measure from Stage A scores AUROC 0.503 on the same test photos, which is chance level.**

## 6. Stage C: vision-LLM description (`backend/app/integrations/d1_wound.py`)
- `vision_review()` line 194 sends the photo (base64) plus a text summary of Stage A and Stage B (`_cv_brief`) to a vision LLM:
  - Groq `qwen/qwen3.8-27b` (Llama-4-Scout at the hackathon, since retired by Groq);
  - or Mistral `pixtral-12b-latest`;
  - or OpenRouter Gemma 3.
- The `SYSTEM` prompt (line 153) says: describe only what is **visible**, never use disease words, say "appears", give a confidence, and answer in **JSON**.
- The reply is validated by the Pydantic model `VisionReview`. `output_guard` removes any sentence with diagnosis wording, and the removed sentences are counted and shown.
- **Offline (stub mode):** the code measurements are restated, clearly labelled "(offline, from code)".

## 7. Stage D: review rule (`decide()` line 239)
**Code decides**; the LLM is only one input. A clinician-only **"attention"** alert is raised if any of these hold:
- the model says **yes** on exudate, elevated_risk or needs_review (`MODEL_REVIEW_HEADS`, line 90);
- compared with the reference photo, ring redness rose by 4 or more, or mean redness by 6 or more;
- the marked area grew by 30% or more;
- a **live** LLM reading suggests review, with a medium- or high-confidence finding on redness, edges or discharge.

The patient only sees 4 sign attributes (`PATIENT_HEADS`, line 92), never "risk" or "urgency" wording.

## 8. Safety design choices worth explaining
- **Abstain instead of guessing:** bad quality → retake; unfamiliar photo → uncertain; near the threshold → uncertain.
- **Relative colour:** compare with the patient's own skin in the same photo. Lighting and skin tone partly cancel out.
- **Never a fake change:** registration must prove that both photos show the same site.
- **Asymmetric authority:** the model can only **add** a reason for review, never remove one. Urgent symptoms go through the rule-based urgent lane, which never waits for any model.
- **No diagnosis words**, filtered by code (`backend/app/safety/output_guard.py`).

## 9. Known weaknesses (what v2 fixes)
See [`PROBLEMS_AND_SOLUTIONS.md`](PROBLEMS_AND_SOLUTIONS.md). In short:
1. The Stage A redness measure is at **chance level** (AUROC 0.503), yet Stage D still alerts on it.
2. **Two analysers:** `backend/app/wound_image.py` (older, numpy-only) feeds the patient's trend chip, while `pipeline.py` feeds the clinician. The two can show different numbers.
3. The model encodes the **whole photo**: the segmentation mask is never used to focus it.
4. **No colour or size calibration:** area is a % of the frame and white balance is unknown.
5. Kernel sizes in **pixels** (25, 51, 121 px) do not scale with distance, and the thresholds are hand-picked.
6. Probabilities are **not calibrated**, and the band is a fixed ±0.05.
7. Docstrings are stale: they still say DINOv2 and a 7×7 grid; the code uses CLIP and 5×5.
8. Thread-unsafe global: `_staple_chain.last_count`.
9. Not evaluated on real patient phone photos or per skin tone.

## 10. How to run it alone
```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -e "..\d1[wound]"
.\.venv\Scripts\python.exe -c "from surgiguard_d1.wound import pipeline as p; import json; r=p.analyze(open('seed/photos/day1.jpg','rb').read()); print(json.dumps(p.strip_private(r), indent=1))"
cd ..\d1; ..\backend\.venv\Scripts\python.exe -m pytest tests/test_wound.py tests/test_wound_model.py -q
```
