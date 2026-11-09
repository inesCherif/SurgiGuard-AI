# Wound CV: problems met and how they were solved

A running engineering journal. Each entry gives **what went wrong**, **why** it happened, **what was done**, and **how we know it worked**.

- **Part 1:** the hackathon (v1, 10–11 October 2026).
- **Part 2:** problems found while auditing v1 for the school project; they define Wound CV v2.
- **Part 3:** new entries, added as v2 is built.

---

## Part 1: hackathon (v1)

### P1. "We have no wound images we are allowed to use"
- **Problem:** the demo needed realistic surgical wound photos. Random internet pictures cannot be used legally or ethically, and invented numbers on a fake image prove nothing.
- **Solution:** used openly licensed Wikimedia photos (`backend/seed/photos/`, credits in `CREDITS.md`). Every number shown is computed by code from the real image ("no fake numbers", commit `4e6763d`).

### P2. No wound-segmentation model fits surgical incisions
- **Problem:** the public wound-segmentation checkpoints (FUSegNet, TiSage, …) are trained on **diabetic-foot and pressure ulcers**. Those are large, round, open wounds. A fresh surgical incision is a **thin line** with staples or sutures. Some checkpoints are also non-commercial or need a GPU.
- **Solution:** a classical, explainable OpenCV pipeline (`pipeline.py`). It compares each pixel with its **local background** in CIELAB, detects **staple chains** with a top-hat filter and a curve fit, and handles marker ink (purple) separately from redness.
- **Check:** synthetic photos with known ground truth in `d1/tests/test_wound.py`: drawn incisions, light and dark skin, marker ink, blurry or dark photos.

### P3. Lighting makes skin look red
- **Problem:** a warm lamp or yellow indoor light shifts every pixel towards red or yellow, so a global "red" threshold fires on healthy skin.
- **Solution:**
  - Colour is always measured **relative to the patient's own skin in the same photo**: a band 25–60 px around the wound, using Lab Δa\*.
  - Global grey-world white balance was tried and **disabled**. When skin fills the frame, grey-world turns the skin grey (`pipeline.py` line 178).

### P4. Comparing two photos taken on different days
- **Problem:** the patient never holds the phone at the same distance or angle. A closer photo makes the wound look bigger, and a photo of a different spot could be reported as "the wound changed".
- **Solution:**
  - **ORB features + RANSAC homography** register the new photo onto the reference.
  - The area is corrected by scale².
  - If the photos do not match (fewer than 12 inliers, ratio below 0.25, absurd scale), the comparison is **refused** instead of showing a fake change.

### P5. Bad photos give confident nonsense
- **Problem:** blurry, dark or glary photos, or photos without enough skin, produced random measurements.
- **Solution:** a quality gate runs first: Laplacian variance (sharpness), L\* (exposure), glare and skin share. "Unusable" photos ask for a retake and never produce a number.

### P6. 48-megapixel photos crashed the app
- **Problem:** recent phones capture 48 Mpx. One decoded bitmap is 8000 × 6000 × 4 bytes ≈ **192 MB**, and the Android app ran out of memory.
- **Solution:** capture at about 2 Mpx and downscale before display (commit `6431361`). The analysis works at 640 px anyway.

### P7. The OpenCV redness score turned out to be at chance level
- **Problem:** once surgeon labels (SurgWound) were available, the hand-made redness measure was evaluated against them. It scored **AUROC 0.503**, which is the same as flipping a coin.
- **Why:** the surgeons' "erythema" label depends on context (spread, pattern, skin tone, where the incision is) that a single Δa\* number cannot capture. Photo-to-photo differences in camera and light dominate the signal.
- **Solution:** trained a model on the surgeons' labels. A **frozen CLIP ViT-B/16** encoder with **logistic-regression heads** reaches test AUROC 0.76–0.84.
- **Lesson:** always measure a hand-made feature against ground truth before trusting it.

### P8. Too little data to train a deep network
- **Problem:** only 549 labelled training photos. Fine-tuning a ViT on that would over-fit.
- **Solution:**
  - Transfer learning: a frozen pre-trained encoder plus linear heads. It trains in about 15 min on a CPU and is reproducible.
  - The encoder was chosen by cross-validation on the training data only: CLIP 0.817 vs DINOv2 0.75.
  - The official test split was used **once**.

### P9. Missing an infection is worse than a false alarm
- **Problem:** the default 0.5 threshold optimises accuracy, not patient safety.
- **Solution:**
  - Thresholds chosen for **sensitivity ≥ 0.85** on out-of-fold predictions.
  - Within ±0.05 of the threshold the model says **"uncertain"** instead of guessing.

### P10. Confident answers on photos unlike the training data
- **Problem:** close-up phone photos with a dark background got confident "yes" or "no" answers.
- **Solution:** an **unfamiliar-photo (out-of-distribution) gate**. If the photo is too far (cosine distance) from its 5 nearest training photos, every attribute becomes uncertain. It flags 2.2% of the test photos (commit `2fc5566`).

### P11. Explanation maps highlighted the background
- **Problem:** raw CLIP patch-token attributions pointed at the background, a known ViT artefact.
- **Solution:** **occlusion sensitivity**: grey out each grid cell and measure the score drop. It is faithful to the full model, but costs about 25 forward passes, so it is opt-in.

### P12. Slow analysis (about 4 s per photo and more)
- **Solution:**
  - One forward pass by default (the evidence map is opt-in).
  - The encoder is **warmed up at server start**.
  - The model and the vision LLM run in a **background job**, so the upload returns immediately with the OpenCV result.

### P13. An LLM might "diagnose"
- **Problem:** vision LLMs readily write "infected" or "healing well", which is a diagnosis.
- **Solution:**
  - A strict system prompt: describe only what is visible, say "appears", give a confidence, answer in JSON.
  - Pydantic validation of the reply.
  - A code `output_guard` deletes diagnosis wording, and the deleted sentences are counted and shown.
  - The LLM can never be the only reason for an alert; a code rule decides.

---

## Part 2: audit of v1 (school project, November 2026)

These are the problems found by reading and testing v1 line by line. They define the Wound CV v2 work. Each will move to Part 3 with its solution and measured result.

| # | Problem | Evidence | Planned fix |
|---|---|---|---|
| A1 | The review rule still alerts on the chance-level redness score | `d1_wound.py` `decide()`: Δring ≥ 4, Δredness ≥ 6; model card: AUROC 0.503 | Rule based only on validated signals; redness trend kept as display-only information |
| A2 | Two different analysers (patient vs clinician) | `backend/app/wound_image.py` (512 px, sharpness cut-off 12) vs `pipeline.py` (640 px, cut-off 25) | One analyser |
| A3 | The model looks at the whole photo, not the wound | `model.predict()` encodes the full image; the mask is unused | Learned wound segmentation, then classify the wound crop |
| A4 | No real size, no colour calibration | Area is a % of the frame; no white reference | Printable calibration card (ArUco + colour patches + mm scale) |
| A5 | Pixel-based kernels (25/51/121 px) do not scale with distance | `_finish()`, `measure()` | Express sizes in mm (with the card) or relative to the wound size |
| A6 | Uncalibrated probabilities, fixed ±0.05 band | `model.py` | Temperature scaling and calibration plots |
| A7 | Stale docs: DINOv2 / 7×7 / 100 KB | `model.py` docstring, model card | Fix |
| A8 | Thread-unsafe global state | `_staple_chain.last_count` | Return the count |
| A9 | Not evaluated per skin tone or on phone photos | Model card "known limits" | Subgroup report plus a small consented phone-photo test set |

---

## Part 3: v2 log

### V1. The vision model disappeared (Nov 2026)
- **Problem:** the wound vision reading silently fell back to "unavailable". Groq had **retired** `meta-llama/llama-4-scout-17b-16e-instruct` (HTTP 404).
- **Why it matters:** a hosted model is an external dependency that can vanish at any time. The design already handled it safely: the code measurements and the rule still ran, and the LLM is never the only gate.
- **Fix:** listed the models Groq serves (`GET /openai/v1/models`) and tested which ones accept an image. `qwen/qwen3.8-27b` does, and gives an accurate description of `day1.jpg` (staples, purple marker, yellowish skin). It is now the default in `VISION_MODELS` (`backend/app/integrations/d1_wound.py`). Mistral Pixtral stays as the alternative (`LLM_ROUTE_WOUND_VISION=mistral:pixtral-12b-latest`).
- **Lesson:** pin model names in one place, check provider availability at startup, and always keep a non-LLM fallback.

*(more entries are added here as the work progresses)*
