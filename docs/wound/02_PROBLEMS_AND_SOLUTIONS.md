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
| A1 ✅ V8 | The review rule still alerts on the chance-level redness score | `d1_wound.py` `decide()`: Δring ≥ 4, Δredness ≥ 6; model card: AUROC 0.503 | Rule based only on validated signals; redness trend kept as display-only information |
| A2 ✅ V7 | Two different analysers (patient vs clinician) | `backend/app/wound_image.py` (512 px, sharpness cut-off 12) vs `pipeline.py` (640 px, cut-off 25) | One analyser |
| A3 | The model looks at the whole photo, not the wound | `model.predict()` encodes the full image; the mask is unused | Learned wound segmentation, then classify the wound crop |
| A4 | No real size, no colour calibration | Area is a % of the frame; no white reference | Printable calibration card (ArUco + colour patches + mm scale) |
| A5 | Pixel-based kernels (25/51/121 px) do not scale with distance | `_finish()`, `measure()` | Express sizes in mm (with the card) or relative to the wound size |
| A6 | Uncalibrated probabilities, fixed ±0.05 band | `model.py` | Temperature scaling and calibration plots |
| A7 ✅ 1b | Stale docs: DINOv2 / 7×7 / 100 KB | `model.py` docstring, model card | Fix |
| A8 ✅ V10 | Thread-unsafe global state | `_staple_chain.last_count` | Return the count |
| A9 | Not evaluated per skin tone or on phone photos | Model card "known limits" | Subgroup report plus a small consented phone-photo test set |

---

## Part 3: v2 log

### V1. The vision model disappeared (Nov 2026)
- **Problem:** the wound vision reading silently fell back to "unavailable". Groq had **retired** `meta-llama/llama-4-scout-17b-16e-instruct` (HTTP 404).
- **Why it matters:** a hosted model is an external dependency that can vanish at any time. The design already handled it safely: the code measurements and the rule still ran, and the LLM is never the only gate.
- **Fix:** listed the models Groq serves (`GET /openai/v1/models`) and tested which ones accept an image. `qwen/qwen3.8-27b` does, and gives an accurate description of `day1.jpg` (staples, purple marker, yellowish skin). It is now the default in `VISION_MODELS` (`backend/app/integrations/d1_wound.py`). Mistral Pixtral stays as the alternative (`LLM_ROUTE_WOUND_VISION=mistral:pixtral-12b-latest`).
- **Lesson:** pin model names in one place, check provider availability at startup, and always keep a non-LLM fallback.

### V2. The trained model failed with `AttributeError`
- **Problem:** the clinic page showed "Model reading unavailable: model error (AttributeError)".
- **Why:** `d1/models/wound_heads.joblib` holds scikit-learn objects (`StandardScaler`, `LogisticRegression`) **pickled with scikit-learn 1.9.1**. This PC had 1.6.1. A pickle stores the internal attributes of the class; another version expects different attributes. scikit-learn even warns `InconsistentVersionWarning`.
- **Fix:** scikit-learn 1.9.1 in the venv, pinned in `backend/requirements.txt` (`>=1.9.1`).
- **Check:** `day3.jpg` → erythema yes (p 0.46 / thr 0.32), edema yes, exudate no, not_healed and needs_review **uncertain** (inside the ±0.05 band), photo familiar (OOD gate passed).
- **Lesson / v2 idea:** a pickle is not a portable model format. Save the version with the model and check it at load time, or export the heads as plain arrays (weights, bias, scaler mean/std) in JSON or ONNX.

### V3. "Re-run review" never re-ran the model
- **Problem:** after fixing V2, `POST /patients/{id}/wound-photos/{obs}/review` still returned the old error.
- **Why:** the endpoint re-ran the OpenCV analysis, the vision LLM and the rule, but **not** `model_reading()`. A failed model reading stayed stuck forever.
- **Fix:** `backend/app/routers/wounds.py` now calls `d1_wound.model_reading()` too.

### V4. Free-tier rate limits stopped the agents
- **Problem:** document analysis never finished; the log showed `HTTP 429 Rate limit reached for model openai/gpt-oss-120b`.
- **Why:** every agent (investigation, digest, check-in, chat, protocol) used the same Groq model, and each model has its own tokens-per-minute quota on the free tier.
- **Fix:** spread agents over models with `LLM_ROUTE_<AGENT>` in `backend/.env`: investigation → gpt-oss-120b; check-in, digest and D2 → gpt-oss-20b; chat and wound vision → Qwen 3.8; plan drafting → Mistral (a separate free quota). Everything stays free.

### V5. Not enough memory on a 15 GB laptop
- **Problem:** native crashes (`OpenBLAS: memory allocation failed`, PowerShell `0xe0434352`, `can't start new thread`) while the Docker build, the backend (CLIP loaded) and a test process with a second CLIP copy all ran at once.
- **Fix:** run heavy jobs one at a time, test the model **inside** the running backend (one CLIP copy), and limit Docker's WSL VM if needed (`%UserProfile%\.wslconfig`: `[wsl2] memory=4GB`).

## Step 1: architecture clean-up (Nov 2026)

Design rationale: [03_V2_ARCHITECTURE.md](03_V2_ARCHITECTURE.md).

### V6. A 430-line file did everything
- **Problem:** `pipeline.py` mixed loading, colour conversion, quality, segmentation, measurement, registration and drawing. Thresholds and kernel sizes were literals spread through it. Swapping the segmenter (step 3) would have meant editing everything.
- **Fix:**
  - 8 modules, one responsibility each (SRP), with every number in `config.py`.
  - A `Segmenter` interface (Strategy) with `ClassicalSegmenter` as today's implementation.
  - `WoundAnalyzer` as a Facade that receives the segmenter by injection (DIP).
  - The old `analyze()` function API is kept, so no caller changed.
- **How we proved nothing broke:** the old file was loaded from git and run side by side with the new code on all 9 demo cases. The outputs were **identical**. All d1 tests passed, plus new architecture tests (fake segmenter injection, concurrent analyses).

### V7. The patient and the clinician were told opposite things
- **Problem:** the patient column of the demo said "Redness area 5.7% → 7.7% (+35%): more redness than the reference photo". The clinician panel said "Not compared: the two photos do not seem to show the same area (10 consistent points)".
- **Why:**
  - Two analysers existed. `backend/app/wound_image.py` (the older one, numpy only, 512 px, its own thresholds) fed the patient view and the investigation agent, and it **compared any two photos without checking they showed the same wound**.
  - The demo photos day 1 and day 3 are of **two different patients** (`seed/photos/CREDITS.md`).
  - So the "+35%" was a comparison between two people.
- **Fix:**
  - `wound_image.py` deleted. The photo is analysed **once** by the d1 pipeline.
  - The patient view is produced by an **Adapter** (`integrations/wound_views.py`) that re-shapes that single result into the fields the app already reads, so the app code didn't change.
  - The investigation tool and the demo seed now use the registered comparison (`d1_wound.compare_photos`).
- **Tests:** two old tests asserted the bug (`trend == "up"` for two different patients). They now assert a **refusal**, and a new test checks that uploading the same site gives a real (stable) trend.
- **Lesson:** duplicated logic drifts. One source of truth, with adapters for each consumer.

### V8. An alert reason that was no better than a coin toss
- **Problem:** the review rule raised a clinician alert when the redness (mean Δa\* ≥ 6) or the redness around the wound (≥ 4) increased.
- **Why it is wrong:** against the surgeons' erythema labels (SurgWound test split), this OpenCV redness measure scores **AUROC 0.503**, which is chance level. Alerting on it adds random alarms and teaches clinicians to ignore alerts (alarm fatigue).
- **Fix:**
  - Rules are now Strategy objects (`integrations/wound_rules.py`): `ModelSignRule` (validated heads, AUROC 0.82–0.84), `AreaGrowthRule` (only for a registered, distance-corrected comparison), `LiveVisionRule` (live LLM, medium/high confidence).
  - The redness trend is still measured and **shown**, but is no longer an alert reason.
  - `decide()` just runs the rules (Open/Closed).
- **Tests:** `test_wound_rules.py` includes "chance-level redness trend no longer raises an alert".

### V9. Model file tied to one library version (follow-up of V2)
- **Problem:** the heads were a scikit-learn pickle. A pickle stores the private attributes of the classes that wrote it, so another scikit-learn version breaks it. Loading a pickle can also execute arbitrary code.
- **Fix:**
  - A logistic-regression head is only `sigmoid(w·(x − mean)/scale + b)`, so `model_bundle.py` stores those arrays in `.npz` plus a versioned JSON manifest, and evaluates them with numpy.
  - Files are loaded with `allow_pickle=False`.
  - A Factory Method picks the reader; an Adapter converts the old pickle.
  - The training script writes both formats.
- **Proof:** max |numpy − scikit-learn| = **1.1 × 10⁻¹⁶** over 549 real feature vectors × 6 heads. The export refuses to write above 10⁻⁹. The day-3 prediction through CLIP is identical. A test loads the bundle with scikit-learn **blocked**.

### V10. Shared mutable state in the staple detector
- **Problem:** `_staple_chain.last_count` stored the staple count **on the function object**. With two photos analysed at the same time (the backend runs jobs in threads), one photo could get the other's count.
- **Fix:** the count is returned inside the immutable `Segmentation` value object.
- **Test:** concurrent analyses give the same results as sequential ones.

*(more entries are added here as the work progresses)*
