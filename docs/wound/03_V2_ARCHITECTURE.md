# Wound CV v2: architecture

How the wound-photo code is organised after v2 step 1 (November 2026):
- why it is built this way;
- which design principles and patterns it uses, and where;
- how to extend it safely.

The problems that drove each decision are in [02_PROBLEMS_AND_SOLUTIONS.md](02_PROBLEMS_AND_SOLUTIONS.md) (entries V6–V10). The v1 baseline is in [01_HOW_V1_WORKS.md](01_HOW_V1_WORKS.md).

---

## 1. Before → after

| | v1 (hackathon) | v2 step 1 |
|---|---|---|
| Classical CV code | one 430-line `pipeline.py` | 8 modules, one responsibility each |
| Analysers in the system | **two**: `wound_image.py` for the patient, `pipeline.py` for the clinician, with different thresholds | **one**; the patient view is an adapter over it |
| Comparison of two photos | the patient path compared **without** checking it was the same wound | always after ORB + RANSAC proof that both photos show the same site |
| Segmenter | hard-wired functions | `Segmenter` interface (Strategy), injected |
| Staple count | stored on the function object (thread-unsafe global) | returned inside an immutable `Segmentation` |
| Model heads | scikit-learn **pickle** (broke on a version change) | portable **JSON + NPZ** bundle, numpy-only inference, versioned |
| Encoder settings | CLIP/DINOv2 settings copy-pasted twice | one `Encoder` class |
| Alert reasons | inline `if`s, including a chance-level redness score | `ReviewRule` objects; the chance-level reason removed |
| Thresholds and kernel sizes | literals scattered through the code | `config.py` (`Limits`, `Kernels`) |

---

## 2. Module map

```
d1/src/surgiguard_d1/wound/
├── config.py        Limits (decision thresholds) · Kernels (pixel sizes) · VERSION · NOT_MEASURED
├── colour.py        load() EXIF + resize · to_lab() CIELAB · skin_mask() YCrCb · morphology helpers
├── quality.py       assess_quality(): exposure, Laplacian sharpness, glare, skin share → good / questionable / unusable
├── segmentation.py  Segmenter (interface) · ClassicalSegmenter · Segmentation (value object) · robust_z()
├── measurement.py   measure(): redness vs own skin, spread ring, shape, crust / pale, closure material
├── registration.py  register(): ORB + RANSAC homography · compare(): trends only for a proven same site
├── render.py        overlay() · encode_jpeg()
├── pipeline.py      WoundAnalyzer (Facade) + analyze() function API
├── model_bundle.py  LinearHead · OODGate · ModelBundle (JSON + NPZ) · load() factory · sklearn adapter
└── model.py         Encoder (frozen CLIP) · embed() · predict() · occlusion_map() · warm_up()

backend/app/
├── wounds.py                    store the photo → run the ONE analysis
└── integrations/
    ├── d1_wound.py              analyze → model_reading → vision_review → decide (background job)
    ├── wound_views.py           Adapter: d1 result → patient fields (analysis / comparison)
    └── wound_rules.py           ReviewRule strategies + review_reasons()
```

**Dependency direction:** `pipeline` → stages → `colour`/`config`. No stage imports another stage except through the `Segmentation` value object. The backend depends on the d1 package; the d1 package never imports the backend.

### Data flow of one photo
```
bytes ─► colour.load ─► quality.assess_quality ──unusable──► "retake" (no numbers)
                               │ ok
                               ▼
                 Segmenter.segment (Strategy) ──none found──► reason
                               │ Segmentation(mask, lab, reference_lab, staple_count …)
                               ▼
                    measurement.measure ──► measurements
                               │ reference photo?
                               ▼
     registration.register (ORB+RANSAC) ──not same site──► refused, with reason
                               │ proven
                               ▼
                    registration.compare ──► trend, area change (distance-corrected)
                               ▼
              backend: wound_views (patient view) + d1 (clinician view)
                               ▼
     background job: model.predict (local CLIP + numpy heads) → vision LLM → wound_rules → alert?
```

---

## 3. Design patterns used (GoF and others)

| Pattern | Where | Why it is the right tool here |
|---|---|---|
| **Facade** | `pipeline.WoundAnalyzer` | Callers (backend, tests) need one call, `analyze(bytes, reference)`, not the order of 6 stages. The stages can change behind it. |
| **Strategy** | `segmentation.Segmenter` / `ClassicalSegmenter` | Step 3 adds a learned segmenter. Both must be swappable at run time (and comparable in a benchmark) without touching quality, measurement or the backend. |
| **Strategy** (rule objects) | `wound_rules.ReviewRule`: `ModelSignRule`, `AreaGrowthRule`, `LiveVisionRule` | Each alert reason is independent, testable on a plain dict, and can be added or removed alone. `review_reasons()` simply runs the tuple, like a chain where every handler can contribute. |
| **Adapter** | `wound_views.patient_analysis / patient_comparison` | The React Native screens expect the v1 field layout. The adapter converts the single analysis instead of keeping a second analyser alive. |
| **Adapter** | `ModelBundle.from_sklearn_dict` | Turns the legacy pickle (scikit-learn objects) into the portable value objects. |
| **Factory Method** | `model_bundle.load(path)` | Chooses the reader from the file type: `.json` → portable, `.joblib` → legacy through the adapter. Callers never branch on formats. |
| **Value Object** (immutable `@dataclass(frozen=True)`) | `Segmentation`, `LinearHead`, `OODGate`, `ModelBundle`, `Limits`, `Kernels` | No hidden mutation between stages; safe to share across threads; easy to compare in tests. |
| **Dependency Injection** | `WoundAnalyzer(segmenter=…, limits=…, kernels=…)` | Tests inject a fake segmenter (`test_wound_architecture.py`). The benchmark will inject each candidate. |
| **Memoised factory** | `get_encoder()` with `lru_cache`, `load_heads()` | The 600 MB encoder is loaded once per process; `warm_up()` does it at server start. |

---

## 4. SOLID, concretely

| Principle | How v2 applies it |
|---|---|
| **S**ingle responsibility | One module per stage. `quality.py` only decides whether a photo is usable; `registration.py` only decides whether two photos match and what changed. |
| **O**pen/closed | New segmenter = new class implementing `segment()`. New alert reason = new `ReviewRule`. `WoundAnalyzer` and `decide()` are not edited. |
| **L**iskov substitution | Any `Segmenter` returns a `Segmentation` with the same contract (`mask is None` ⇒ `reason` set; otherwise `reference_lab` set). The fake segmenter in the tests passes the same pipeline. |
| **I**nterface segregation | `Segmenter` has one method; `ReviewRule` has one method and only reads a dict. Nothing depends on methods it does not use. |
| **D**ependency inversion | `WoundAnalyzer` depends on the `Segmenter` abstraction, not on `ClassicalSegmenter`. The backend rules depend on the stored observation format, not on how the d1 code computed it. |

**DRY (no duplicated code):**
- One image loader and CIELAB conversion: v1 had two, a hand-written sRGB→Lab in numpy and OpenCV's.
- One analyser instead of two.
- The encoder settings live in `Encoder` instead of being copied in `embed()` and `_heatmap()`.
- The model review heads are defined once, in `ModelSignRule`.

---

## 5. Computer-vision best practices applied

| Practice | Where |
|---|---|
| **Quality gate first**: never measure a bad image | `quality.py`; "unusable" → retake, no numbers |
| **Relative, not absolute, colour**: compare with the patient's own skin in the same photo | `segmentation._finish` reference band, `measurement` Δa\* |
| **Perceptual colour space** (CIELAB) for colour differences | `colour.to_lab` |
| **Local background normalisation** instead of global thresholds (uneven light) | `ClassicalSegmenter._score`: Gaussian-blur background per channel |
| **Robust statistics** (median / MAD) so the wound itself does not distort the normalisation | `segmentation.robust_z` |
| **Geometric verification before any longitudinal comparison** | `registration.register`: RANSAC inliers, inlier ratio, plausible scale |
| **Scale correction** of areas between photos | `compare`: area ratio ÷ scale² |
| **Explainability**: show what was measured | `render.overlay`, occlusion map |
| **Abstention**: uncertainty band, out-of-distribution gate | `LinearHead.decide`, `OODGate` |
| **Test-time augmentation** (mirror) for stabler features | `model.embed` |
| **Frozen pre-trained encoder + linear probe** on a small dataset (549 photos) | `model.py`, `train_wound_model.py` |
| **Held-out test split used once**; thresholds chosen on out-of-fold predictions | `train_wound_model.py` |
| **Portable, safe model files** (`allow_pickle=False`), versioned, verified on export | `model_bundle.py`, `export_wound_heads.py` |
| **Reproducibility**: one config file, deterministic code paths | `config.py` |

**Still to do** (next v2 steps):
- Kernel sizes in pixels → millimetres with a calibration card (step 2).
- Real white balance and colour correction from reference patches (step 2).
- Learned segmentation measured with Dice/IoU (step 3).
- Classify the wound crop rather than the whole photo; calibrated probabilities, e.g. temperature scaling (step 4).
- Evaluation per skin tone and on real phone photos.

---

## 6. How the refactor was proven safe

A refactor is only safe if behaviour is **measured** to be unchanged:

1. **Regression against the old code.** The v1 `pipeline.py` was loaded from git and run side by side with v2 on the 3 real demo photos and all 6 ordered pairs. All 9 outputs were identical, number for number.
2. **Numerical equivalence of the model.** The numpy heads vs scikit-learn's `predict_proba` on the 549 real training feature vectors × 6 heads: max |difference| = **1.1 × 10⁻¹⁶** (machine precision). The export script refuses to write a file above 10⁻⁹. Through the real CLIP encoder, the day-3 photo gives exactly the same 6 probabilities as before.
3. **Test suites.** d1: 183 + 18 new tests (architecture, bundle). Backend: 194 tests, including rewritten tests for the old bug and new rule and adapter tests.
4. **Behaviour that changed on purpose** (each with a test):
   - photos of different sites are no longer compared in the patient view;
   - the redness trend is no longer an alert reason.

---

## 7. How to extend

**Add a segmenter** (step 3):
```python
class LearnedSegmenter:                      # implements the Segmenter interface
    name = "unet-v1"
    def segment(self, bgr) -> Segmentation:  # same contract as ClassicalSegmenter
        ...
analyzer = WoundAnalyzer(segmenter=LearnedSegmenter())
```

**Add an alert reason:**
```python
@dataclass(frozen=True)
class MyRule:
    name: str = "my_rule"
    def reason(self, value: dict) -> str | None: ...
REVIEW_RULES = (*REVIEW_RULES, MyRule())
```

**Retrain the heads:** run `scripts/train_wound_model.py`. It writes the pickle and the portable bundle; `scripts/export_wound_heads.py` re-converts and verifies any pickle.
