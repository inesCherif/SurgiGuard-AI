# Start here: reading order

The files are numbered, so read them in that order. Each line says what you will learn and roughly how long it takes.

## Part A: the whole project (`docs/`)

| # | File | What you learn | Time |
|---|---|---|---|
| 00 | **this file** | the reading order | 1 min |
| 01 | [01_RUN_WITHOUT_HARDWARE.md](01_RUN_WITHOUT_HARDWARE.md) | start everything on a PC, with no ESP32 and no phone | 5 min |
| 02 | [02_ARCHITECTURE.md](02_ARCHITECTURE.md) | the problem, the 5 programs, how a reading and a photo flow through them | 10 min |
| 03 | [03_INTERFACES.md](03_INTERFACES.md) | every screen with a screenshot: what it shows and which code or AI is behind it | 15 min |
| 04 | [04_REBUILD_GUIDE.md](04_REBUILD_GUIDE.md) | rebuild the project from zero, file by file in build order; every model, command, config and fix | 45 min |

## Part B: wound computer vision, my part (`docs/wound/`)

| # | File | What you learn | Time |
|---|---|---|---|
| 01 | [wound/01_HOW_V1_WORKS.md](wound/01_HOW_V1_WORKS.md) | the hackathon pipeline, stage by stage: quality gate, CIELAB segmentation, staples, ORB registration, CLIP model, vision LLM, review rule | 30 min |
| 02 | [wound/02_PROBLEMS_AND_SOLUTIONS.md](wound/02_PROBLEMS_AND_SOLUTIONS.md) | the engineering journal: every problem met, its cause, the fix and the proof (hackathon P1–P13, audit A1–A9, v2 V1–V14…) | 20 min |
| 03 | [wound/03_V2_ARCHITECTURE.md](wound/03_V2_ARCHITECTURE.md) | v2 design: module map, GoF patterns, SOLID, DRY, CV best practices, how the refactor was proven safe | 20 min |
| 04 | [wound/04_CALIBRATION.md](wound/04_CALIBRATION.md) | v2 step 2: calibration card (ArUco homography, white balance in linear RGB, exact mm²); results on ground-truth photos and honest limits | 20 min |
| 05 | [wound/05_SEGMENTATION.md](wound/05_SEGMENTATION.md) | v2 step 3: learned segmentation. Benchmark for any segmenter, click-to-outline tool, U-Net trained on FUSeg (Dice 0.82 vs classical 0.17 on 200 test photos), ONNX inference | 20 min |

**To prepare an interview or the defence:** read B-01 for *what* it does, B-02 for *what went wrong and how I solved it*, and B-03 for *how it is engineered*.

## Other reference files
- [`../README.md`](../README.md): project summary and the original run instructions
- [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md): who did what
- [`../d1/models/WOUND_MODEL_CARD.md`](../d1/models/WOUND_MODEL_CARD.md): the trained model's data, method and test results
- `../PROJECT.md`, `../INTEGRATION.md`: the team's hackathon brief and integration notes

## Project status (last updated 9 November 2026)

| Step | Status |
|---|---|
| Own repo, runs without hardware, full documentation | ✅ done |
| Wound v2, step 1: architecture clean-up (modules, Strategy, Facade, portable model, one analyser, rule strategies) | ✅ done: [wound/03_V2_ARCHITECTURE.md](wound/03_V2_ARCHITECTURE.md) |
| Wound v2, step 2: calibration card (real mm / cm², colour correction) | ✅ done: [wound/04_CALIBRATION.md](wound/04_CALIBRATION.md) |
| Wound v2, step 3: learned wound segmentation (new `Segmenter`, Dice / IoU vs classical) | 🟡 parts 1–3 done: [wound/05_SEGMENTATION.md](wound/05_SEGMENTATION.md). ⏭ **next = part 4**: outline the 160 SurgWound photos (`d1/scripts/outline_wounds.py`), fine-tune, score on incisions, choose the default |
| Wound v2, step 4: classifier on the wound crop, calibrated probabilities, honest benchmark | planned |

**To resume:** `.\scripts\demo.ps1` starts everything (Docker must be running). Run tests with the backend stopped:
`$env:HF_HUB_OFFLINE="1"; $env:LLM_MODE="stub"`, then `pytest` in `d1/` and `backend/`.
