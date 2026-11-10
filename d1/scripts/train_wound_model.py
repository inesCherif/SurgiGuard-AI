"""Train and evaluate the D1-F wound-attribute model on SurgWound (see src/surgiguard_d1/wound/model.py).

    python scripts/prepare_surgwound.py <hf files> <prep>      # once
    python scripts/train_wound_model.py <prep>                  # ~10 min on a laptop CPU

Protocol (no test-set peeking):
1. Frozen encoder features for every image (cached in <prep>/features*.npz). The encoder (CLIP ViT-B/16 vs
   DINOv2-small) is chosen by cross-validation on train+val only; CLIP won (0.817 vs 0.75).
2. Per attribute, on the official train+val split only: 5-fold stratified cross-validation picks the
   regularisation C (by AUROC), then out-of-fold probabilities pick the operating threshold for a target
   sensitivity of 0.85 and an "uncertain" band.
3. The final head is fitted on train+val and evaluated ONCE on the official test split: AUROC with a
   bootstrap 95% CI, sensitivity, specificity, balanced accuracy, abstention rate, and baselines
   (majority class; for erythema also the OpenCV redness measure used before this model).
Writes models/wound_heads.json + .npz (portable bundle, numpy-only inference), models/wound_heads.joblib (legacy)
and models/wound_model_metrics.json.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard_d1.wound import model as wm  # noqa: E402
from surgiguard_d1.wound import prob_calibration as pc  # noqa: E402

POS = {
    "erythema": ("erythema", {"Existent"}, {"Non-existent"}),
    "edema": ("edema", {"Existent"}, {"Non-existent"}),
    "exudate": ("exudate", {"Serous", "Sanguineous", "Purulent", "Seropurulent"}, {"Non-existent"}),
    "not_healed": ("healing", {"Not Healed"}, {"Healed"}),
    "elevated_risk": ("infection_risk", {"Medium", "High"}, {"Low"}),
    "needs_review": ("urgency", {"Clinic Visit (Yellow): Requires professional evaluation within 48 hours",
                                 "Emergency Care (Red): Seek immediate medical attention"},
                     {"Home Care (Green): Manage with routine care"}),
}
TARGET_SENSITIVITY = 0.85
OOD_K = 5
CS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
RNG = np.random.default_rng(0)


def features(prep: Path, rows: list[dict], encoder: str = wm.ENCODER) -> np.ndarray:
    cache = prep / ("features.npz" if encoder == wm.ENCODER else f"features_{encoder.replace('/', '_')}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["images"]) == [r["image"] for r in rows]:
            return z["x"]
    from PIL import Image
    xs = []
    for i, r in enumerate(rows):
        xs.append(wm.embed([Image.open(prep / r["image"])], encoder=encoder)[0])
        if i % 50 == 0:
            print(f"  features {i}/{len(rows)}", flush=True)
    x = np.stack(xs)
    np.savez(cache, x=x, images=np.array([r["image"] for r in rows]))
    return x


def _segmenter_id(segmenter: Path) -> str:
    """Name + size + modification time: a new or retrained segmenter invalidates the crop cache (journal V25)."""
    st = Path(segmenter).stat()
    return f"{Path(segmenter).name}:{st.st_size}:{int(st.st_mtime)}"


def crops(prep: Path, rows: list[dict], segmenter: Path) -> list[dict]:
    """Box around the wound for every photo (WoundCropper + learned segmenter), cached in <prep>/crops.json.
    The cache records which segmenter made it and is rebuilt when the segmenter changes."""
    cache, sid = prep / "crops.json", _segmenter_id(segmenter)
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if isinstance(c, dict) and c.get("segmenter") == sid and                 [b["image"] for b in c["boxes"]] == [r["image"] for r in rows]:
            return c["boxes"]
        print("  crop cache made with another segmenter (or old format): rebuilding", flush=True)
    from surgiguard_d1.wound.seg_data import read_bgr
    cropper = wm.get_cropper(str(segmenter))
    out = []
    for i, r in enumerate(rows):
        c = cropper.crop(read_bgr(prep / r["image"]))
        out.append({"image": r["image"], "box": list(c.box), "found": c.found})
        if i % 100 == 0:
            print(f"  crops {i}/{len(rows)}", flush=True)
    cache.write_text(json.dumps({"segmenter": sid, "boxes": out}), encoding="utf-8")
    return out


def crop_features(prep: Path, rows: list[dict], boxes: list[dict]) -> np.ndarray:
    """Encoder features of the wound crop of every photo, cached in <prep>/features_crop.npz together with the
    boxes they were computed from (new boxes -> features recomputed)."""
    cache = prep / "features_crop.npz"
    key = np.array([b["box"] for b in boxes], np.int64)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["images"]) == [r["image"] for r in rows] and "boxes" in z.files and np.array_equal(z["boxes"], key):
            return z["x"]
    from PIL import Image, ImageOps
    xs = []
    for i, (r, b) in enumerate(zip(rows, boxes)):
        im = ImageOps.exif_transpose(Image.open(prep / r["image"])).convert("RGB")
        xs.append(wm.embed([im.crop(tuple(b["box"]))])[0])
        if i % 50 == 0:
            print(f"  crop features {i}/{len(rows)}", flush=True)
    x = np.stack(xs)
    np.savez(cache, x=x, images=np.array([r["image"] for r in rows]), boxes=key)
    return x


def bootstrap_auc(y, p, n=1000):
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def at_threshold(y, p, thr, band):
    keep = np.abs(p - thr) >= band
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1) & keep).sum()); fn = int(((pred == 0) & (y == 1) & keep).sum())
    tn = int(((pred == 0) & (y == 0) & keep).sum()); fp = int(((pred == 1) & (y == 0) & keep).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"sensitivity": round(sens, 3), "specificity": round(spec, 3), "balanced_accuracy": round((sens + spec) / 2, 3),
            "abstained_pct": round(float((~keep).mean() * 100), 1), "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def opencv_redness(prep: Path, rows: list[dict]) -> np.ndarray:
    from surgiguard_d1.wound import analyze
    out = []
    for r in rows:
        m = analyze((prep / r["image"]).read_bytes()).get("measurements") or {}
        out.append(m.get("redness_mean_delta_a", np.nan))
    return np.array(out, dtype=float)


ENCODER_CANDIDATES = ["facebook/dinov2-small", "openai/clip-vit-base-patch16"]


def choose_encoder(prep: Path, rows: list[dict]) -> str:
    """Pick the encoder by mean cross-validated AUROC on train+val only (the test split is never looked at)."""
    split = np.array([r["split"] for r in rows])
    scores = {}
    for enc in ENCODER_CANDIDATES:
        x = features(prep, rows, enc)
        aucs = []
        for name, (col, pos, neg) in POS.items():
            lab = np.array([1 if r[col] in pos else 0 if r[col] in neg else -1 for r in rows])
            tr = (split != "test") & (lab >= 0)
            best = 0
            for c in CS:
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=4000))
                oof = cross_val_predict(pipe, x[tr], lab[tr], cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                        method="predict_proba")[:, 1]
                best = max(best, roc_auc_score(lab[tr], oof))
            aucs.append(best)
        scores[enc] = round(float(np.mean(aucs)), 3)
        print(f"encoder {enc}: mean CV AUROC {scores[enc]} {[round(a, 3) for a in aucs]}", flush=True)
    return max(scores, key=scores.get), scores


def main(prep: Path, view: str = "whole", segmenter: Path | None = None, calibrate: bool = True,
         encoder: str | None = None) -> None:
    rows = list(csv.DictReader(open(prep / "labels.csv", encoding="utf-8")))
    if encoder:    # reuse an encoder choice already made by CV (scores carried over from the previous metrics file)
        prev = ROOT / "models" / "wound_model_metrics.json"
        encoder_scores = json.loads(prev.read_text(encoding="utf-8")).get("encoder_cv_auroc", {}) if prev.exists() else {}
    else:
        encoder, encoder_scores = choose_encoder(prep, rows)
    print("chosen encoder:", encoder, flush=True)
    x = features(prep, rows, encoder)
    view_meta = {"kind": "whole"}
    if view == "whole+crop":       # step 4: whole-photo features + features of the crop around the wound
        boxes = crops(prep, rows, segmenter)
        x = np.concatenate([x, crop_features(prep, rows, boxes)], 1)
        view_meta = {"kind": "whole+crop", "segmenter": segmenter.resolve().relative_to(ROOT / "models").as_posix(),
                     "margin": wm.CROP_MARGIN, "min_side_frac": wm.CROP_MIN_SIDE,
                     "crop_found_pct": round(float(np.mean([b["found"] for b in boxes]) * 100), 1)}
    split = np.array([r["split"] for r in rows])
    heads, metrics = {}, {}
    for name, (col, pos, neg) in POS.items():
        lab = np.array([1 if r[col] in pos else 0 if r[col] in neg else -1 for r in rows])
        tr = (split != "test") & (lab >= 0)
        te = (split == "test") & (lab >= 0)
        y_tr, y_te = lab[tr], lab[te]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        best_c, best_auc = None, -1
        for c in CS:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=4000))
            oof = cross_val_predict(pipe, x[tr], y_tr, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y_tr, oof)
            if auc > best_auc:
                best_c, best_auc, best_oof = c, auc, oof
        # calibration (step 4): Platt scaling fitted on out-of-fold LOGITS of the chosen C, never on the test split.
        # The operating point stays chosen on the RAW scale and is then mapped through the (monotone) calibration,
        # so every yes / no / uncertain decision is unchanged; only the probability shown becomes honest (V24).
        calib_a, calib_c = 1.0, 0.0
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, class_weight="balanced", max_iter=4000))
        oof_z = cross_val_predict(pipe, x[tr], y_tr, cv=cv, method="decision_function")
        if calibrate:
            calib_a, calib_c = pc.fit_platt(oof_z, y_tr)
        to_cal = lambda p_raw: float(pc.apply(np.log(p_raw / (1 - p_raw)), calib_a, calib_c))  # noqa: E731
        # threshold: highest one that still reaches the target sensitivity on out-of-fold predictions
        cands = np.sort(np.unique(best_oof))[::-1]
        thr = next((t for t in cands if (best_oof[y_tr == 1] >= t).mean() >= TARGET_SENSITIVITY), 0.5)
        band = 0.05
        scaler = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=4000).fit(scaler.transform(x[tr]), y_tr)
        z_te = clf.decision_function(scaler.transform(x[te]))
        p_te = pc.apply(z_te, 1.0, 0.0)                    # raw scale: decisions and metrics as before
        p_te_cal = pc.apply(z_te, calib_a, calib_c)       # calibrated scale: what is shown
        test = {"n": int(te.sum()), "positives": int(y_te.sum()), "auroc": round(float(roc_auc_score(y_te, p_te)), 3),
                "auroc_95ci": bootstrap_auc(y_te, p_te), **at_threshold(y_te, p_te, thr, band),
                "no_abstention": at_threshold(y_te, p_te, thr, 0.0),
                "majority_baseline_balanced_accuracy": 0.5,
                "majority_baseline_accuracy": round(float(max(y_te.mean(), 1 - y_te.mean())), 3)}
        m = {"label_column": col, "positive": sorted(pos), "negative": sorted(neg), "train_n": int(tr.sum()),
             "train_positives": int(y_tr.sum()), "C": best_c, "cv_auroc": round(float(best_auc), 3),
             "threshold": round(float(thr), 4), "band": band,
             "cv_at_threshold": at_threshold(y_tr, best_oof, thr, band), "test": test,
             "calibration": {"method": "platt" if calibrate else "none", "a": round(calib_a, 4), "c": round(calib_c, 4),
                             "cv_ece_raw": round(pc.ece(pc.apply(oof_z, 1.0, 0.0), y_tr), 4),
                             "cv_ece": round(pc.ece(pc.apply(oof_z, calib_a, calib_c), y_tr), 4),
                             "test_ece_raw": round(pc.ece(p_te, y_te), 4), "test_ece": round(pc.ece(p_te_cal, y_te), 4),
                             "test_brier_raw": round(pc.brier(p_te, y_te), 4),
                             "test_brier": round(pc.brier(p_te_cal, y_te), 4),
                             "reliability_test": pc.reliability(p_te_cal, y_te, bins=5),
                             "threshold_calibrated": round(to_cal(thr), 4),
                             "band_calibrated": [round(to_cal(max(thr - band, 1e-6)), 4),
                                                 round(to_cal(min(thr + band, 1 - 1e-6)), 4)]}}
        if name == "erythema":
            red = opencv_redness(prep, [r for r, k in zip(rows, te) if k])
            ok = ~np.isnan(red)
            m["test"]["opencv_redness_baseline_auroc"] = round(float(roc_auc_score(y_te[ok], red[ok])), 3)
            m["test"]["opencv_redness_baseline_coverage_pct"] = round(float(ok.mean() * 100), 1)
        heads[name] = {"model": clf, "scaler": scaler, "threshold": to_cal(thr), "band": band, "metrics": m,
                       "calib_a": calib_a, "calib_c": calib_c,
                       "band_low": to_cal(max(thr - band, 1e-6)), "band_high": to_cal(min(thr + band, 1 - 1e-6))}
        metrics[name] = m
        print(f"{name:14s} CV AUROC {best_auc:.3f} | TEST AUROC {test['auroc']} {test['auroc_95ci']} "
              f"sens {test['sensitivity']} spec {test['specificity']} abstain {test['abstained_pct']}% (n={test['n']}, +{test['positives']})",
              flush=True)
    # Out-of-distribution gate: distance to the 5 nearest training photos (cosine, encoder space). A new photo farther
    # than 97.5 % of the training photos are from their own neighbours is "unfamiliar": the model abstains on it.
    trv = x[split != "test"]
    unit = trv / np.linalg.norm(trv, axis=1, keepdims=True)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -1)
    knn = 1 - np.sort(sims, axis=1)[:, -OOD_K:].mean(1)
    ood_thr = float(np.percentile(knn, 97.5))
    te_unit = x[split == "test"] / np.linalg.norm(x[split == "test"], axis=1, keepdims=True)
    te_knn = 1 - np.sort(te_unit @ unit.T, axis=1)[:, -OOD_K:].mean(1)
    ood = {"k": OOD_K, "threshold": ood_thr, "reference": unit.astype(np.float16),
           "test_flagged_pct": round(float((te_knn > ood_thr).mean() * 100), 1)}
    print(f"OOD gate: threshold {ood_thr:.4f}, test photos flagged {ood['test_flagged_pct']}%", flush=True)
    bundle = {"version": "wound-attr-0.4" if view != "whole" or calibrate else "wound-attr-0.3", "view": view_meta, "encoder": encoder, "encoder_cv_auroc": encoder_scores, "ood": ood, "trained_at": datetime.now(timezone.utc).isoformat(),
              "dataset": "SurgWound (Hugging Face Sonulee/SurgWound, CC BY-SA 4.0; surgeon labels, OSU Wexner)",
              "heads": heads}
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    import joblib
    joblib.dump(bundle, out / "wound_heads.joblib", compress=3)
    public = {k: v for k, v in bundle.items() if k not in ("heads", "ood")} | {"ood": {k: v for k, v in ood.items() if k != "reference"}}
    (out / "wound_model_metrics.json").write_text(json.dumps(public | {"heads": metrics},
                                                             indent=1), encoding="utf-8")
    from surgiguard_d1.wound.model_bundle import ModelBundle  # portable copy: inference never depends on the pickle
    manifest = ModelBundle.from_sklearn_dict(bundle).save(out / "wound_heads.json")
    print("saved", out / "wound_heads.joblib", "and", manifest)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prep", type=Path)
    ap.add_argument("--view", choices=["whole", "whole+crop"], default="whole+crop",
                    help="features of the whole photo, or whole + crop around the wound (step 4, chosen by CV)")
    ap.add_argument("--segmenter", type=Path, default=ROOT / "models" / "wound_seg" / "unet_v2.onnx")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--encoder", help="skip the encoder choice and use this one (chosen earlier by CV)")
    a = ap.parse_args()
    main(a.prep, a.view, a.segmenter, not a.no_calibration, a.encoder)
