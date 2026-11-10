"""Step 4 experiment: does classifying a crop around the wound beat the whole photo? Are the probabilities calibrated?

    python scripts/wound_view_experiment.py <prepared surgwound folder> --segmenter models/wound_seg/unet_v2.onnx

Pre-registered protocol (decided before looking at any test number):
1. Views: "whole" (the photo, as today), "crop" (WoundCropper + segmenter v2), "both" (the two feature vectors
   concatenated). The segmenter was trained on SurgWound TRAIN photos only, so no test photo was ever seen.
2. Per view and attribute: the same 5-fold CV and C grid as train_wound_model.py, on train+val only.
3. The view used from now on = the best MEAN CV AUROC over the six attributes. The test split is then scored for
   every view (so nothing is hidden), with bootstrap CIs and a paired bootstrap of (chosen - whole).
4. Calibration of the chosen view: none vs temperature vs Platt, compared by a second-level CV on the out-of-fold
   logits (fit on 4/5, measure on 1/5); the winner is fitted on all out-of-fold logits and applied once to test.
Writes models/wound_view_experiment.json and caches crops/features in the prepared folder.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_wound_model as twm  # noqa: E402  (labels, crops, features: one implementation)
from surgiguard_d1.wound import prob_calibration as pc  # noqa: E402

RNG = np.random.default_rng(0)


def view_features(prep: Path, rows: list[dict], boxes: list[dict], view: str) -> np.ndarray:
    return twm.features(prep, rows) if view == "whole" else twm.crop_features(prep, rows, boxes)


def labels(rows, col, pos, neg):
    return np.array([1 if r[col] in pos else 0 if r[col] in neg else -1 for r in rows])


def best_cv(x, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    best = (-1, None, None)
    for c in twm.CS:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=4000))
        z = cross_val_predict(pipe, x, y, cv=cv, method="decision_function")
        auc = roc_auc_score(y, z)
        if auc > best[0]:
            best = (auc, c, z)
    return best


def fit_predict(x_tr, y_tr, x_te, c):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=4000))
    return pipe.fit(x_tr, y_tr).decision_function(x_te)


def paired_auc_diff(y, za, zb, n=2000):
    d = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(roc_auc_score(y[i], za[i]) - roc_auc_score(y[i], zb[i]))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def calibration_cv(z, y):
    """Second-level CV on out-of-fold logits: ECE / Brier of none, temperature, Platt on held-out folds."""
    res = {k: {"ece": [], "brier": []} for k in ("none", "temperature", "platt")}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(z, y):
        for k, (a, c) in {"none": (1.0, 0.0), "temperature": pc.fit_platt(z[tr], y[tr], with_bias=False),
                          "platt": pc.fit_platt(z[tr], y[tr])}.items():
            p = pc.apply(z[te], a, c)
            res[k]["ece"].append(pc.ece(p, y[te]))
            res[k]["brier"].append(pc.brier(p, y[te]))
    return {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in res.items()}


def main(a) -> None:
    prep = Path(a.prep)
    rows = list(csv.DictReader(open(prep / "labels.csv", encoding="utf-8")))
    split = np.array([r["split"] for r in rows])
    boxes = twm.crops(prep, rows, Path(a.segmenter))
    found = np.array([b["found"] for b in boxes])
    print(f"wound found (crop used) on {found.mean():.1%} of photos; test split {found[split == 'test'].mean():.1%}")
    X = {"whole": view_features(prep, rows, boxes, "whole"), "crop": view_features(prep, rows, boxes, "crop")}
    X["both"] = np.concatenate([X["whole"], X["crop"]], 1)
    cv_table, store = {}, {}
    for name, (col, pos, neg) in twm.POS.items():
        lab = labels(rows, col, pos, neg)
        tr, te = (split != "test") & (lab >= 0), (split == "test") & (lab >= 0)
        for view, x in X.items():
            auc, c, oof = best_cv(x[tr], lab[tr])
            z_te = fit_predict(x[tr], lab[tr], x[te], c)
            store[(name, view)] = dict(c=c, oof=oof, y_tr=lab[tr], z_te=z_te, y_te=lab[te])
            cv_table.setdefault(name, {})[view] = round(float(auc), 3)
        print(name, cv_table[name], flush=True)
    mean_cv = {v: round(float(np.mean([cv_table[n][v] for n in cv_table])), 3) for v in X}
    chosen = max(mean_cv, key=mean_cv.get)
    print("mean CV AUROC", mean_cv, "-> chosen view:", chosen, flush=True)
    result = {"protocol": __doc__.split("Pre-registered protocol")[1].strip(), "segmenter": a.segmenter,
              "crop_found_pct": round(float(found.mean() * 100), 1),
              "crop_found_test_pct": round(float(found[split == 'test'].mean() * 100), 1),
              "cv_auroc": cv_table, "mean_cv_auroc": mean_cv, "chosen_view": chosen, "test": {}, "calibration": {}}
    for name in twm.POS:
        t = {}
        for view in X:
            s = store[(name, view)]
            t[view] = {"auroc": round(float(roc_auc_score(s["y_te"], s["z_te"])), 3),
                       "ci": twm.bootstrap_auc(s["y_te"], s["z_te"])}
        if chosen != "whole":
            s, w = store[(name, chosen)], store[(name, "whole")]
            t[f"{chosen}_minus_whole_95ci"] = paired_auc_diff(s["y_te"], s["z_te"], w["z_te"])
        result["test"][name] = t
        s = store[(name, chosen)]
        cvcal = calibration_cv(s["oof"], s["y_tr"])
        best = min(("temperature", "platt"), key=lambda k: cvcal[k]["ece"])
        a_, c_ = pc.fit_platt(s["oof"], s["y_tr"], with_bias=(best == "platt"))
        p_raw, p_cal = pc.apply(s["z_te"], 1.0, 0.0), pc.apply(s["z_te"], a_, c_)
        result["calibration"][name] = {
            "cv": cvcal, "chosen": best, "a": round(a_, 4), "c": round(c_, 4),
            "test_raw": {"ece": round(pc.ece(p_raw, s["y_te"]), 4), "brier": round(pc.brier(p_raw, s["y_te"]), 4)},
            "test_calibrated": {"ece": round(pc.ece(p_cal, s["y_te"]), 4), "brier": round(pc.brier(p_cal, s["y_te"]), 4)},
            "test_positive_rate": round(float(s["y_te"].mean()), 3),
            "reliability_test_calibrated": pc.reliability(p_cal, s["y_te"], bins=5)}
        print(f"{name:14s} test AUROC " + " ".join(f"{v} {t[v]['auroc']}" for v in X) +
              f" | ECE raw {result['calibration'][name]['test_raw']['ece']} -> {best} "
              f"{result['calibration'][name]['test_calibrated']['ece']}", flush=True)
    out = ROOT / "models" / "wound_view_experiment.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prep")
    ap.add_argument("--segmenter", default=str(ROOT / "models" / "wound_seg" / "unet_v2.onnx"))
    main(ap.parse_args())
