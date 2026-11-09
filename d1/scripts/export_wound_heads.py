"""Convert the legacy scikit-learn pickle into the portable bundle (JSON manifest + NPZ arrays), and prove it.

    python scripts/export_wound_heads.py [models/wound_heads.joblib] [models/wound_heads.json]

Needs the scikit-learn version that wrote the pickle (1.9.1) ONCE, to read it. After that, inference needs numpy only.
The script checks that the portable heads give the same probabilities as scikit-learn (max |diff| < 1e-9) on the
OOD reference features (real training photos) and refuses to write anything if they do not.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surgiguard_d1.wound.model_bundle import ModelBundle, load  # noqa: E402


def main(src: Path, dst: Path) -> None:
    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # an InconsistentVersionWarning must stop the export, not pass silently
        legacy = joblib.load(src)
    bundle = ModelBundle.from_sklearn_dict(legacy)
    x = np.asarray(legacy["ood"]["reference"], np.float64)      # real feature vectors from the training photos
    worst = 0.0
    for name, h in legacy["heads"].items():
        p_sklearn = h["model"].predict_proba(h["scaler"].transform(x))[:, 1]
        p_numpy = bundle.heads[name].probability(x)
        worst = max(worst, float(np.abs(p_sklearn - p_numpy).max()))
    if worst > 1e-9:
        raise SystemExit(f"portable heads differ from scikit-learn (max |diff| {worst:.2e}): nothing written")
    manifest = bundle.save(dst)
    again = load(manifest)                                       # round trip through the files
    for name in bundle.heads:
        assert np.array_equal(again.heads[name].probability(x[:20]), bundle.heads[name].probability(x[:20]))
    size = manifest.stat().st_size + manifest.with_suffix(".npz").stat().st_size
    print(f"wrote {manifest.name} + {manifest.with_suffix('.npz').name} ({size / 1024:.0f} KB); "
          f"max |p_numpy - p_sklearn| = {worst:.1e} over {len(x)} photos x {len(bundle.heads)} heads")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]) if a else ROOT / "models" / "wound_heads.joblib",
         Path(a[1]) if len(a) > 1 else ROOT / "models" / "wound_heads.json")
