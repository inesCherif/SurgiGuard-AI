"""D1-F learned model: surgical-wound attributes from a photo (SurgWound surgeon labels).

Design (small expert-labelled dataset, CPU):
- Encoder: openai/clip-vit-base-patch16 (ViT-B/16), FROZEN. Chosen over facebook/dinov2-small by cross-validation
  on the training split only (mean CV AUROC 0.817 vs 0.75). Features = [CLS token, mean of patch tokens] (1536
  numbers), averaged with the mirrored photo (test-time augmentation).
- Heads: one standardised, class-balanced logistic regression per attribute, stored in the portable bundle
  (models/wound_heads.json + .npz, see model_bundle.py) and evaluated with numpy only.
- Operating point: threshold chosen for sensitivity >= 0.85 on out-of-fold predictions (missing a sign is worse than
  a false alarm), and an "uncertain" band around it where the model abstains.
- Unfamiliar-photo gate: too far from the training photos -> every attribute is "uncertain".
- Explanation: occlusion sensitivity on a 5x5 grid (grey out a cell, measure the score drop). Opt-in: ~25 passes.
- Step 4: the bundle declares its feature view (meta["view"]). "whole+crop" = whole-photo features + features of a
  square crop around the wound found by the learned segmenter (WoundCropper), chosen by cross-validation; the heads
  also carry Platt calibration so a probability means what it says (prob_calibration.py).

The output describes visible attributes as surgeons labelled them. Decision support for the care team, never a
diagnosis; it can only ADD a reason for review, never remove one.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import model_bundle
from .model_bundle import ModelBundle
from .render import encode_jpeg

ENCODER = "openai/clip-vit-base-patch16"
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
HEADS_FILE = MODELS_DIR / "wound_heads.json"          # portable bundle (numpy only)
LEGACY_HEADS_FILE = MODELS_DIR / "wound_heads.joblib"  # v1 pickle, read only through the adapter
SIZE = 224
EVIDENCE_GRID = 5
CROP_MARGIN = 0.35        # context kept around the wound, x its long side (peri-wound skin is where redness is judged)
CROP_MIN_SIDE = 0.25      # minimum crop side, fraction of the photo's short side

# What each head says, in plain words (positive class first).
HEADS = {
    "erythema": ("redness around the wound (erythema)", "no clear redness"),
    "edema": ("swelling around the wound (edema)", "no clear swelling"),
    "exudate": ("visible fluid or discharge (exudate)", "no visible discharge"),
    "not_healed": ("wound not yet healed", "wound looks healed"),
    "elevated_risk": ("surgeons would rate infection risk medium/high", "surgeons would rate infection risk low"),
    "needs_review": ("surgeons would want a professional review (clinic or emergency)", "surgeons would suggest routine home care"),
}


class Encoder:
    """A frozen image encoder: hides the processor differences between model families (CLIP vs DINOv2)."""

    def __init__(self, name: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
        torch.set_num_threads(min(4, torch.get_num_threads()))   # leave CPU for the API
        self.name = name
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = (CLIPVisionModel if "clip" in name else AutoModel).from_pretrained(name).eval()
        # CLIP keeps its own resize + centre crop; DINOv2 is resized to SIZE x SIZE without cropping.
        self._kw = {} if "clip" in name else {"size": {"height": SIZE, "width": SIZE}, "do_center_crop": False}

    def tokens(self, images: list):
        """PIL images -> last hidden state (n, 1 + n_patches, d)."""
        import torch
        with torch.no_grad():
            return self.model(**self.processor(images=images, return_tensors="pt", **self._kw)).last_hidden_state

    @staticmethod
    def pool(h) -> np.ndarray:
        """[CLS token, mean of patch tokens] per image."""
        import torch
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).numpy()


@lru_cache(maxsize=2)
def get_encoder(name: str = ENCODER) -> Encoder:
    return Encoder(name)


def embed(images: list, with_patches: bool = False, encoder: str | None = None):
    """PIL images -> features (n, 2*d) averaged with the mirror image; optionally the unmirrored patch tokens."""
    from PIL import ImageOps
    enc = get_encoder(encoder or ENCODER)
    feats, patches = [], []
    for im in images:
        im = im.convert("RGB")
        h = enc.tokens([im, ImageOps.mirror(im)])            # (2, 1 + n_patches, d)
        feats.append(Encoder.pool(h).mean(0))
        if with_patches:
            patches.append(h[0, 1:].numpy())
    feats = np.stack(feats)
    return (feats, patches) if with_patches else feats


@lru_cache(maxsize=2)
def get_cropper(segmenter_path: str, margin: float = CROP_MARGIN, min_side_frac: float = CROP_MIN_SIDE):
    """One WoundCropper (+ its ONNX session) per segmenter file, created once."""
    from .crop import WoundCropper
    from .learned import LearnedSegmenter
    return WoundCropper(LearnedSegmenter(segmenter_path), margin, min_side_frac)


def _view_cropper(bundle: ModelBundle):
    v = bundle.meta.get("view") or {"kind": "whole"}
    if v.get("kind") != "whole+crop":
        return None
    return get_cropper(str(MODELS_DIR / v["segmenter"]), v.get("margin", CROP_MARGIN), v.get("min_side_frac", CROP_MIN_SIDE))


def photo_features(im, bundle: ModelBundle):
    """Features in the layout the heads were trained on, plus the crop used (or None for the whole-photo view).
    Strategy chosen by the bundle, so an older whole-photo bundle keeps working unchanged."""
    x = embed([im], encoder=bundle.encoder)
    cropper = _view_cropper(bundle)
    if cropper is None:
        return x, None
    crop = cropper.crop(np.asarray(im)[..., ::-1])        # PIL RGB -> BGR for the segmenter
    x = np.concatenate([x, embed([im.crop(crop.box)], encoder=bundle.encoder)], 1)
    return x, crop


def heads_file() -> Path:
    """The portable bundle if present, otherwise the legacy pickle."""
    return HEADS_FILE if HEADS_FILE.exists() else LEGACY_HEADS_FILE


@lru_cache(maxsize=1)
def load_heads(path: str | None = None) -> ModelBundle:
    return model_bundle.load(path or heads_file())


def available() -> tuple[bool, str | None]:
    if not heads_file().exists():
        return False, f"{HEADS_FILE.name} missing: run scripts/train_wound_model.py then scripts/export_wound_heads.py"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"{e.name} not installed (pip install torch transformers)"
    return True, None


def warm_up() -> None:
    """Load the heads and the encoder once at server start, so the first photo does not pay for it."""
    get_encoder(load_heads().encoder or ENCODER)


def _open(data: bytes):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def predict(data: bytes, heatmap_for: str | None = "erythema") -> dict:
    """Photo bytes -> per-attribute probability, decision (yes / no / uncertain) and an optional evidence map."""
    bundle = load_heads()
    im = _open(data)
    x, crop = photo_features(im, bundle)
    familiar, distance = (True, None) if bundle.ood is None else bundle.ood.familiar(x[0])
    out = {}
    for name, head in bundle.heads.items():
        p = float(head.probability(x)[0])
        decision = head.decide(p, familiar)
        pos, neg = HEADS[name]
        test = head.metrics.get("test", {})
        out[name] = {"probability": round(p, 3), "threshold": round(head.threshold, 3), "decision": decision,
                     "text": pos if decision == "yes" else neg if decision == "no" else f"uncertain: {pos}?",
                     "test_auroc": test.get("auroc"), "test_sensitivity": test.get("sensitivity"),
                     "test_specificity": test.get("specificity")}
    result = {"in_distribution": familiar, "ood_distance": None if distance is None else round(distance, 4),
              "ood_threshold": None if bundle.ood is None else round(bundle.ood.threshold, 4),
              "version": bundle.version, "encoder": bundle.encoder or ENCODER, "trained_on": bundle.dataset,
              "attributes": out,
              "view": (bundle.meta.get("view") or {"kind": "whole"}).get("kind"),
              "crop": None if crop is None else {"box": list(crop.box), "wound_found": crop.found}}
    if heatmap_for and heatmap_for in bundle.heads and familiar:
        result["_heatmap"] = occlusion_map(bundle.heads[heatmap_for], im, bundle.encoder,
                                           crop_box=None if crop is None else crop.box)
    return result


def occlusion_map(head, im, encoder: str | None, grid: int = EVIDENCE_GRID, crop_box=None) -> np.ndarray:
    """Grey out each grid cell and measure how much the head's probability DROPS (> 0: the cell supported it).
    Model-agnostic and faithful to the whole model, unlike raw ViT patch attributions (which highlight background).
    With a crop view, the cells are greyed on the full photo and BOTH feature halves are recomputed (whole + the same
    crop box), so the map explains the real decision."""
    from PIL import ImageDraw
    enc = get_encoder(encoder or ENCODER)
    base = im.convert("RGB") if crop_box is not None else im.convert("RGB").resize((SIZE, SIZE))
    cw, ch = base.width / grid, base.height / grid
    variants = [base]
    for i in range(grid):
        for j in range(grid):
            v = base.copy()
            ImageDraw.Draw(v).rectangle([j * cw, i * ch, (j + 1) * cw, (i + 1) * ch], fill=(128, 128, 128))
            variants.append(v)

    def pooled(images):
        return np.concatenate([Encoder.pool(enc.tokens(images[k:k + 2])) for k in range(0, len(images), 2)])  # small batches
    feats = pooled(variants)
    if crop_box is not None:
        feats = np.concatenate([feats, pooled([v.crop(tuple(crop_box)) for v in variants])], 1)
    p = head.probability(feats)
    return (p[0] - p[1:]).reshape(grid, grid)


def heatmap_overlay(data: bytes, grid: np.ndarray) -> bytes:
    """JPEG: the photo with the positive-evidence map on top (red = where the evidence came from)."""
    import cv2
    im = np.asarray(_open(data))
    scale = 512 / max(im.shape[:2])
    im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
    g = np.clip(grid, 0, None)
    g = g / (g.max() + 1e-9)
    g = cv2.resize(g.astype(np.float32), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    g = np.clip((g - 0.35) / 0.65, 0, 1)   # colour only the regions that mattered most
    color = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    out = (im * (1 - 0.45 * g[..., None]) + color * 0.45 * g[..., None]).astype(np.uint8)
    return encode_jpeg(out[..., ::-1])
