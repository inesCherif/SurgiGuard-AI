"""Train the learned wound segmenter, export it to ONNX and benchmark it against the classical one.

    <training venv>/python scripts/train_segmenter.py --train <root> [<root> ...] --test <root> --out models/wound_seg/unet

Each <root> is a MaskFolder (<root>/images, <root>/labels). `--train` roots are split 90/10 into train and
validation (validation picks the best epoch); `--test` is never seen during training and is scored once at the
end with SegmenterBenchmark, for BOTH the learned and the classical segmenter (same code, same images).
Writes <out>.onnx, <out>.json (model card with scores), <out>.pt (best weights), <out>_benchmark.json and
<out>_last.ckpt (full state after every epoch; `--resume` continues from it after a crash).
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surgiguard_d1.wound import ClassicalSegmenter  # noqa: E402
from surgiguard_d1.wound.learned import LearnedSegmenter  # noqa: E402
from surgiguard_d1.wound.seg_data import MaskFolder  # noqa: E402
from surgiguard_d1.wound.seg_eval import SegmenterBenchmark  # noqa: E402
from surgiguard_d1.wound.training.data import SegDataset, prepare, split  # noqa: E402
from surgiguard_d1.wound.training.unet import ResNetUNet, dice_bce_loss  # noqa: E402


def _val_dice(model, loader, device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            p = (torch.sigmoid(model(x.to(device))) > 0.5).float()
            y = y.to(device)
            inter = (p * y).sum(dim=(1, 2, 3))
            denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
            d = torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))
            scores += d.tolist()
    return float(np.mean(scores))


def train(args) -> Path:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Split each root separately, THEN oversample its training part (weight w = repeat w times): a photo can never
    # be in both train and validation, and a small root (our 120 incisions) still weighs in the training mix.
    weights = args.weights or [1] * len(args.train)
    samples, tr, va = [], [], []
    for root, w in zip(args.train, weights):
        own = list(MaskFolder(Path(root)))
        t, v = split(own, 0.1, args.seed)
        samples += own
        tr += t * int(w)
        va += v
    print(f"{len(tr)} train items / {len(va)} val images (weights {weights}), device {device}")
    tl = DataLoader(SegDataset(prepare(tr, args.size), True, args.seed), batch_size=args.batch, shuffle=True,
                    drop_last=True)
    vl = DataLoader(SegDataset(prepare(va, args.size), False), batch_size=args.batch)
    model = ResNetUNet(pretrained=args.init is None).to(device)
    if args.init:      # fine-tuning: start from an already trained model (e.g. the FUSeg one) instead of ImageNet
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
        print("initialised from", args.init)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history, start = -1.0, -1, [], 0
    ckpt = Path(str(out) + "_last.ckpt")
    if args.resume and ckpt.is_file():
        # Full training state, not only the weights: optimiser moments, LR schedule position, AMP scale and RNG,
        # so a resumed run continues exactly where it stopped (journal V17).
        st = torch.load(ckpt, map_location=device, weights_only=False)
        for obj, key in ((model, "model"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")):
            obj.load_state_dict(st[key])
        best, best_epoch, history, start = st["best"], st["best_epoch"], st["history"], st["epoch"]
        torch.set_rng_state(st["torch_rng"].cpu())   # map_location moved it to the GPU (journal V21)
        tl.dataset.rng = st["aug_rng"]
        print(f"resumed after epoch {start} (best val dice {best:.4f} at epoch {best_epoch})")
    for epoch in range(start, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = dice_bce_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        vd = _val_dice(model, vl, device)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_dice": vd})
        flag = ""
        if vd > best:
            best, best_epoch, flag = vd, epoch + 1, "  *best*"
            torch.save(model.state_dict(), out.with_suffix(".pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "best_epoch": best_epoch,
                    "history": history, "torch_rng": torch.get_rng_state(), "aug_rng": tl.dataset.rng}, ckpt)
        print(f"epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}  val dice {vd:.4f}  {time.time() - t0:.0f}s{flag}",
              flush=True)
        if epoch + 1 - best_epoch >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break
    model.load_state_dict(torch.load(out.with_suffix(".pt"), map_location=device))
    _export(model.cpu().eval(), out, args, samples=len(samples), best=best, best_epoch=best_epoch, history=history)
    return out.with_suffix(".onnx")


def export_only(args) -> Path:
    """Export the best weights without loading any training photo (the export needs ~1 GB; journal V21)."""
    out = Path(args.out)
    st = torch.load(Path(str(out) + "_last.ckpt"), map_location="cpu", weights_only=False)
    model = ResNetUNet(pretrained=False)
    model.load_state_dict(torch.load(out.with_suffix(".pt"), map_location="cpu", weights_only=True))
    n = sum(len(MaskFolder(Path(r))) for r in args.train)
    _export(model.eval(), out, args, samples=n, best=st["best"], best_epoch=st["best_epoch"], history=st["history"])
    return out.with_suffix(".onnx")


def _export(model, out: Path, args, **info) -> None:
    torch.onnx.export(model, torch.zeros(1, 3, args.size, args.size), str(out.with_suffix(".onnx")),
                      input_names=["image"], output_names=["logits"], opset_version=17)
    card = {"input_size": args.size, "threshold": 0.5, "version": args.version,
            "trained_on": [f"{Path(r).parent.name}/{Path(r).name}" for r in args.train], "weights": args.weights, "init": args.init,
            "scores": {"val_dice_network_res": info["best"], "best_epoch": info["best_epoch"],
                       "n_labelled_train_val": info["samples"]},
            "architecture": "U-Net, ResNet18 encoder (ImageNet), BCE + soft Dice, AdamW + OneCycle, AMP",
            "history": info["history"]}
    out.with_suffix(".json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print("exported", out.with_suffix(".onnx"))


def benchmark(onnx_path: Path, test_roots, out: Path) -> dict:
    tests = [s for r in test_roots for s in MaskFolder(Path(r))]
    bench = SegmenterBenchmark(tests)
    results = {}
    for seg in (LearnedSegmenter(onnx_path), ClassicalSegmenter()):
        t0 = time.time()
        rep = bench.run(seg)
        results[seg.name] = {**rep.summary(), "seconds_per_image": (time.time() - t0) / len(tests),
                             "per_image": {k: asdict(v) for k, v in rep.per_image.items()}}
        print({k: v for k, v in results[seg.name].items() if k != "per_image"})
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    card_path = onnx_path.with_suffix(".json")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    learned = results[next(k for k in results if k.startswith("learned"))]
    card["scores"].update({f"test_{k}": learned[k] for k in ("dice_mean", "iou_mean", "precision_mean",
                                                            "recall_mean", "width_ratio_median", "n_images")})
    card["tested_on"] = [f"{Path(r).parent.name}/{Path(r).name}" for r in test_roots]
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", default="models/wound_seg/unet")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", nargs="+", type=int, help="oversampling factor per --train root (default 1 each)")
    ap.add_argument("--init", help="start from these weights (.pt) instead of ImageNet: fine-tuning")
    ap.add_argument("--resume", action="store_true", help="continue from <out>_last.ckpt after a crash")
    ap.add_argument("--export-only", action="store_true", help="export <out>.pt to ONNX (after training) and benchmark")
    ap.add_argument("--benchmark-only", action="store_true", help="skip training, score an existing <out>.onnx")
    a = ap.parse_args()
    if a.benchmark_only:
        onnx = Path(a.out).with_suffix(".onnx")
    elif a.export_only:
        onnx = export_only(a)
    else:
        onnx = train(a)
    benchmark(onnx, a.test, Path(str(Path(a.out)) + "_benchmark.json"))
