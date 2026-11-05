#!/usr/bin/env python3
"""CHASEDB1 Structural QAT with thin-vessel proxy weight 8.

This runner never reads the DRIVE or CHASE test manifest during training. It
freezes every group bit logit and updates only convolution weights and biases.
The training loss uses FOV background weight 1, vessel weight 2, and thin
vessel proxy weight 8. Validation metrics remain unweighted FOV metrics.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models.UNetFamily import U_Net
from lib.datasetV2 import TrainDatasetV2, create_patch_idx, data_preprocess
from tools.group_trainable_bits import replace_conv_layers_with_groups


DEFAULT_EXP = ROOT / "experiments/DRIVE_Unet_GroupBits_structural_seed2027"
DEFAULT_OUT = ROOT / "experiments/DRIVE_Unet_GroupBits_QAT_pilot"
TRAIN_MANIFEST = Path('/data/xpp/data/CHASE_DB1-main/CHASE_DB1-main/prepared_seed2026/train.txt')
VAL_MANIFEST = Path('/data/xpp/data/CHASE_DB1-main/CHASE_DB1-main/prepared_seed2026/val.txt')
TARGET_BITS = 4.269841064
GROUP_SIZE = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=str(DEFAULT_EXP))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--patches", type=int, default=320)
    p.add_argument("--calibration-patches", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=str(ROOT / "experiments/DRIVE_Unet_QAT_patch_cache_seed2027"))
    p.add_argument("--profile-log", default="")
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--thin-weight", type=float, default=8.0)
    p.add_argument("--vessel-weight", type=float, default=2.0)
    return p.parse_args()


def load_assignment(exp: Path):
    path = exp / "assignment.json"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text())
    assignment = obj.get("assignment", {})
    if len(assignment) != 857:
        raise RuntimeError(f"assignment group count {len(assignment)} != 857")
    bits = [int(v) for v in assignment.values()]
    if any(v not in (4, 8) for v in bits):
        raise RuntimeError("assignment contains a bitwidth other than 4 or 8")
    mean_bits = float(obj["parameter_weighted_mean_bitwidth"])
    if abs(mean_bits - TARGET_BITS) > 0.01:
        raise RuntimeError(f"budget mismatch: {mean_bits}")
    return obj, assignment, path


def build_model(exp: Path, assignment: dict, device):
    ckpt_path = exp / "best_checkpoint.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = U_Net(1, 2)
    wrappers = replace_conv_layers_with_groups(model, GROUP_SIZE)
    model.load_state_dict(checkpoint["net"], strict=True)
    if sum(w.num_groups for w in wrappers.values()) != 857:
        raise RuntimeError("wrapped group count is not 857")
    for name, wrapper in wrappers.items():
        for gid in range(wrapper.num_groups):
            key = f"{name}:{gid}"
            if key not in assignment:
                raise RuntimeError(f"missing assignment key {key}")
            # +/- 20 makes the sigmoid effectively discrete while retaining the
            # exact existing fake-quant branch implementation.
            wrapper.bit_logit.data[gid] = 20.0 if int(assignment[key]) == 8 else -20.0
            wrapper.bit_logit.requires_grad_(False)
        wrapper.weight.requires_grad_(True)
        if wrapper.bias is not None:
            wrapper.bias.requires_grad_(True)
    model.to(device)
    return model, wrappers, checkpoint


def make_loader(manifest: Path, patches: int, batch_size: int, seed: int, shuffle: bool,
                cache_dir: Path, split: str, log):
    t0 = time.time(); log(f"[{split}] reading manifest: {manifest}")
    images, masks, fovs = data_preprocess(str(manifest))
    log(f"[{split}] manifest/data read complete in {time.time()-t0:.3f}s")
    args = argparse.Namespace(train_patch_height=64, train_patch_width=64,
                              N_patches=patches, inside_FOV="center")
    expected = {"manifest_sha256": sha256(manifest), "seed": seed, "patch_size": [64, 64],
                "patches": patches, "inside_FOV": "center", "split": split}
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{split}_indices.meta.json"
    index_path = cache_dir / f"{split}_indices.npy"
    if meta_path.exists() or index_path.exists():
        if not (meta_path.exists() and index_path.exists()):
            raise RuntimeError(f"incomplete {split} cache; refusing to overwrite")
        cached = json.loads(meta_path.read_text())
        if cached != expected:
            raise RuntimeError(f"{split} cache metadata mismatch; refusing to overwrite: {cached} != {expected}")
        indices = np.load(index_path, allow_pickle=True).tolist()
        log(f"[{split}] reused patch cache with {len(indices)} indices")
    else:
        t1 = time.time(); log(f"[{split}] generating {patches} patch indices")
        indices = create_patch_idx(fovs, args, seed=seed)
        np.save(index_path, np.asarray(indices, dtype=object), allow_pickle=True)
        meta_path.write_text(json.dumps(expected, indent=2))
        log(f"[{split}] patch index generation/cache write: {time.time()-t1:.3f}s")
    t2 = time.time()
    ds = TrainDatasetV2(images, masks, fovs, indices, mode="val", args=args)
    log(f"[{split}] Dataset init: {time.time()-t2:.3f}s")
    t3 = time.time(); loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    log(f"[{split}] DataLoader init: {time.time()-t3:.3f}s; batches={len(loader)}")
    return loader


def batch_loss(logits, targets, fovs, thin_weight=1.0, vessel_weight=1.0):
    valid = fovs > 0
    if not bool(valid.any()):
        raise RuntimeError("batch has no FOV pixels")
    pixel_ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    vessel = targets == 1
    density = torch.nn.functional.avg_pool2d(
        vessel.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)
    thin = vessel & valid & (density <= (4.0 / 9.0))
    weights = torch.ones_like(pixel_ce)
    weights = torch.where(vessel & valid, torch.as_tensor(vessel_weight, device=weights.device), weights)
    weights = torch.where(thin, torch.as_tensor(thin_weight, device=weights.device), weights)
    return (pixel_ce * weights * valid.float()).sum() / weights[valid].sum().clamp_min(1.0)


def evaluate(model, loader, device):
    model.eval()
    losses, ys, scores = [], [], []
    with torch.no_grad():
        for x, y, fov in loader:
            x, y, fov = x.to(device), y.to(device), fov.to(device)
            logits = model(x)
            if tuple(logits.shape[1:]) != (2, 64, 64):
                raise RuntimeError(f"unexpected output shape {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise RuntimeError("NaN/Inf in validation output")
            losses.append(float(batch_loss(logits, y, fov).item()))
            prob = torch.softmax(logits, dim=1)[:, 1]
            valid = fov > 0
            ys.append((y[valid] == 1).cpu().numpy().astype(np.uint8))
            scores.append(prob[valid].cpu().numpy())
    y = np.concatenate(ys); s = np.concatenate(scores)
    pred = s >= 0.5
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    if not math.isfinite(auc) or not math.isfinite(f1):
        raise RuntimeError("non-finite validation metric")
    return float(np.mean(losses)), f1, auc


def budget_stats(wrappers):
    total_params = 0
    total_bits = 0.0
    n4 = n8 = 0
    for w in wrappers.values():
        per_ch = w.weight.shape[1] * w.weight.shape[2] * w.weight.shape[3]
        for gid, (start, end) in enumerate(w.group_slices):
            b = int(w.discrete_bits()[gid].item())
            n = (end - start) * per_ch
            total_params += n; total_bits += n * b
            n4 += b == 4; n8 += b == 8
    mean = total_bits / total_params
    return {"int4_groups": int(n4), "int8_groups": int(n8),
            "group_count": int(n4 + n8),
            "parameter_weighted_mean_bitwidth": mean,
            "target_weighted_mean_bitwidth": TARGET_BITS,
            "absolute_budget_error": abs(mean - TARGET_BITS)}


def main():
    args = parse_args()
    out = Path(args.output_dir)
    exp = Path(args.experiment)
    if out.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out}")
    if not (exp / "best_checkpoint.pt").exists():
        raise FileNotFoundError(exp / "best_checkpoint.pt")
    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        raise FileNotFoundError("train/validation manifest missing")
    assignment_obj, assignment, assignment_path = load_assignment(exp)
    out.mkdir(parents=False)
    log_path = Path(args.profile_log) if args.profile_log else out / "profile.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    log("preflight assignment/checkpoint/manifest checks passed")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cache_dir = Path(args.cache_dir)
    prep_start = time.time()
    train_loader = make_loader(TRAIN_MANIFEST, args.patches, args.batch_size, args.seed, True, cache_dir, "train", log)
    val_loader = make_loader(VAL_MANIFEST, args.batch_size, args.batch_size, args.seed + 1, False, cache_dir, "validation", log)
    log(f"all data preparation complete: {time.time()-prep_start:.3f}s")
    model, wrappers, init_ckpt = build_model(exp, assignment, device)
    initial_budget = budget_stats(wrappers)
    if initial_budget["group_count"] != 857 or initial_budget["absolute_budget_error"] > 0.01:
        raise RuntimeError("fixed assignment budget audit failed before training: " + json.dumps(initial_budget))
    before_weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith(".weight")}
    before_logits = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith("bit_logit")}
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(p.requires_grad for n, p in model.named_parameters() if n.endswith("bit_logit")):
        raise RuntimeError("incorrect QAT parameter freezing")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    (out / "assignment_copy.json").write_text(json.dumps(assignment_obj, indent=2))
    for src, name in [(TRAIN_MANIFEST, "train_manifest.txt"), (VAL_MANIFEST, "validation_manifest.txt")]:
        (out / name).write_text(src.read_text()); (out / f"{name}.sha256").write_text(sha256(src) + "  " + name + "\n")
    config = {"method": "structural_qat_thin8", "seed": args.seed, "epochs": args.epochs,
              "patches": args.patches, "calibration_patches": args.calibration_patches,
              "batch_size": args.batch_size, "learning_rate": args.lr,
              "early_stopping_patience": args.early_stopping_patience,
              "group_size": GROUP_SIZE, "group_count": 857, "fake_quant": True,
              "loss": "FOV weighted pixel CE", "background_weight": 1.0,
              "vessel_weight": args.vessel_weight, "thin_vessel_weight": args.thin_weight,
              "freeze_bit_logits": True, "test_used": False, "official_test_run": False,
              "source_experiment": str(exp), "source_checkpoint": str(exp / "best_checkpoint.pt"),
              "source_assignment": str(assignment_path), "train_manifest_sha256": sha256(TRAIN_MANIFEST),
              "validation_manifest_sha256": sha256(VAL_MANIFEST), "budget": budget_stats(wrappers)}
    (out / "qat_config.json").write_text(json.dumps(config, indent=2))
    rows = []; best = None; t0 = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    epoch_timings = []
    no_improve = 0; stop_reason = "reached_max_epochs"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time(); log(f"starting epoch {epoch}/{args.epochs}")
        model.train(); train_losses = []
        first_batch_elapsed = None
        for batch_index, (x, y, fov) in enumerate(train_loader, start=1):
            batch_start = time.time()
            x, y, fov = x.to(device), y.to(device), fov.to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(x)
            if tuple(logits.shape[1:]) != (2, 64, 64): raise RuntimeError(f"shape {tuple(logits.shape)}")
            if not torch.isfinite(logits).all(): raise RuntimeError("NaN/Inf in train output")
            loss = batch_loss(logits, y, fov, args.thin_weight, args.vessel_weight)
            loss.backward(); optimizer.step(); train_losses.append(float(loss.item()))
            if first_batch_elapsed is None:
                first_batch_elapsed = time.time() - batch_start
                log(f"first batch complete: {first_batch_elapsed:.3f}s; shape={tuple(logits.shape)}")
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                log(f"epoch {epoch} batch {batch_index}/{len(train_loader)} loss={float(loss.item()):.6f} batch_s={time.time()-batch_start:.3f}")
        val_loss, val_f1, val_auc = evaluate(model, val_loader, device)
        epoch_elapsed = time.time() - epoch_start
        epoch_timings.append({"epoch": epoch, "epoch_seconds": epoch_elapsed,
                              "first_batch_seconds": first_batch_elapsed,
                              "batches_per_second": len(train_loader) / max(epoch_elapsed, 1e-9)})
        log(f"epoch {epoch} complete: {epoch_elapsed:.3f}s val_f1={val_f1:.6f} val_auc={val_auc:.6f}")
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_f1": val_f1, "val_auc": val_auc}
        rows.append(row)
        state = {"net": model.state_dict(), "epoch": epoch, "seed": args.seed, "config": config, "metrics": row, "test_used": False}
        torch.save(state, out / "latest_checkpoint.pt")
        if best is None or val_f1 > best["val_f1"]:
            best = row; no_improve = 0; torch.save(state, out / "best_checkpoint.pt")
        else:
            no_improve += 1
            log(f"early-stopping patience {no_improve}/{args.early_stopping_patience}")
            if no_improve >= args.early_stopping_patience:
                stop_reason = "early_stopping_patience_reached"
                break
    elapsed = time.time() - t0
    with (out / "training_log.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    changed = any(not torch.equal(before_weights[n], p.detach().cpu()) for n, p in model.named_parameters() if n in before_weights)
    logits_unchanged = all(torch.equal(before_logits[n].cpu(), p.detach().cpu()) for n, p in model.named_parameters() if n in before_logits)
    stats = budget_stats(wrappers)
    epochs_completed = len(rows)
    audit = {"qat_smoke": args.epochs <= 2, "profile": args.epochs == 1, "epochs_completed": epochs_completed,
             "best_epoch": int(best["epoch"]), "stop_reason": stop_reason,
             "patience_count_at_stop": no_improve, "conv_weights_changed": changed,
             "bit_logits_unchanged": logits_unchanged, "budget": stats,
             "budget_pass": stats["absolute_budget_error"] <= 0.01, "test_used": False,
             "official_test_run": False, "training_seconds": elapsed,
             "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else "unavailable",
             "train_manifest_sha256": sha256(TRAIN_MANIFEST), "validation_manifest_sha256": sha256(VAL_MANIFEST),
             "data_cache_dir": str(cache_dir), "epoch_timings": epoch_timings,
             "output_shape": [args.batch_size, 2, 64, 64], "finite_outputs": True}
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "README.md").write_text("Structural fixed-assignment fake-quant QAT with thin-vessel weight 8.\\n\\n" + json.dumps({"source_checkpoint": str(exp / "best_checkpoint.pt"), "source_assignment": str(assignment_path), "conv_weights_updated": changed, "bit_logits_frozen": logits_unchanged, "weighted_mean_bits": stats["parameter_weighted_mean_bitwidth"], "fake_quant": True, "thin_vessel_weight": args.thin_weight, "vessel_weight": args.vessel_weight, "test_used": False, "official_test_run": False, "real_hardware_validated": False, "data_cache_dir": str(cache_dir), "epochs_completed": epochs_completed, "best_epoch": int(best["epoch"]), "stop_reason": stop_reason, "patience_count_at_stop": no_improve}, indent=2))
    if not changed or not logits_unchanged or not audit["budget_pass"]:
        raise RuntimeError("QAT audit failed: " + json.dumps(audit))
    print(json.dumps({"output": str(out), "audit": audit, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
