#!/usr/bin/env python3
"""Diagonal-Hessian HAWQ-style group allocation for CHASEDB1.

The score is a second-order diagonal-Hessian approximation evaluated on
training calibration patches only:

    0.5 * sum_i H_ii * (Q4(w_i) - w_i)^2,

where H_ii is approximated by the squared gradient of the FOV-only CE loss.
The resulting ranking is discretized under the same parameter-weighted
INT4/INT8 budget used by the other CHASEDB1 baselines.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.UNetFamily import U_Net
from tools.group_trainable_bits import (
    _symmetric_weight_fake_quant,
    replace_conv_layers_with_groups,
)
from tools.run_chasedb1_group_stability import (
    TARGET,
    TRAIN,
    FP32,
    choose,
    loader,
    rows_from,
    stats,
)


OUT = ROOT / "experiments/CHASEDB1_GroupBits_hessian_hawq_seed2026"
GROUP_SIZE = 8
CALIBRATION_PATCHES = 320
CALIBRATION_BATCH = 32
SEED = 2026


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_scores(model, wrappers, batches, device):
    """Compute an empirical diagonal-Hessian quantization sensitivity."""
    scores = {
        name: torch.zeros(wrapper.num_groups, dtype=torch.float64)
        for name, wrapper in wrappers.items()
    }
    losses = []
    model.eval()
    for inputs, targets, fov in batches:
        inputs, targets, fov = inputs.to(device), targets.to(device), fov.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        valid = fov > 0
        loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], targets[valid])
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        for name, wrapper in wrappers.items():
            if wrapper.weight.grad is None:
                raise RuntimeError(f"missing weight gradient for {name}")
            grad = wrapper.weight.grad.detach()
            q4 = _symmetric_weight_fake_quant(wrapper.weight.detach(), 7)
            delta = q4 - wrapper.weight.detach()
            h_diag = grad.square()
            for group_id, (start, end) in enumerate(wrapper.group_slices):
                score = 0.5 * (h_diag[start:end] * delta[start:end].square()).sum()
                scores[name][group_id] += score.double().cpu()
    scale = float(max((float(v.max()) for v in scores.values()), default=1.0))
    return scores, float(np.mean(losses)), max(scale, 1e-12)


def main():
    if OUT.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {OUT}")
    if not TRAIN.is_file() or not FP32.is_file():
        raise FileNotFoundError(f"missing train manifest or checkpoint: {TRAIN}, {FP32}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = U_Net(1, 2)
    checkpoint = torch.load(FP32, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["net"], strict=True)
    wrappers = replace_conv_layers_with_groups(model, GROUP_SIZE)
    if sum(wrapper.num_groups for wrapper in wrappers.values()) != 857:
        raise RuntimeError("expected exactly 857 channel groups")
    model.to(device)
    for wrapper in wrappers.values():
        wrapper.bit_logit.requires_grad_(False)
        wrapper.weight.requires_grad_(True)
        if wrapper.bias is not None:
            wrapper.bias.requires_grad_(False)

    calibration_loader = loader(TRAIN, CALIBRATION_PATCHES, CALIBRATION_BATCH, SEED, shuffle=False)
    batches = [(x.to(device), y.to(device), f.to(device)) for x, y, f in calibration_loader]
    scores, baseline_loss, score_scale = compute_scores(model, wrappers, batches, device)

    records = rows_from(wrappers)
    for record in records:
        value = float(scores[record["layer_name"]][record["group_id"]])
        record["hessian_hawq_score"] = value
        record["normalized_hessian_hawq_score"] = value / score_scale
    ranked = sorted(records, key=lambda row: row["hessian_hawq_score"], reverse=True)
    assignment = choose(records, ranked, TARGET)
    budget = stats(records, assignment)
    budget_error = abs(budget["parameter_weighted_mean_bitwidth"] - TARGET)
    if budget_error > 0.01:
        raise RuntimeError(f"budget mismatch: {budget_error}")

    OUT.mkdir(parents=False)
    with (OUT / "group_assignment.csv").open("w", newline="") as handle:
        fields = list(records[0].keys()) + ["bit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "bit": assignment[record["key"]]})

    payload = {
        "method": "hessian_hawq_diagonal",
        "seed": SEED,
        "group_size": GROUP_SIZE,
        "total_groups": len(records),
        **budget,
        "target_weighted_mean_bitwidth": TARGET,
        "absolute_budget_error": budget_error,
        "calibration_patches": CALIBRATION_PATCHES,
        "calibration_batch_size": CALIBRATION_BATCH,
        "baseline_calibration_loss": baseline_loss,
        "score_max_before_normalization": score_scale,
        "assignment": assignment,
        "selected_group_names": [key for key, bit in assignment.items() if bit == 8],
        "budget_match_passed": True,
        "metadata": {
            "checkpoint": str(FP32),
            "checkpoint_sha256": sha256(FP32),
            "train_manifest": str(TRAIN),
            "train_manifest_sha256": sha256(TRAIN),
            "validation_or_test_used": False,
            "selection_used_validation_labels": False,
            "selection_used_test_images": False,
            "score_definition": "0.5 * sum(diag(H)_i * (Q4(w_i)-w_i)^2), diag(H) approximated by squared CE gradients",
        },
    }
    (OUT / "assignment.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "calibration_manifest_used.txt").write_text(TRAIN.read_text())
    (OUT / "calibration_manifest_used.txt.sha256").write_text(
        sha256(TRAIN) + "  calibration_manifest_used.txt\n"
    )
    (OUT / "audit.json").write_text(json.dumps({
        "method": "hessian_hawq_diagonal",
        "group_count": len(records),
        "budget_match_passed": True,
        "training_manifest_used": True,
        "validation_manifest_read": False,
        "test_manifest_read": False,
        "score_definition": payload["metadata"]["score_definition"],
    }, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "Diagonal-Hessian HAWQ-style allocation for CHASEDB1. Scores use "
        "training calibration patches only and approximate diag(H) with squared "
        "FOV-only CE gradients. The 857 G=8 groups are assigned INT4/INT8 under "
        "the same parameter-weighted target as Learned-G8 and Structural. No "
        "validation or test data was read.\n"
    )
    print(json.dumps({"output": str(OUT), "device": str(device), **budget}, indent=2))


if __name__ == "__main__":
    main()
