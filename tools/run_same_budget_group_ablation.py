import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.UNetFamily import U_Net
from tools.group_trainable_bits import replace_conv_layers_with_groups

OUT = ROOT / "experiments/DRIVE_Unet_GroupBits_same_budget_ablation_seed2026_budget_retry"
OLD_OUT = ROOT / "experiments/DRIVE_Unet_GroupBits_same_budget_ablation_seed2026"
TRAIN = ROOT / "lists/DRIVE/train_images_seed2026.txt"
VAL = ROOT / "lists/DRIVE/val_images_seed2026.txt"
TEST_MANIFEST = ROOT / "prepare_dataset/data_path_list/DRIVE/test.txt"
FP32 = ROOT / "experiments/DRIVE_Unet_FP32_full_retry/best_model.pth"
GCKPT = ROOT / "experiments/DRIVE_Unet_GroupBits_stage1_full_seed2026/best_checkpoint.pt"
SENS = ROOT / "experiments/DRIVE_Unet_GroupBits_structural_sensitivity_stability/patches1024_seed2026.csv"
TARGET_REFERENCE = 4.269841
TOLERANCE = 0.01
EXPECTED_GROUPS = 857


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def directory_sha(path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        rel = item.relative_to(path).as_posix().encode()
        if item.is_dir():
            h.update(b"D\0" + rel + b"\0")
        elif item.is_file():
            h.update(b"F\0" + rel + b"\0")
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()


def load_grouped_model():
    model = U_Net(1, 2)
    checkpoint = torch.load(FP32, map_location="cpu")
    model.load_state_dict(checkpoint["net"], strict=True)
    return model, replace_conv_layers_with_groups(model, 8)


def records(wrappers):
    result = []
    for layer_name, wrapper in wrappers.items():
        elements_per_output_channel = (
            wrapper.weight.shape[1] * wrapper.weight.shape[2] * wrapper.weight.shape[3]
        )
        for group_id, (start, end) in enumerate(wrapper.group_slices):
            result.append(
                {
                    "key": f"{layer_name}:{group_id}",
                    "layer_name": layer_name,
                    "group_id": group_id,
                    "channel_start": start,
                    "channel_end": end - 1,
                    "weight_numel": int((end - start) * elements_per_output_channel),
                }
            )
    return result


def stats(group_records, assignment):
    total = sum(item["weight_numel"] for item in group_records)
    bits = sum(item["weight_numel"] * assignment[item["key"]] for item in group_records)
    weighted = bits / total
    ordinary = sum(assignment[item["key"]] for item in group_records) / len(group_records)
    return {
        "int4_groups": sum(assignment[item["key"]] == 4 for item in group_records),
        "int8_groups": sum(assignment[item["key"]] == 8 for item in group_records),
        "ordinary_mean_bitwidth": ordinary,
        "ordinary_average_bits": ordinary,
        "parameter_weighted_mean_bitwidth": weighted,
        "parameter_weighted_average_bits": weighted,
        "theoretical_conv2d_weight_bytes": bits / 8.0,
        "theoretical_weight_bytes": bits / 8.0,
        "fp32_conv2d_weight_bytes": total * 4.0,
        "theoretical_compression_ratio": (total * 4.0) / (bits / 8.0),
    }


def validate_assignment(group_records, assignment):
    expected = {item["key"] for item in group_records}
    return set(assignment) == expected and all(value in (4, 8) for value in assignment.values())


def choose_budget(group_records, ranked_records, target_bits):
    """Parameter-mass 0/1 budget with ranked greedy seed and local refinement."""
    total = sum(item["weight_numel"] for item in group_records)
    target_int8_weight = (target_bits - 4.0) * total / 4.0
    weights = {item["key"]: item["weight_numel"] for item in group_records}
    rank_position = {item["key"]: pos for pos, item in enumerate(ranked_records)}

    def objective(chosen, current_weight):
        return (
            abs(current_weight - target_int8_weight),
            sum(rank_position[key] for key in chosen),
        )

    chosen = set()
    current_weight = 0
    for item in ranked_records:
        candidate = chosen | {item["key"]}
        candidate_weight = current_weight + item["weight_numel"]
        if objective(candidate, candidate_weight) < objective(chosen, current_weight):
            chosen, current_weight = candidate, candidate_weight

    while True:
        current_objective = objective(chosen, current_weight)
        best = None
        for item in group_records:
            key = item["key"]
            if key in chosen:
                candidate = chosen - {key}
                candidate_weight = current_weight - weights[key]
            else:
                candidate = chosen | {key}
                candidate_weight = current_weight + weights[key]
            candidate_objective = objective(candidate, candidate_weight)
            if candidate_objective < (best[0] if best else current_objective):
                best = (candidate_objective, candidate, candidate_weight)

        selected = [item for item in group_records if item["key"] in chosen]
        unselected = [item for item in group_records if item["key"] not in chosen]
        for selected_item in selected:
            for unselected_item in unselected:
                candidate = (chosen - {selected_item["key"]}) | {unselected_item["key"]}
                candidate_weight = (
                    current_weight
                    - weights[selected_item["key"]]
                    + weights[unselected_item["key"]]
                )
                candidate_objective = objective(candidate, candidate_weight)
                if candidate_objective < (best[0] if best else current_objective):
                    best = (candidate_objective, candidate, candidate_weight)

        if best is None or best[0] >= current_objective:
            break
        _, chosen, current_weight = best

    return {item["key"]: (8 if item["key"] in chosen else 4) for item in group_records}


def resolve_group_probabilities(state, layer_name, num_groups):
    suffixes = ("bit_logit", "bit_logits", "group_logits", "p_group")
    candidates = []
    for key, value in state.items():
        if key.rsplit(".", 1)[-1] in suffixes:
            base = key.rsplit(".", 1)[0]
            if key.startswith(layer_name + ".") or base.endswith(layer_name):
                candidates.append((key, value))
    for key, value in candidates:
        tensor = torch.as_tensor(value).detach().flatten()
        if tensor.numel() < num_groups:
            continue
        tensor = tensor[:num_groups].float()
        return key, tensor.clamp(0, 1) if key.endswith("p_group") else torch.sigmoid(tensor)
    return None, None


def offline():
    if OUT.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {OUT}")
    required = [TRAIN, VAL, TEST_MANIFEST, FP32, GCKPT, SENS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required offline input(s): " + ", ".join(missing))

    old_digest_before = directory_sha(OLD_OUT)
    model, wrappers = load_grouped_model()
    del model
    group_records = records(wrappers)
    if len(group_records) != EXPECTED_GROUPS:
        raise SystemExit(f"EXPECTED_{EXPECTED_GROUPS}_GROUPS_BUT_FOUND_{len(group_records)}")

    checkpoint = torch.load(GCKPT, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    probabilities, resolved_keys, unmatched = {}, {}, []
    for layer_name, wrapper in wrappers.items():
        key, values = resolve_group_probabilities(state, layer_name, wrapper.num_groups)
        if key is None:
            unmatched.extend(f"{layer_name}:{group_id}" for group_id in range(wrapper.num_groups))
        else:
            resolved_keys[layer_name] = key
            probabilities[layer_name] = values
    if unmatched:
        raise SystemExit("UNMATCHED_GROUPS: " + ",".join(unmatched[:20]))

    learned = {
        item["key"]: (
            8
            if float(probabilities[item["layer_name"]][item["group_id"]]) >= 0.5
            else 4
        )
        for item in group_records
    }
    learned_stats = stats(group_records, learned)
    target_bits = learned_stats["parameter_weighted_mean_bitwidth"]
    if abs(target_bits - TARGET_REFERENCE) > 0.01:
        print(f"WARNING: learned target {target_bits:.9f} differs from {TARGET_REFERENCE:.6f}")

    sensitivity_rows = list(csv.DictReader(SENS.open(newline="")))
    sensitivity = {f"{row['layer_name']}:{row['group_id']}": row for row in sensitivity_rows}
    structural = sorted(
        group_records,
        key=lambda item: float(sensitivity[item["key"]]["normalized_sensitivity"]),
        reverse=True,
    )
    ordinary = sorted(
        group_records,
        key=lambda item: float(sensitivity[item["key"]]["delta_loss_all"]),
        reverse=True,
    )
    low = sorted(
        group_records,
        key=lambda item: float(sensitivity[item["key"]]["normalized_sensitivity"]),
    )
    assignments = {
        "learned_g8": learned,
        "structural_sensitivity": choose_budget(group_records, structural, target_bits),
        "ordinary_loss": choose_budget(group_records, ordinary, target_bits),
        "low_sensitivity": choose_budget(group_records, low, target_bits),
    }
    for seed in (2026, 2027, 2028):
        ranked = list(group_records)
        random.Random(seed).shuffle(ranked)
        assignments[f"random_seed{seed}"] = choose_budget(group_records, ranked, target_bits)

    OUT.mkdir(parents=False)
    train_hash, val_hash, test_manifest_hash = sha(TRAIN), sha(VAL), sha(TEST_MANIFEST)
    summary, details, budget_results = [], [], {}
    for method, assignment in assignments.items():
        valid = validate_assignment(group_records, assignment)
        method_stats = stats(group_records, assignment)
        error = abs(method_stats["parameter_weighted_mean_bitwidth"] - target_bits)
        passed = valid and error <= TOLERANCE
        seed = int(method.rsplit("seed", 1)[1]) if "seed" in method else None
        payload = {
            "method": method,
            "seed": seed,
            "group_size": 8,
            "total_groups": len(group_records),
            "int4_groups": method_stats["int4_groups"],
            "int8_groups": method_stats["int8_groups"],
            "ordinary_mean_bitwidth": method_stats["ordinary_mean_bitwidth"],
            "parameter_weighted_mean_bitwidth": method_stats["parameter_weighted_mean_bitwidth"],
            "target_weighted_mean_bitwidth": target_bits,
            "absolute_budget_error": error,
            "theoretical_conv2d_weight_bytes": method_stats["theoretical_conv2d_weight_bytes"],
            "theoretical_compression_ratio": method_stats["theoretical_compression_ratio"],
            "selected_group_names": [key for key, value in assignment.items() if value == 8],
            "assignment": assignment,
            "budget_match_passed": passed,
            "metadata": {
                "train_manifest_sha256": train_hash,
                "validation_manifest_sha256": val_hash,
                "test_images_read": False,
                "selection_used_validation_labels": False,
                "selection_used_test_images": False,
                "random_seed": seed,
            },
        }
        (OUT / f"assignment_{method}.json").write_text(json.dumps(payload, indent=2) + "\n")
        summary.append({"method": method, "seed": seed, **method_stats, "target_weighted_mean_bitwidth": target_bits, "absolute_budget_error": error, "budget_match_passed": passed})
        budget_results[method] = {"absolute_budget_error": error, "budget_match_passed": passed, "assignment_valid": valid}
        details.extend({"method": method, **item, "bit": assignment[item["key"]]} for item in group_records)

    with (OUT / "allocation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "layer_group_assignment_details.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (OUT / "manifest_sha256.json").write_text(json.dumps({"train_manifest_sha256": train_hash, "validation_manifest_sha256": val_hash, "test_manifest_sha256": test_manifest_hash, "test_images_read": False}, indent=2) + "\n")

    old_digest_after = directory_sha(OLD_OUT)
    failures = [method for method, result in budget_results.items() if not result["budget_match_passed"]]
    audit = {
        "offline_only": True,
        "group_count_expected": EXPECTED_GROUPS,
        "group_count_actual": len(group_records),
        "all_857_groups_matched": len(group_records) == EXPECTED_GROUPS,
        "all_assignments_only_int4_or_int8": all(validate_assignment(group_records, assignment) for assignment in assignments.values()),
        "target_weighted_mean_bitwidth": target_bits,
        "reference_target_weighted_mean_bitwidth": TARGET_REFERENCE,
        "budget_tolerance_bits": TOLERANCE,
        "budget_results": budget_results,
        "budget_match_failed_methods": failures,
        "train_manifest_sha256": train_hash,
        "test_images_read": False,
        "smoke_test_run": False,
        "official_test_run": False,
        "training_run": False,
        "old_output_directory_sha256_before": old_digest_before,
        "old_output_directory_sha256_after": old_digest_after,
        "old_output_directory_unchanged": old_digest_before == old_digest_after,
        "resolved_checkpoint_bit_keys": resolved_keys,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / "README.md").write_text("Offline same-budget assignments only. Parameter-weighted 0/1 budget selection uses ranked greedy initialization plus local add/remove/swap refinement. No training, smoke test, official test, or test-image read was performed.\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit("BUDGET_MATCH_FAILED: " + ",".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    if args.phase != "offline":
        raise SystemExit("ONLY_OFFLINE_PHASE_ALLOWED_FOR_BUDGET_RETRY")
    offline()
