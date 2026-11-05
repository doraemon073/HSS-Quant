import csv
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_full_ablation_multiseed"
SEEDS = (2026, 2027, 2028)
METRICS = (
    "AUC", "Dice/F1", "IoU", "Accuracy", "Sensitivity", "Specificity",
    "Thin-vessel recall", "Thin-vessel Dice", "Skeleton Dice", "clDice",
)
SIZE_METRICS = (
    "Mean layer discrete bit-width",
    "Mean layer continuous bit-width",
    "Parameter-weighted mean discrete bit-width",
    "Theoretical Conv2d weight payload bytes",
    "Theoretical compression ratio vs FP32 Conv2d weights",
)
ALL_METRICS = METRICS + SIZE_METRICS + ("Test inference seconds", "GPU peak allocated GiB")


def read_csv(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_run(seed, is_no_struct):
    if is_no_struct and seed == 2026:
        run_dir = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_no_struct"
    elif is_no_struct:
        run_dir = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_no_struct_seed{}".format(seed)
    else:
        run_dir = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_seed{}".format(seed)
    test_dir = run_dir / "test_results"
    config_path = run_dir / "quant_config.json"
    test_config_path = test_dir / "quant_config.json"
    best_path = run_dir / "best_checkpoint.pt"
    map_path = run_dir / "layer_mapping.json"
    required = (config_path, test_config_path, best_path, map_path, test_dir / "performance.txt", test_dir / "result.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Seed {} {} result incomplete: {}".format(seed, suffix, missing))

    config = json.loads(config_path.read_text())
    test_config = json.loads(test_config_path.read_text())
    mapping = json.loads(map_path.read_text())
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    expected_lambda = 0.0 if is_no_struct else 1.0
    if float(config["lambda_struct"]) != expected_lambda:
        raise ValueError("Unexpected structural-loss weight in {}".format(run_dir))
    hyperparameters = (
        int(config["epochs"]), int(config["training_patches"]), int(config["batch_size"]),
        int(config["calibration_patches"]), float(config["lambda_bit"]),
        float(config["learning_rate_bits"]),
    )
    if hyperparameters != (20, 5000, 64, 320, 0.001, 0.01):
        raise ValueError("Fixed hyperparameter mismatch in {}: {}".format(run_dir, hyperparameters))
    if config["test_used_for_training_validation_calibration_early_stopping"] is not False:
        raise ValueError("Test set use is not disabled in {}".format(run_dir))
    if int(test_config["inference_passes"]) != 1 or test_config["training_performed"]:
        raise ValueError("Expected a single evaluation-only test pass for {}".format(run_dir))
    if test_config["recalibration_performed"] or test_config["bit_widths_modified"]:
        raise ValueError("Test evaluation changed calibration or learned bits in {}".format(run_dir))
    if test_config["best_checkpoint"] != str(best_path) or int(checkpoint["epoch"]) != int(test_config["best_epoch"]):
        raise ValueError("Test output is not tied to the matching best checkpoint in {}".format(run_dir))
    if test_config["official_test_manifest"] != str(ROOT / "prepare_dataset/data_path_list/DRIVE/test.txt"):
        raise ValueError("Unexpected test manifest for {}".format(run_dir))
    if not test_config["fov_only_pixel_metrics"] or int(test_config["test_images"]) != 20:
        raise ValueError("Expected FOV-only metrics on 20 test images for {}".format(run_dir))
    if len(mapping) != 23 or len(checkpoint["p_l"]) != 23:
        raise ValueError("Expected 23 Conv2d layers in {}".format(run_dir))
    if set(mapping) != set(checkpoint["p_l"]) or set(mapping) != set(checkpoint["discrete_bit_width"]):
        raise ValueError("Layer names disagree in {}".format(run_dir))
    for layer_name, probability in checkpoint["p_l"].items():
        expected_bit = 8 if float(probability) >= 0.5 else 4
        if int(checkpoint["discrete_bit_width"][layer_name]) != expected_bit:
            raise ValueError("Saved hard bit does not match p(INT8) in {}: {}".format(run_dir, layer_name))

    test_metrics = test_config["test_metrics"]
    struct_metrics = test_config["structure_metrics"]
    metric_values = {name: float(test_metrics[name]) for name in METRICS[:6]}
    metric_values.update({
        "Thin-vessel recall": float(struct_metrics["thin_vessel_recall"]),
        "Thin-vessel Dice": float(struct_metrics["thin_vessel_dice"]),
        "Skeleton Dice": float(struct_metrics["skeleton_dice"]),
        "clDice": float(struct_metrics["clDice"]),
        "Mean layer discrete bit-width": float(test_config["mean_discrete_bit_width_layer_average"]),
        "Mean layer continuous bit-width": float(test_config["mean_continuous_bit_width_layer_average"]),
        "Parameter-weighted mean discrete bit-width": float(test_config["parameter_weighted_mean_discrete_bit_width"]),
        "Theoretical Conv2d weight payload bytes": float(test_config["theoretical_conv_weight_payload_bytes"]),
        "Theoretical compression ratio vs FP32 Conv2d weights": float(test_config["theoretical_compression_ratio_vs_fp32_conv_weights"]),
        "Test inference seconds": float(test_config["test_seconds"]),
        "GPU peak allocated GiB": float(test_config["gpu_peak_allocated_gib"]),
    })
    return {
        "seed": seed,
        "method": "no_struct" if is_no_struct else "full",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_f1": float(checkpoint["validation_metrics"]["val_f1"]),
        "lambda_struct": expected_lambda,
        "test_inference_passes": 1,
        "metrics": metric_values,
        "mapping": mapping,
        "probabilities": checkpoint["p_l"],
        "bits": checkpoint["discrete_bit_width"],
        "config": config,
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError("Refusing to overwrite existing multiseed report: {}".format(OUTPUT_DIR))

    runs = {}
    for seed in SEEDS:
        runs[(seed, "full")] = load_run(seed, is_no_struct=False)
        runs[(seed, "no_struct")] = load_run(seed, is_no_struct=True)

    reference = runs[(2026, "full")]["config"]
    manifest_keys = ("training_manifest", "validation_manifest", "checkpoint")
    for result in runs.values():
        for key in manifest_keys:
            if result["config"][key] != reference[key]:
                raise ValueError("Fixed data/checkpoint differs across runs for {}".format(key))

    OUTPUT_DIR.mkdir(parents=True)
    raw_rows = []
    paired_rows = []
    for seed in SEEDS:
        full = runs[(seed, "full")]
        no_struct = runs[(seed, "no_struct")]
        for result in (full, no_struct):
            row = {
                "seed": result["seed"],
                "method": result["method"],
                "lambda_struct": result["lambda_struct"],
                "best_epoch": result["best_epoch"],
                "best_validation_f1": result["best_validation_f1"],
                "run_dir": result["run_dir"],
                "best_checkpoint": result["best_checkpoint"],
                "test_inference_passes": result["test_inference_passes"],
            }
            row.update(result["metrics"])
            raw_rows.append(row)
        delta = {metric: no_struct["metrics"][metric] - full["metrics"][metric] for metric in ALL_METRICS}
        paired_rows.append({"seed": seed, "delta_definition": "no_struct minus full", **delta})

    write_csv(OUTPUT_DIR / "seed_results.csv", raw_rows)
    write_csv(OUTPUT_DIR / "paired_differences.csv", paired_rows)

    aggregate_rows = []
    for method in ("full", "no_struct"):
        for metric in ("best_epoch", "best_validation_f1") + ALL_METRICS:
            values = [
                float(runs[(seed, method)]["best_epoch"] if metric == "best_epoch"
                      else runs[(seed, method)]["best_validation_f1"] if metric == "best_validation_f1"
                      else runs[(seed, method)]["metrics"][metric])
                for seed in SEEDS
            ]
            aggregate_rows.append({
                "method": method,
                "metric": metric,
                "mean": statistics.mean(values),
                "sample_std_ddof_1": statistics.stdev(values),
                "n": len(values),
            })
    write_csv(OUTPUT_DIR / "mean_std.csv", aggregate_rows)

    layer_rows = []
    for layer_name in sorted(runs[(2026, "full")]["mapping"]):
        full_reference = runs[(2026, "full")]["mapping"][layer_name]
        row = {
            "layer": layer_name,
            "group": full_reference["module"],
            "weight_numel": int(full_reference["weight_numel"]),
        }
        for seed in SEEDS:
            for method in ("full", "no_struct"):
                result = runs[(seed, method)]
                if result["mapping"][layer_name] != full_reference:
                    raise ValueError("Layer metadata mismatch for {}".format(layer_name))
                row["p_int8_{}_seed{}".format(method, seed)] = float(result["probabilities"][layer_name])
                row["bitwidth_{}_seed{}".format(method, seed)] = "INT{}".format(result["bits"][layer_name])
        layer_rows.append(row)
    write_csv(OUTPUT_DIR / "bitwidth_allocation_comparison.csv", layer_rows)

    group_counts = {}
    for group in ("encoder", "bottleneck", "decoder", "output"):
        group_counts[group] = {}
        names = [row["layer"] for row in layer_rows if row["group"] == group]
        for seed in SEEDS:
            group_counts[group][str(seed)] = {}
            for method in ("full", "no_struct"):
                group_counts[group][str(seed)][method] = {
                    str(bit): sum(runs[(seed, method)]["bits"][name] == bit for name in names)
                    for bit in (4, 8)
                }

    summary = {
        "seeds": list(SEEDS),
        "methods": {"full": {"lambda_struct": 1.0}, "no_struct": {"lambda_struct": 0.0}},
        "fixed_hyperparameters": {
            "epochs": 20,
            "training_patches_per_epoch": 5000,
            "batch_size": 64,
            "training_only_calibration_patches": 320,
            "learning_rate_bits": 0.01,
            "lambda_bit": 0.001,
            "device": "cuda:0",
            "training_manifest": reference["training_manifest"],
            "validation_manifest": reference["validation_manifest"],
            "fp32_checkpoint": reference["checkpoint"],
            "official_test_manifest": "prepare_dataset/data_path_list/DRIVE/test.txt",
        },
        "test_inference_passes_by_seed_and_method": {
            str(seed): {method: 1 for method in ("full", "no_struct")} for seed in SEEDS
        },
        "existing_full_method_results_reused_without_retest_for_seeds": [2027, 2028],
        "no_struct_runs_trained_and_tested_for_seeds": [2027, 2028],
        "all_metrics_fov_only": True,
        "test_used_for_training_calibration_early_stopping_or_bit_selection": False,
        "all_per_layer_assignments": "bitwidth_allocation_comparison.csv",
        "group_bit_counts": group_counts,
        "difference_sign": "no_struct minus full",
        "statistics": "Per-method mean and sample standard deviation (ddof=1) over 3 seeds on the same fixed split.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    means = {method: {metric: next(row["mean"] for row in aggregate_rows
                                  if row["method"] == method and row["metric"] == metric)
                      for metric in METRICS}
             for method in ("full", "no_struct")}
    stds = {method: {metric: next(row["sample_std_ddof_1"] for row in aggregate_rows
                                 if row["method"] == method and row["metric"] == metric)
                     for metric in METRICS}
            for method in ("full", "no_struct")}
    paired_means = {metric: statistics.mean([row[metric] for row in paired_rows]) for metric in METRICS}
    delta_recall = [row["Thin-vessel recall"] for row in paired_rows]
    delta_cldice = [row["clDice"] for row in paired_rows]
    recall_signs = ["improved" if value < 0 else "not improved" for value in delta_recall]
    cldice_signs = ["improved" if value < 0 else "not improved" for value in delta_cldice]

    readme = """# StructuralTrainableBits Structural-Loss Multi-Seed Ablation

## Protocol

This report compares the complete StructuralTrainableBits method (`lambda_struct=1`) with its no_struct ablation (`lambda_struct=0`) for seeds 2026, 2027, and 2028. Each run uses 20 epochs, 5,000 patches per epoch, batch size 64, 320 calibration patches from the training manifest only, `lr_bits=0.01`, `lambda_bit=0.001`, the fixed 16/4 train-validation split, and the same epoch-9 FP32 checkpoint. FP32 weights are frozen and only per-layer bit-width logits are optimized.

The complete-method test outputs for seeds 2027 and 2028 were reused from existing runs; those seeds were not tested again. The complete-method seed2026 result was likewise reused. Only no_struct seeds 2027 and 2028 were newly trained here and each was evaluated once after training using its best validation checkpoint. no_struct seed2026 was reused from its existing single-run ablation. Thus every seed/method pair has one recorded test inference, with no repeat evaluation. The DRIVE test set was not used for training, calibration, early stopping, or bit-width selection. All pixel metrics use the official FOV region.

## Paired comparison

`paired_differences.csv` defines deltas as no_struct minus full. Negative deltas therefore favor the full method for metrics where larger is better. Across the three seeds, mean thin-vessel recall is {:.6f} (sample SD {:.6f}) for full and {:.6f} (sample SD {:.6f}) for no_struct; mean paired difference is {:.6f}. The full method improves thin-vessel recall in {} of 3 paired seeds. Mean clDice is {:.6f} (sample SD {:.6f}) for full and {:.6f} (sample SD {:.6f}) for no_struct; mean paired difference is {:.6f}. The full method improves clDice in {} of 3 paired seeds.

These are descriptive results for three random seeds on one fixed split, not independent split-level estimates or a significance test. Report the per-seed differences and sample standard deviations; do not claim a universal benefit if the paired outcomes are mixed.

## Files

- `seed_results.csv`: overall, structural, bitwidth, storage and per-run test efficiency values for all six seed/method pairs.
- `paired_differences.csv`: per-seed differences, defined as no_struct minus full.
- `mean_std.csv`: per-method means and sample standard deviations (`ddof=1`).
- `bitwidth_allocation_comparison.csv`: p(INT8) and final hard INT4/INT8 assignment for each of 23 layers across all six runs.
- `summary.json`: fixed protocol, test-count provenance, and group-wise bit allocation counts.

Quantized predictions are fake quantization executed with FP32 arithmetic. Theoretical Conv2d weight compression is a payload estimate, not measured storage or hardware acceleration. Real TensorRT/integer-kernel deployment and speed have not been validated.
""".format(
        means["full"]["Thin-vessel recall"], stds["full"]["Thin-vessel recall"],
        means["no_struct"]["Thin-vessel recall"], stds["no_struct"]["Thin-vessel recall"],
        paired_means["Thin-vessel recall"], sum(value < 0 for value in delta_recall),
        means["full"]["clDice"], stds["full"]["clDice"],
        means["no_struct"]["clDice"], stds["no_struct"]["clDice"],
        paired_means["clDice"], sum(value < 0 for value in delta_cldice),
    )
    (OUTPUT_DIR / "README.md").write_text(readme)

    print("Multi-seed ablation report:", OUTPUT_DIR)
    print("Per-seed paired deltas (no_struct - full):")
    for row in paired_rows:
        print(row["seed"], "thin_recall", row["Thin-vessel recall"], "clDice", row["clDice"])
    print("Full improves thin-vessel recall/clDice in seeds:", recall_signs, cldice_signs)
    print("Output files:", sorted(path.name for path in OUTPUT_DIR.iterdir()))


if __name__ == "__main__":
    main()