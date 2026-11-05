import csv
import json
import statistics
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments"
SEEDS = (2026, 2027, 2028)
OUTPUT_DIR = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_reproducibility_report"
METRICS = (
    "AUC",
    "Dice/F1",
    "IoU",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Thin-vessel recall",
    "Thin-vessel Dice",
    "Skeleton Dice",
    "clDice",
    "Mean layer discrete bit-width",
    "Mean layer continuous bit-width",
    "Parameter-weighted mean discrete bit-width",
    "Theoretical Conv2d weight payload bytes",
    "Theoretical compression ratio vs FP32 Conv2d weights",
    "Test inference seconds",
    "GPU peak allocated GiB",
)


def read_csv(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_seed(seed):
    run_dir = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_seed{}".format(seed)
    test_dir = run_dir / "test_results"
    required = (
        run_dir / "quant_config.json",
        run_dir / "training_log.csv",
        run_dir / "best_checkpoint.pt",
        run_dir / "layer_mapping.json",
        test_dir / "quant_config.json",
        test_dir / "performance.txt",
        test_dir / "comparison.csv",
        test_dir / "result.npy",
        test_dir / "test_manifest_used.txt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Seed {} is incomplete: {}".format(seed, missing))

    run_config = json.loads((run_dir / "quant_config.json").read_text())
    test_config = json.loads((test_dir / "quant_config.json").read_text())
    training_rows = read_csv(run_dir / "training_log.csv")
    if len(training_rows) != 20:
        raise ValueError("Seed {} has {} training epochs, expected 20".format(seed, len(training_rows)))
    if int(run_config["epochs"]) != 20 or int(run_config["training_patches"]) != 5000:
        raise ValueError("Seed {} training configuration does not match the fixed run".format(seed))
    if int(test_config["inference_passes"]) != 1 or test_config["training_performed"]:
        raise ValueError("Seed {} test config is not a one-pass evaluation-only run".format(seed))
    if test_config["recalibration_performed"] or test_config["bit_widths_modified"]:
        raise ValueError("Seed {} evaluation recalibrated or changed bit-widths".format(seed))
    if test_config["best_checkpoint"].endswith("latest_checkpoint.pt"):
        raise ValueError("Seed {} test did not use its best checkpoint".format(seed))
    if len(list((test_dir / "result_img").glob("*.png"))) != 20:
        raise ValueError("Seed {} does not have 20 test prediction images".format(seed))

    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    if not 1 <= int(checkpoint["epoch"]) <= 20:
        raise ValueError("Seed {} best checkpoint epoch is outside the 20-epoch run".format(seed))
    layer_mapping = json.loads((run_dir / "layer_mapping.json").read_text())
    if len(layer_mapping) != 23 or len(checkpoint["p_l"]) != 23:
        raise ValueError("Seed {} checkpoint/mapping must describe 23 Conv2d layers".format(seed))
    if set(layer_mapping) != set(checkpoint["p_l"]) or set(layer_mapping) != set(checkpoint["discrete_bit_width"]):
        raise ValueError("Seed {} checkpoint and layer map disagree".format(seed))
    for layer, probability in checkpoint["p_l"].items():
        expected_bits = 8 if probability >= 0.5 else 4
        if int(checkpoint["discrete_bit_width"][layer]) != expected_bits:
            raise ValueError("Seed {} has inconsistent hard bit assignment for {}".format(seed, layer))

    test_metrics = test_config["test_metrics"]
    structures = test_config["structure_metrics"]
    values = {
        "seed": seed,
        "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_f1": float(checkpoint["validation_metrics"]["val_f1"]),
        "AUC": float(test_metrics["AUC"]),
        "Dice/F1": float(test_metrics["Dice/F1"]),
        "IoU": float(test_metrics["IoU"]),
        "Accuracy": float(test_metrics["Accuracy"]),
        "Sensitivity": float(test_metrics["Sensitivity"]),
        "Specificity": float(test_metrics["Specificity"]),
        "Thin-vessel recall": float(structures["thin_vessel_recall"]),
        "Thin-vessel Dice": float(structures["thin_vessel_dice"]),
        "Skeleton Dice": float(structures["skeleton_dice"]),
        "clDice": float(structures["clDice"]),
        "Mean layer discrete bit-width": float(test_config["mean_discrete_bit_width_layer_average"]),
        "Mean layer continuous bit-width": float(test_config["mean_continuous_bit_width_layer_average"]),
        "Parameter-weighted mean discrete bit-width": float(test_config["parameter_weighted_mean_discrete_bit_width"]),
        "Theoretical Conv2d weight payload bytes": int(test_config["theoretical_conv_weight_payload_bytes"]),
        "Theoretical compression ratio vs FP32 Conv2d weights": float(test_config["theoretical_compression_ratio_vs_fp32_conv_weights"]),
        "Test inference seconds": float(test_config["test_seconds"]),
        "GPU peak allocated GiB": float(test_config["gpu_peak_allocated_gib"]),
        "test_images": int(test_config["test_images"]),
    }
    if values["test_images"] != 20 or not test_config["fov_only_pixel_metrics"]:
        raise ValueError("Seed {} test result is not the expected 20-image FOV evaluation".format(seed))

    layer_rows = []
    for name, info in layer_mapping.items():
        layer_rows.append({
            "layer": name,
            "group": info["module"],
            "weight_numel": int(info["weight_numel"]),
            "p_int8_seed{}".format(seed): float(checkpoint["p_l"][name]),
            "bitwidth_seed{}".format(seed): "INT{}".format(checkpoint["discrete_bit_width"][name]),
        })
    return values, layer_rows


def write_csv(path, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError("Refusing to overwrite reproducibility report: {}".format(OUTPUT_DIR))

    seed_results = []
    layers_by_name = {}
    for seed in SEEDS:
        result, layer_rows = load_seed(seed)
        seed_results.append(result)
        for row in layer_rows:
            name = row["layer"]
            if name not in layers_by_name:
                layers_by_name[name] = {"layer": name, "group": row["group"], "weight_numel": row["weight_numel"]}
            elif (layers_by_name[name]["group"], layers_by_name[name]["weight_numel"]) != (row["group"], row["weight_numel"]):
                raise ValueError("Layer metadata differs between seeds for {}".format(name))
            layers_by_name[name].update({key: value for key, value in row.items() if key.startswith(("p_int8_", "bitwidth_"))})
    if len(layers_by_name) != 23:
        raise ValueError("Expected 23 shared Conv2d layer names, found {}".format(len(layers_by_name)))

    OUTPUT_DIR.mkdir(parents=True)
    write_csv(OUTPUT_DIR / "seed_results.csv", seed_results)

    stats_rows = []
    for metric in ("best_epoch", "best_validation_f1") + METRICS:
        observations = [float(row[metric]) for row in seed_results]
        stats_rows.append({
            "metric": metric,
            "mean": statistics.mean(observations),
            "sample_std_ddof_1": statistics.stdev(observations),
            "n": len(observations),
        })
    write_csv(OUTPUT_DIR / "mean_std.csv", stats_rows)

    layer_rows = [layers_by_name[name] for name in sorted(layers_by_name)]
    write_csv(OUTPUT_DIR / "bitwidth_allocation_comparison.csv", layer_rows)

    group_counts = {}
    for group in ("encoder", "bottleneck", "decoder", "output"):
        group_counts[group] = {}
        for seed in SEEDS:
            group_layers = [row for row in layer_rows if row["group"] == group]
            key = "bitwidth_seed{}".format(seed)
            group_counts[group][str(seed)] = {
                bit: sum(row[key] == "INT{}".format(bit) for row in group_layers)
                for bit in (4, 8)
            }

    train_configs = [json.loads((EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_seed{}".format(seed) / "quant_config.json").read_text()) for seed in SEEDS]
    test_configs = [json.loads((EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_seed{}".format(seed) / "test_results" / "quant_config.json").read_text()) for seed in SEEDS]
    if len({(config["lambda_bit"], config["lambda_struct"], config["learning_rate_bits"], config["batch_size"], config["calibration_patches"], config["training_patches"]) for config in train_configs}) != 1:
        raise ValueError("Training hyperparameters differ across seed runs")
    summary = {
        "seeds": list(SEEDS),
        "seed_source": "Run directory names and executed CLI arguments; the trainer config did not persist a seed field.",
        "training_hyperparameters": {
            "epochs": train_configs[0]["epochs"],
            "training_patches_per_epoch": train_configs[0]["training_patches"],
            "calibration_patches_training_only": train_configs[0]["calibration_patches"],
            "batch_size": train_configs[0]["batch_size"],
            "learning_rate_bits": train_configs[0]["learning_rate_bits"],
            "lambda_bit": train_configs[0]["lambda_bit"],
            "lambda_struct": train_configs[0]["lambda_struct"],
            "checkpoint": train_configs[0]["checkpoint"],
            "train_manifest": train_configs[0]["training_manifest"],
            "validation_manifest": train_configs[0]["validation_manifest"],
            "official_test_used_during_training_calibration_selection": False,
        },
        "per_seed_test_inference_count": {str(seed): config["inference_passes"] for seed, config in zip(SEEDS, test_configs)},
        "all_test_metrics_fov_only": all(config["fov_only_pixel_metrics"] for config in test_configs),
        "mean_and_sample_standard_deviation": {row["metric"]: {"mean": row["mean"], "sample_std_ddof_1": row["sample_std_ddof_1"]} for row in stats_rows},
        "bitwidth_group_counts": group_counts,
        "per_layer_assignment_csv": "bitwidth_allocation_comparison.csv",
        "interpretation": "Descriptive variability over three fixed-seed runs on one fixed split; not a multi-split estimate of generalization.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    readme = """# StructuralTrainableBits Reproducibility Runs

This report summarizes three runs with seeds 2026, 2027, and 2028. Each run used the same frozen epoch-9 FP32 checkpoint, the same fixed 16-original training / 4-original validation split, and the same method/hyperparameters. The official 20-image DRIVE test set was used only after training, once per seed, with that run's best validation checkpoint. Test images were not used for training, calibration, early stopping, or bit-width assignment.

The fixed training settings were 20 epochs, 5,000 training patches per epoch, batch size {}, 320 calibration patches sampled only from the training manifest, learning rate {}, `lambda_bit={}`, and `lambda_struct={}`. FP32 network weights remained frozen; only 23 per-Conv2d bit logits were optimized. The saved best checkpoint was selected by validation F1 independently for each seed.

`seed_results.csv` contains per-seed best epoch, best validation F1, official-test FOV metrics, and bit-width/storage statistics. `mean_std.csv` reports the three-run mean and sample standard deviation (`ddof=1`). `bitwidth_allocation_comparison.csv` lists every Conv2d layer's p(INT8) and final INT4/INT8 assignment from each seed's best checkpoint. `summary.json` records protocol, fixed hyperparameters, and group-wise bit counts.

## Scope and caveats

All quantized inference is fake quantization with FP32 arithmetic, not actual integer-kernel execution. The reported model sizes and compression ratios are theoretical Conv2d weight-payload estimates; they exclude quantization scales, biases, non-Conv2d state, and runtime overhead. TensorRT or other real hardware deployment and speed are not evaluated here.

The current trainer's run configuration did not persist the seed field. Seed identities in this summary are taken from the explicitly seeded launch commands and their separate output directory names. The reported standard deviations describe variability across three random seeds on the same fixed split; they are not confidence intervals and do not measure variability across dataset splits. Results should be interpreted descriptively, without implying a general performance guarantee.
""".format(
        train_configs[0]["batch_size"], train_configs[0]["learning_rate_bits"],
        train_configs[0]["lambda_bit"], train_configs[0]["lambda_struct"],
    )
    (OUTPUT_DIR / "README.md").write_text(readme)

    print("Generated reproducibility report:", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)
    print("Per-seed test results:")
    for row in seed_results:
        print(row["seed"], "best_epoch", row["best_epoch"], "AUC", row["AUC"],
              "Dice/F1", row["Dice/F1"], "clDice", row["clDice"],
              "mean_bits", row["Mean layer discrete bit-width"])


if __name__ == "__main__":
    main()
