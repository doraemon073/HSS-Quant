import csv
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
FULL_RUN = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_seed2026"
NO_STRUCT_RUN = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_no_struct"
OUTPUT_DIR = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_full_ablation"
METRIC_NAMES = (
    "AUC", "Dice/F1", "IoU", "Accuracy", "Sensitivity", "Specificity",
    "Thin-vessel recall", "Thin-vessel Dice", "Skeleton Dice", "clDice",
)


def load_arm(run_dir, expected_lambda_struct):
    train_config_path = run_dir / "quant_config.json"
    test_config_path = run_dir / "test_results" / "quant_config.json"
    mapping_path = run_dir / "layer_mapping.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    for path in (train_config_path, test_config_path, mapping_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    train_config = json.loads(train_config_path.read_text())
    test_config = json.loads(test_config_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if train_config["lambda_struct"] != expected_lambda_struct:
        raise ValueError("Unexpected lambda_struct in {}".format(run_dir))
    if train_config["seed"] if "seed" in train_config else False:
        raise ValueError("Legacy run config unexpectedly encodes a seed; verify provenance manually")
    if (train_config["epochs"], train_config["training_patches"], train_config["batch_size"],
            train_config["calibration_patches"], train_config["lambda_bit"],
            train_config["learning_rate_bits"]) != (20, 5000, 64, 320, 0.001, 0.01):
        raise ValueError("Training hyperparameters differ from the fixed ablation protocol")
    if test_config["inference_passes"] != 1 or test_config["training_performed"]:
        raise ValueError("Each arm must use a single test inference after training")
    if test_config["recalibration_performed"] or test_config["bit_widths_modified"]:
        raise ValueError("Evaluation modified or recalibrated learned bit-widths")
    if test_config["best_checkpoint"] != str(checkpoint_path):
        raise ValueError("Test output is not tied to this arm's best checkpoint")
    if int(checkpoint["epoch"]) != int(test_config["best_epoch"]):
        raise ValueError("Saved best checkpoint epoch differs from test provenance")
    if len(mapping) != 23 or len(checkpoint["p_l"]) != 23:
        raise ValueError("Expected exactly 23 learned Conv2d layer assignments")
    if set(mapping) != set(checkpoint["p_l"]) or set(mapping) != set(checkpoint["discrete_bit_width"]):
        raise ValueError("Layer mapping and checkpoint assignments differ")
    return train_config, test_config, mapping, checkpoint


def write_csv(path, rows):
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError("Refusing to overwrite existing full_ablation directory: {}".format(OUTPUT_DIR))

    full_train, full_test, full_mapping, full_checkpoint = load_arm(FULL_RUN, 1.0)
    no_train, no_test, no_mapping, no_checkpoint = load_arm(NO_STRUCT_RUN, 0.0)
    if full_mapping != no_mapping:
        raise ValueError("The two ablation arms do not share the same Conv2d mapping")
    if full_train["checkpoint"] != no_train["checkpoint"]:
        raise ValueError("The two ablation arms do not use the same FP32 source checkpoint")
    for key in ("training_manifest", "validation_manifest"):
        if full_train[key] != no_train[key]:
            raise ValueError("The arms do not share {}".format(key))
    if full_test["official_test_manifest"] != no_test["official_test_manifest"]:
        raise ValueError("The arms do not share the official test manifest")

    OUTPUT_DIR.mkdir(parents=True)
    arms = (
        ("full", "StructuralTrainableBits full method", FULL_RUN, full_train, full_test, full_mapping, full_checkpoint),
        ("no_struct", "StructuralTrainableBits without structural loss", NO_STRUCT_RUN, no_train, no_test, no_mapping, no_checkpoint),
    )
    metric_rows = []
    layer_rows = []
    for arm_key, label, run_dir, train_config, test_config, mapping, checkpoint in arms:
        metrics = test_config["test_metrics"]
        row = {
            "method": label,
            "arm": arm_key,
            "seed": 2026,
            "lambda_struct": train_config["lambda_struct"],
            "best_epoch": checkpoint["epoch"],
            "best_validation_f1": checkpoint["validation_metrics"]["val_f1"],
        }
        row.update({name: metrics[name] for name in METRIC_NAMES})
        row.update({
            "mean_discrete_bitwidth_per_layer": test_config["mean_discrete_bit_width_layer_average"],
            "mean_continuous_bitwidth_per_layer": test_config["mean_continuous_bit_width_layer_average"],
            "parameter_weighted_mean_discrete_bitwidth": test_config["parameter_weighted_mean_discrete_bit_width"],
            "theoretical_conv_weight_payload_bytes": test_config["theoretical_conv_weight_payload_bytes"],
            "theoretical_compression_ratio_vs_fp32_conv_weights": test_config["theoretical_compression_ratio_vs_fp32_conv_weights"],
            "test_inference_seconds": test_config["test_seconds"],
            "gpu_peak_allocated_gib": test_config["gpu_peak_allocated_gib"],
            "best_checkpoint": str(run_dir / "best_checkpoint.pt"),
            "test_manifest": test_config["official_test_manifest"],
            "test_inference_passes": test_config["inference_passes"],
        })
        metric_rows.append(row)
        for name, details in mapping.items():
            probability = float(checkpoint["p_l"][name])
            bits = int(checkpoint["discrete_bit_width"][name])
            layer_rows.append({
                "arm": arm_key,
                "method": label,
                "layer": name,
                "group": details["module"],
                "weight_numel": details["weight_numel"],
                "p_int8": probability,
                "final_bitwidth": "INT{}".format(bits),
                "discrete_bits": bits,
            })

    write_csv(OUTPUT_DIR / "ablation_results.csv", metric_rows)
    write_csv(OUTPUT_DIR / "layer_bitwidth_comparison.csv", layer_rows)
    (OUTPUT_DIR / "full_method_reference.json").write_text(json.dumps({
        "full_arm_reused_from_existing_run": str(FULL_RUN),
        "no_struct_arm": str(NO_STRUCT_RUN),
        "full_arm_training_rerun": False,
        "full_arm_test_rerun": False,
        "no_struct_training_epochs": 20,
        "no_struct_test_inference_passes": no_test["inference_passes"],
        "official_test_used_for_training_calibration_or_selection": False,
        "fixed_train_manifest": full_train["training_manifest"],
        "fixed_validation_manifest": full_train["validation_manifest"],
        "fixed_test_manifest": full_test["official_test_manifest"],
    }, indent=2))

    metric_by_arm = {row["arm"]: row for row in metric_rows}
    deltas = {name: metric_by_arm["no_struct"][name] - metric_by_arm["full"][name]
              for name in METRIC_NAMES}
    readme = """# StructuralTrainableBits Minimal Ablation

## Protocol

Both arms use seed 2026, the same epoch-9 FP32 checkpoint, fixed train/validation manifests, 20 epochs, 5,000 patches per epoch, batch size 64, 320 training-only calibration patches, `lambda_bit=0.001`, and bit-logit learning rate 0.01. The only method difference is `lambda_struct`: 1.0 for the full method and 0.0 for no_struct. FP32 weights remain frozen; only the 23 per-layer bit logits are trained.

The full-method result is reused from the existing, exactly matching seed-2026 run at `{}`. Its test inference was not repeated. The no_struct arm was trained in `{}` and its best validation checkpoint (epoch {}) was evaluated once on the official DRIVE test set. Test data was not used for training, calibration, validation-based checkpoint selection, or bit-width assignment. Metrics are FOV-only.

## Results

| Method | Best epoch | AUC | Dice/F1 | IoU | Accuracy | Sensitivity | Specificity | Thin recall | Thin Dice | Skeleton Dice | clDice | Mean layer bits | Parameter-weighted bits | Theoretical Conv2d weight bytes | Compression ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full method (`lambda_struct=1`) | {f_epoch} | {f_auc:.6f} | {f_dice:.6f} | {f_iou:.6f} | {f_acc:.6f} | {f_sens:.6f} | {f_spec:.6f} | {f_thin:.6f} | {f_thind:.6f} | {f_skel:.6f} | {f_cl:.6f} | {f_bits:.3f} | {f_wbits:.3f} | {f_bytes} | {f_comp:.3f}x |
| No structural loss (`lambda_struct=0`) | {n_epoch} | {n_auc:.6f} | {n_dice:.6f} | {n_iou:.6f} | {n_acc:.6f} | {n_sens:.6f} | {n_spec:.6f} | {n_thin:.6f} | {n_thind:.6f} | {n_skel:.6f} | {n_cl:.6f} | {n_bits:.3f} | {n_wbits:.3f} | {n_bytes} | {n_comp:.3f}x |

Delta (no_struct minus full): thin-vessel recall {d_thin:+.6f}; thin-vessel Dice {d_thind:+.6f}; skeleton Dice {d_skel:+.6f}; clDice {d_cl:+.6f}; AUC {d_auc:+.6f}; Dice/F1 {d_dice:+.6f}.

## Interpretation and limitations

On this single fixed split and seed, removing the structural term changes thin-vessel recall and clDice only modestly; the full method has higher values for both, but the differences are small. Skeleton Dice is slightly higher in no_struct. Overall Dice/F1 is nearly unchanged, while AUC is slightly higher in no_struct. This one-seed ablation does not establish statistical significance or a general causal effect; the structural term's benefit is mixed rather than uniformly demonstrated.

Both arms use fake quantization with FP32 arithmetic. The storage size and compression ratio are theoretical Conv2d weight-payload estimates, not measured file-size reductions or hardware speedups. Real INT4/INT8 kernel or TensorRT deployment is not verified.

`ablation_results.csv` contains the two-arm metric and efficiency comparison. `layer_bitwidth_comparison.csv` records the final probability and hard INT4/INT8 assignment for every Conv2d layer in both arms. `full_method_reference.json` documents the reuse provenance and test protocol.
""".format(
        FULL_RUN.relative_to(ROOT), NO_STRUCT_RUN.relative_to(ROOT), no_checkpoint["epoch"],
        f_epoch=metric_by_arm["full"]["best_epoch"], f_auc=metric_by_arm["full"]["AUC"],
        f_dice=metric_by_arm["full"]["Dice/F1"], f_iou=metric_by_arm["full"]["IoU"],
        f_acc=metric_by_arm["full"]["Accuracy"], f_sens=metric_by_arm["full"]["Sensitivity"],
        f_spec=metric_by_arm["full"]["Specificity"], f_thin=metric_by_arm["full"]["Thin-vessel recall"],
        f_thind=metric_by_arm["full"]["Thin-vessel Dice"], f_skel=metric_by_arm["full"]["Skeleton Dice"],
        f_cl=metric_by_arm["full"]["clDice"], f_bits=metric_by_arm["full"]["mean_discrete_bitwidth_per_layer"],
        f_wbits=metric_by_arm["full"]["parameter_weighted_mean_discrete_bitwidth"],
        f_bytes=metric_by_arm["full"]["theoretical_conv_weight_payload_bytes"],
        f_comp=metric_by_arm["full"]["theoretical_compression_ratio_vs_fp32_conv_weights"],
        n_epoch=metric_by_arm["no_struct"]["best_epoch"], n_auc=metric_by_arm["no_struct"]["AUC"],
        n_dice=metric_by_arm["no_struct"]["Dice/F1"], n_iou=metric_by_arm["no_struct"]["IoU"],
        n_acc=metric_by_arm["no_struct"]["Accuracy"], n_sens=metric_by_arm["no_struct"]["Sensitivity"],
        n_spec=metric_by_arm["no_struct"]["Specificity"], n_thin=metric_by_arm["no_struct"]["Thin-vessel recall"],
        n_thind=metric_by_arm["no_struct"]["Thin-vessel Dice"], n_skel=metric_by_arm["no_struct"]["Skeleton Dice"],
        n_cl=metric_by_arm["no_struct"]["clDice"], n_bits=metric_by_arm["no_struct"]["mean_discrete_bitwidth_per_layer"],
        n_wbits=metric_by_arm["no_struct"]["parameter_weighted_mean_discrete_bitwidth"],
        n_bytes=metric_by_arm["no_struct"]["theoretical_conv_weight_payload_bytes"],
        n_comp=metric_by_arm["no_struct"]["theoretical_compression_ratio_vs_fp32_conv_weights"],
        d_thin=deltas["Thin-vessel recall"], d_thind=deltas["Thin-vessel Dice"],
        d_skel=deltas["Skeleton Dice"], d_cl=deltas["clDice"], d_auc=deltas["AUC"],
        d_dice=deltas["Dice/F1"],
    )
    (OUTPUT_DIR / "README.md").write_text(readme)
    print("Ablation report generated:", OUTPUT_DIR)
    print("no_struct minus full deltas:", deltas)
    print("Full best epoch:", full_checkpoint["epoch"], "no_struct best epoch:", no_checkpoint["epoch"])


if __name__ == "__main__":
    main()
