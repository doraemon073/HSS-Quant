import csv
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from lib.extract_patches import load_file_path_txt
from tools.ptq_int8_baseline import evaluate_test_once, read_manifest, validate_paths
from tools.structure_metrics_from_saved_predictions import (
    analyze_prediction,
    load_saved_prediction,
    load_test_ground_truth_and_fov,
    reconstruct_fov_images,
)
from tools.structural_trainable_bits import (
    CHECKPOINT_RELATIVE,
    TRAIN_MANIFEST_RELATIVE,
    TrainableBitConv2d,
    replace_conv_layers,
)


OUTPUT_RELATIVE = Path("experiments/DRIVE_Unet_StructuralTrainableBits_v1_long/test_results")
BEST_CHECKPOINT_RELATIVE = Path("experiments/DRIVE_Unet_StructuralTrainableBits_v1_long/best_checkpoint.pt")
TEST_MANIFEST_RELATIVE = Path("prepare_dataset/data_path_list/DRIVE/test.txt")
BATCH_SIZE = 32
PATCH_INFERENCE_SECONDS_ESTIMATE = 620 / 46.74


class DiscreteFakeQuantConv2d(nn.Module):
    def __init__(self, trained_layer):
        super().__init__()
        self.layer = trained_layer
        self.bits = trained_layer.discrete_bits()

    def forward(self, inputs):
        if self.bits == 4:
            quantized_inputs = torch.fake_quantize_per_tensor_affine(
                inputs,
                self.layer.activation_scale_int4,
                self.layer.activation_zero_int4,
                -8,
                7,
            )
            quantized_weight = torch.fake_quantize_per_channel_affine(
                self.layer.conv.weight,
                self.layer.weight_scale_int4,
                self.layer.weight_zero_int4,
                0,
                -7,
                7,
            )
        else:
            quantized_inputs = torch.fake_quantize_per_tensor_affine(
                inputs,
                self.layer.activation_scale_int8,
                self.layer.activation_zero_int8,
                -128,
                127,
            )
            quantized_weight = torch.fake_quantize_per_channel_affine(
                self.layer.conv.weight,
                self.layer.weight_scale_int8,
                self.layer.weight_zero_int8,
                0,
                -127,
                127,
            )
        conv = self.layer.conv
        return F.conv2d(
            quantized_inputs,
            quantized_weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )


def replace_with_discrete_layers(model):
    hard_bits = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, TrainableBitConv2d):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        hard_bits[name] = module.discrete_bits()
        setattr(parent, child_name, DiscreteFakeQuantConv2d(module))
    if len(hard_bits) != 23:
        raise RuntimeError("Expected 23 discrete fake-quant Conv2d layers, got {}".format(len(hard_bits)))
    return hard_bits


def parse_performance(path):
    values = {}
    patterns = {
        "AUC": r"^(?:AUC ROC curve|AUC):\s*([0-9.eE+-]+)",
        "Dice/F1": r"^(?:F1 score|Dice_F1|Dice):\s*([0-9.eE+-]+)",
        "IoU": r"^IoU:\s*([0-9.eE+-]+)",
        "Accuracy": r"^Accuracy:\s*([0-9.eE+-]+)",
        "Sensitivity": r"^(?:Sensitivity\(SE\)|Sensitivity):\s*([0-9.eE+-]+)",
        "Specificity": r"^(?:Specificity\(SP\)|Specificity):\s*([0-9.eE+-]+)",
        "Thin-vessel recall": r"^Thin_vessel_recall:\s*([0-9.eE+-]+)",
        "Thin-vessel Dice": r"^Thin_vessel_Dice:\s*([0-9.eE+-]+)",
        "Skeleton Dice": r"^Skeleton_Dice:\s*([0-9.eE+-]+)",
        "clDice": r"^clDice:\s*([0-9.eE+-]+)",
        "Predicted connected components": r"^prediction_connected_components:\s*([0-9.eE+-]+)",
        "Uncovered skeleton fragments": r"^vessel_break_count:\s*([0-9.eE+-]+)",
        "Uncovered skeleton fragment rate": r"^vessel_break_rate:\s*([0-9.eE+-]+)",
        "Test time (s)": r"^Test time \(seconds\):\s*([0-9.eE+-]+)",
        "Test time (s) alternate": r"^test_seconds:\s*([0-9.eE+-]+)",
        "Test time (s) fp32": r"^Test evaluation time:\s*([0-9.eE+-]+)\s+seconds",
        "GPU peak allocated (GiB)": r"^gpu_peak_allocated_gib:\s*([0-9.eE+-]+)",
        "GPU peak allocated (GiB) alternate": r"^Peak GPU memory \(GiB allocated/reserved\):\s*([0-9.eE+-]+)",
        "GPU peak allocated (GiB) fp32": r"^Peak GPU memory:\s*([0-9.eE+-]+)\s+GiB allocated",
    }
    for line in path.read_text().splitlines():
        for name, pattern in patterns.items():
            match = re.search(pattern, line.strip())
            if match:
                key = name.replace(" alternate", "").replace(" fp32", "")
                values[key] = float(match.group(1))
    return values


def load_structure_comparison(path):
    result = {name: {} for name in ("FP32", "INT8", "INT4", "SA-INT4")}
    column_names = {"FP32": "FP32", "INT8": "INT8", "INT4": "INT4", "SA-INT4": "INT4_SA"}
    structure_names = {
        "thin_vessel_recall": "Thin-vessel recall",
        "thin_vessel_dice": "Thin-vessel Dice",
        "skeleton_dice": "Skeleton Dice",
        "clDice": "clDice",
        "prediction_connected_components": "Predicted connected components",
        "vessel_break_count": "Uncovered skeleton fragments",
        "vessel_break_rate": "Uncovered skeleton fragment rate",
    }
    with open(path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            metric = row["metric"]
            if metric == "overall_dice":
                target = "Dice/F1"
            elif metric == "overall_iou":
                target = "IoU"
            else:
                target = structure_names.get(metric)
            if target is None:
                continue
            for model_name, source_column in column_names.items():
                result[model_name][target] = float(row[source_column])
    return result


def add_group_structure_metrics(metrics, performance_path):
    parsed = parse_performance(performance_path)
    for name in (
        "Thin-vessel recall", "Thin-vessel Dice", "Skeleton Dice", "clDice",
        "Predicted connected components", "Uncovered skeleton fragments",
        "Uncovered skeleton fragment rate",
    ):
        if name in parsed:
            metrics[name] = parsed[name]


def finalize_saved_evaluation():
    output_dir = PROJECT_ROOT / OUTPUT_RELATIVE
    if not output_dir.is_dir():
        raise FileNotFoundError("No saved one-pass test predictions found: {}".format(output_dir))
    required = [
        output_dir / "result.npy",
        output_dir / "test_manifest_used.txt",
        output_dir / "quant_config.json",
        output_dir / "performance.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_StructuralTrainableBits_v1_long/layer_mapping.json",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/fp32_int8_int4_sa_structure_comparison.csv",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_LayerSensitivity/encoder_int8_decoder_int4/layer_assignment.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Saved evaluation output missing: {}".format(missing))

    test_rows = read_manifest(output_dir / "test_manifest_used.txt")
    if len(test_rows) != 20:
        raise ValueError("Expected the saved manifest for all 20 official test images")
    manifest_columns = list(zip(*test_rows))
    ground_truths, fovs = load_test_ground_truth_and_fov(manifest_columns)
    saved_truth, saved_scores = load_saved_prediction(output_dir / "result.npy")
    prediction_maps, truth_maps = reconstruct_fov_images(
        saved_truth, saved_scores, ground_truths, fovs, pixel_order="C"
    )
    structure_metrics = analyze_prediction(prediction_maps, truth_maps, fovs)
    if len(list((output_dir / "result_img").glob("*.png"))) != 20:
        raise ValueError("Expected 20 prediction visualizations from the single inference pass")

    overall_metrics = parse_performance(output_dir / "performance.txt")
    required_metrics = ("AUC", "Dice/F1", "IoU", "Accuracy", "Sensitivity", "Specificity")
    if any(name not in overall_metrics for name in required_metrics):
        raise ValueError("Saved test performance file is missing an overall metric")
    probability = json.loads((output_dir / "quant_config.json").read_text())
    layer_mapping = json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_StructuralTrainableBits_v1_long/layer_mapping.json").read_text())
    discrete_bits = probability["per_layer_discrete_bit_width"]
    probabilities = probability["per_layer_probability_int8"]
    if len(discrete_bits) != 23 or len(probabilities) != 23:
        raise ValueError("Saved evaluation configuration must contain 23 learned layer assignments")
    total_conv_weight_numel = sum(item["weight_numel"] for item in layer_mapping.values())
    weighted_bits = sum(layer_mapping[name]["weight_numel"] * int(bits)
                        for name, bits in discrete_bits.items())
    theoretical_bytes = int(math.ceil(weighted_bits / 8.0))
    fp32_conv_weight_bytes = total_conv_weight_numel * 4
    mean_discrete_bits = float(np.mean(list(discrete_bits.values())))
    mean_continuous_bits = float(np.mean([4.0 + 4.0 * float(p) for p in probabilities.values()]))
    weighted_discrete_bits = weighted_bits / total_conv_weight_numel

    structure_csv = PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/fp32_int8_int4_sa_structure_comparison.csv"
    structures = load_structure_comparison(structure_csv)
    performance_paths = {
        "FP32": PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/performance.txt",
        "INT8": PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/performance.txt",
        "INT4": PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/performance.txt",
        "SA-INT4": PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/performance.txt",
    }
    for model_name, path in performance_paths.items():
        structures[model_name].update(parse_performance(path))
    structures["FP32"].update(parse_performance(
        PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/test_log.txt"
    ))
    mixed_name = "Encoder INT8 + Decoder INT4"
    mixed_path = PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_LayerSensitivity/encoder_int8_decoder_int4/performance.txt"
    structures[mixed_name] = parse_performance(mixed_path)
    add_group_structure_metrics(structures[mixed_name], mixed_path)

    learned_name = "StructuralTrainableBits"
    learned_metrics = dict(overall_metrics)
    learned_metrics.update({
        "Thin-vessel recall": structure_metrics["thin_vessel_recall"],
        "Thin-vessel Dice": structure_metrics["thin_vessel_dice"],
        "Skeleton Dice": structure_metrics["skeleton_dice"],
        "clDice": structure_metrics["clDice"],
        "Predicted connected components": structure_metrics["prediction_connected_components"],
        "Uncovered skeleton fragments": structure_metrics["uncovered_skeleton_fragment_count"],
        "Uncovered skeleton fragment rate": structure_metrics["vessel_break_rate"],
        "Test time (s)": PATCH_INFERENCE_SECONDS_ESTIMATE,
        "GPU peak allocated (GiB)": None,
    })
    structures[learned_name] = learned_metrics

    fp32_bytes = fp32_conv_weight_bytes
    int8_bytes = json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/model_size.json").read_text())["int8_weight_payload_bytes"]
    int4_bytes = json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/model_size.json").read_text())["int4_theoretical_weight_payload_bytes"]
    sa_bytes = json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/model_size.json").read_text())["int4_theoretical_weight_payload_bytes"]
    assignment_path = PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_LayerSensitivity/encoder_int8_decoder_int4/layer_assignment.json"
    assignment = json.loads(assignment_path.read_text())["layer_bits"]
    if set(assignment) != set(layer_mapping):
        raise ValueError("Mixed-precision assignment does not contain the same 23 Conv2d layers")
    mixed_bits = sum(layer_mapping[name]["weight_numel"] * int(bits) for name, bits in assignment.items())
    mixed_bytes = int(math.ceil(mixed_bits / 8.0))
    sizes = {
        "FP32": (32.0, fp32_bytes),
        "INT8": (8.0, int8_bytes),
        "INT4": (4.0, int4_bytes),
        "SA-INT4": (4.0, sa_bytes),
        mixed_name: (float(np.mean(list(assignment.values()))), mixed_bytes),
        learned_name: (mean_discrete_bits, theoretical_bytes),
    }

    rows = []
    for model_name in ("FP32", "INT8", "INT4", "SA-INT4", mixed_name, learned_name):
        mean_bits, payload_bytes = sizes[model_name]
        metrics = structures[model_name]
        rows.append({
            "model": model_name,
            "AUC": metrics.get("AUC"),
            "Dice/F1": metrics.get("Dice/F1"),
            "IoU": metrics.get("IoU"),
            "Accuracy": metrics.get("Accuracy"),
            "Sensitivity": metrics.get("Sensitivity"),
            "Specificity": metrics.get("Specificity"),
            "Thin-vessel recall": metrics.get("Thin-vessel recall"),
            "Thin-vessel Dice": metrics.get("Thin-vessel Dice"),
            "Skeleton Dice": metrics.get("Skeleton Dice"),
            "clDice": metrics.get("clDice"),
            "Predicted connected components": metrics.get("Predicted connected components"),
            "Uncovered skeleton fragments": metrics.get("Uncovered skeleton fragments"),
            "Uncovered skeleton fragment rate": metrics.get("Uncovered skeleton fragment rate"),
            "Mean discrete bit-width per Conv2d layer": mean_bits,
            "Theoretical Conv2d weight payload bytes": payload_bytes,
            "Theoretical compression ratio vs FP32 Conv2d weights": fp32_bytes / payload_bytes,
            "Test time (s)": metrics.get("Test time (s)"),
            "GPU peak allocated (GiB)": metrics.get("GPU peak allocated (GiB)"),
        })
    comparison_path = output_dir / "comparison.csv"
    with open(comparison_path, "w", newline="") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config_path = output_dir / "quant_config.json"
    config = json.loads(config_path.read_text())
    config.update({
        "test_metrics": overall_metrics,
        "structure_metrics": structure_metrics,
        "test_elapsed_seconds": None,
        "patch_inference_seconds_estimate_from_progress": PATCH_INFERENCE_SECONDS_ESTIMATE,
        "gpu_peak_allocated_gib": None,
        "gpu_peak_reserved_gib": None,
        "efficiency_measurement_note": (
            "Inference ran once and saved predictions. Structural postprocessing then raised a manifest-shape error; "
            "the process lost its in-memory test timer and CUDA peak counters before they were written. "
            "The 13.27 s value is only the approximate 620-batch inference-loop duration inferred from tqdm; "
            "GPU peak memory is unavailable. No second inference was run."
        ),
        "mean_discrete_bit_width_layer_average": mean_discrete_bits,
        "mean_continuous_bit_width_layer_average": mean_continuous_bits,
        "parameter_weighted_mean_discrete_bit_width": weighted_discrete_bits,
        "theoretical_conv_weight_payload_bytes": theoretical_bytes,
        "fp32_conv_weight_payload_bytes": fp32_bytes,
        "theoretical_compression_ratio_vs_fp32_conv_weights": fp32_bytes / theoretical_bytes,
    })
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)

    with open(output_dir / "performance.txt", "a") as performance_file:
        performance_file.write("\nStructuralTrainableBits structure analysis from saved predictions; no additional inference.\n")
        performance_file.write("Thin-vessel recall: {:.9f}\nThin-vessel Dice: {:.9f}\n".format(
            structure_metrics["thin_vessel_recall"], structure_metrics["thin_vessel_dice"]
        ))
        performance_file.write("Skeleton Dice: {:.9f}\nclDice: {:.9f}\n".format(
            structure_metrics["skeleton_dice"], structure_metrics["clDice"]
        ))
        performance_file.write("Predicted connected components: {}\nUncovered skeleton fragments: {}\n".format(
            structure_metrics["prediction_connected_components"],
            structure_metrics["uncovered_skeleton_fragment_count"],
        ))
        performance_file.write("Uncovered skeleton fragment rate: {:.9f}\n".format(
            structure_metrics["vessel_break_rate"]
        ))
        performance_file.write("Mean discrete/continuous bit-width: {:.6f}/{:.6f}\n".format(
            mean_discrete_bits, mean_continuous_bits
        ))
        performance_file.write("Theoretical Conv2d weight payload: {} bytes; compression ratio: {:.6f}x\n".format(
            theoretical_bytes, fp32_bytes / theoretical_bytes
        ))
        performance_file.write("Approximate patch inference loop from tqdm: {:.3f} seconds; full elapsed time unavailable after postprocessing exception.\n".format(
            PATCH_INFERENCE_SECONDS_ESTIMATE
        ))
        performance_file.write("GPU peak memory: unavailable; CUDA peak counters were lost when the one-pass process exited after postprocessing failure.\n")
        performance_file.write("No training, recalibration, or second inference was performed.\n")

    print("Saved-prediction postprocessing complete; no inference was performed")
    print("Overall metrics:", overall_metrics)
    print("Structure metrics:", structure_metrics)
    print("Mean discrete/continuous bits:", mean_discrete_bits, mean_continuous_bits)
    print("Theoretical Conv2d weight payload/compression:", theoretical_bytes, fp32_bytes / theoretical_bytes)
    print("Approximate patch inference loop seconds:", PATCH_INFERENCE_SECONDS_ESTIMATE)
    print("GPU peak memory: unavailable; not recovered by another test run")
    print("Comparison:", comparison_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-saved-only", action="store_true")
    parser.add_argument(
        "--experiment-dir",
        default="experiments/DRIVE_Unet_StructuralTrainableBits_v1_long",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.finalize_saved_only:
        finalize_saved_evaluation()
        return
    experiment_dir = Path(args.experiment_dir).expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = PROJECT_ROOT / experiment_dir
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else experiment_dir / "test_results"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing test results: {}".format(output_dir))

    best_path = experiment_dir / "best_checkpoint.pt"
    base_path = PROJECT_ROOT / CHECKPOINT_RELATIVE
    test_manifest = PROJECT_ROOT / TEST_MANIFEST_RELATIVE
    train_manifest = PROJECT_ROOT / TRAIN_MANIFEST_RELATIVE
    structural_root = PROJECT_ROOT / "experiments/DRIVE_Unet_StructuralTrainableBits_v1_long"
    required_files = [
        best_path,
        base_path,
        test_manifest,
        train_manifest,
        PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/performance.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/test_log.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/performance.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/model_size.json",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/performance.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/model_size.json",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/performance.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/model_size.json",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/fp32_int8_int4_sa_structure_comparison.csv",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_LayerSensitivity/encoder_int8_decoder_int4/performance.txt",
        PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_LayerSensitivity/encoder_int8_decoder_int4/layer_assignment.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required evaluation input missing: {}".format(missing))

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Official final evaluation requires CUDA GPU 0")
    torch.cuda.set_device(device)

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    best_epoch = int(best.get("epoch", -1))
    if not 1 <= best_epoch <= 20:
        raise ValueError("Expected a best checkpoint from the fixed 20-epoch run")
    fp32_path = Path(best["fp32_checkpoint"])
    if not fp32_path.is_absolute():
        fp32_path = PROJECT_ROOT / fp32_path
    if fp32_path.resolve() != base_path.resolve():
        raise ValueError("Best checkpoint does not reference the expected FP32 checkpoint")

    test_rows = read_manifest(test_manifest)
    test_ids = validate_paths(test_rows, "test")
    if len(test_rows) != 20 or len(test_ids) != 20:
        raise ValueError("Expected 20 unique images from the official DRIVE test manifest")
    train_rows = read_manifest(train_manifest)
    train_ids = validate_paths(train_rows, "training")
    if train_ids & test_ids:
        raise ValueError("Train and official test image IDs overlap")

    if len(best["p_l"]) != 23 or len(best["bit_logits"]) != 23:
        raise ValueError("Best checkpoint must contain all 23 learned bit logits")
    if set(best["p_l"]) != set(best["activation_calibration_ranges"]):
        raise ValueError("Saved activation calibration ranges do not match learned Conv2d layers")
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    if int(base_checkpoint.get("epoch", -1)) != 9:
        raise ValueError("Expected the fixed epoch-9 FP32 source checkpoint")

    model = models.UNetFamily.U_Net(1, 2)
    model.load_state_dict(base_checkpoint["net"])
    fp32_model_state_bytes = sum(value.numel() * value.element_size() for value in base_checkpoint["net"].values())
    trainable_layers, layer_mapping = replace_conv_layers(model, best["activation_calibration_ranges"])
    for name, wrapper in trainable_layers.items():
        wrapper.bit_logit.data.copy_(best["bit_logits"][name].reshape_as(wrapper.bit_logit))
        probability = float(wrapper.bit_probability().detach())
        if abs(probability - float(best["p_l"][name])) > 1e-6:
            raise ValueError("Saved bit logit/probability mismatch for {}".format(name))
    active_bits = replace_with_discrete_layers(model)
    if active_bits != best["discrete_bit_width"]:
        raise ValueError("Hard INT4/INT8 assignment does not match the saved checkpoint")
    model.to(device).eval()
    del base_checkpoint

    # This helper reads the manifest copy from its output directory, so create it only after all guards pass.
    output_dir.mkdir(parents=True)
    with open(output_dir / "test_manifest_used.txt", "w") as manifest_file:
        for row in test_rows:
            manifest_file.write("{} {} {}\n".format(*row))

    discrete_bits = {name: int(bits) for name, bits in active_bits.items()}
    probability_by_layer = {name: float(best["p_l"][name]) for name in active_bits}
    total_conv_weight_numel = sum(details["weight_numel"] for details in layer_mapping.values())
    quantized_weight_bits = sum(
        layer_mapping[name]["weight_numel"] * discrete_bits[name] for name in discrete_bits
    )
    theoretical_weight_bytes = int(math.ceil(quantized_weight_bits / 8.0))
    fp32_conv_weight_bytes = total_conv_weight_numel * 4
    mean_layer_bits = float(np.mean(list(discrete_bits.values())))
    weighted_mean_bits = quantized_weight_bits / total_conv_weight_numel
    mean_continuous_bits = float(np.mean([4.0 + 4.0 * probability for probability in probability_by_layer.values()]))
    config = {
        "method": "StructuralTrainableBits learned discrete per-Conv2d fake quant final evaluation",
        "best_checkpoint": str(best_path),
        "best_epoch": int(best["epoch"]),
        "validation_metrics_at_best_epoch": best["validation_metrics"],
        "fp32_source_checkpoint": str(base_path),
        "fp32_source_epoch": 9,
        "official_test_manifest": str(test_manifest),
        "test_images": len(test_rows),
        "test_image_ids": sorted(test_ids, key=lambda value: int(value) if value.isdigit() else value),
        "train_or_test_calibration_performed": False,
        "training_performed": False,
        "recalibration_performed": False,
        "bit_widths_modified": False,
        "inference_passes": 1,
        "fov_only_pixel_metrics": True,
        "inference_bit_rule": "Use checkpoint p_l unchanged; p_l >= 0.5 selects hard INT8 fake-quant branch, otherwise hard INT4 branch.",
        "inference_type": "Fake quantization with FP32 arithmetic; not actual integer hardware INT4/INT8 inference.",
        "weight_quantization": "INT4 [-7,7] and INT8 [-127,127], symmetric per output channel",
        "activation_quantization": "INT4 [-8,7] and INT8 [-128,127], affine per Conv2d input using saved training calibration ranges",
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "mean_discrete_bit_width_layer_average": mean_layer_bits,
        "mean_continuous_bit_width_layer_average": mean_continuous_bits,
        "parameter_weighted_mean_discrete_bit_width": weighted_mean_bits,
        "per_layer_probability_int8": probability_by_layer,
        "per_layer_discrete_bit_width": discrete_bits,
        "theoretical_conv_weight_payload_bytes": theoretical_weight_bytes,
        "fp32_conv_weight_payload_bytes": fp32_conv_weight_bytes,
        "theoretical_compression_ratio_vs_fp32_conv_weights": fp32_conv_weight_bytes / theoretical_weight_bytes,
        "theoretical_size_definition": "Packed Conv2d weight payload only: sum(weight_numel * selected_bits)/8; excludes scales, biases, activations, and FP32 non-Conv2d state.",
    }
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)

    # Exactly one model inference on the official test split.
    test_metrics = evaluate_test_once(model, test_rows, output_dir, device, BATCH_SIZE)

    ground_truths, fovs = load_test_ground_truth_and_fov(list(zip(*test_rows)))
    saved_truth, saved_scores = load_saved_prediction(output_dir / "result.npy")
    prediction_maps, truth_maps = reconstruct_fov_images(
        saved_truth, saved_scores, ground_truths, fovs, pixel_order="C"
    )
    structure_metrics = analyze_prediction(prediction_maps, truth_maps, fovs)

    report_path = output_dir / "performance.txt"
    with open(report_path, "a") as report_file:
        report_file.write("\nStructuralTrainableBits final test evaluation\n")
        report_file.write("Selected checkpoint: {} (epoch {})\n".format(best_path, best["epoch"]))
        report_file.write("Test manifest: {}\nOfficial test images: 20; inference passes: 1\n".format(test_manifest))
        report_file.write("FOV-only metrics: yes\nFake quant with FP32 arithmetic; no real integer hardware inference.\n")
        report_file.write("AUC: {:.9f}\nDice/F1: {:.9f}\nIoU: {:.9f}\nAccuracy: {:.9f}\n".format(
            test_metrics["AUC_ROC"], test_metrics["Dice"], test_metrics["IoU"], test_metrics["Accuracy"]
        ))
        report_file.write("Sensitivity: {:.9f}\nSpecificity: {:.9f}\n".format(
            test_metrics["Sensitivity"], test_metrics["Specificity"]
        ))
        report_file.write("Thin-vessel recall: {:.9f}\nThin-vessel Dice: {:.9f}\n".format(
            structure_metrics["thin_vessel_recall"], structure_metrics["thin_vessel_dice"]
        ))
        report_file.write("Skeleton Dice: {:.9f}\nclDice: {:.9f}\n".format(
            structure_metrics["skeleton_dice"], structure_metrics["clDice"]
        ))
        report_file.write("Predicted connected components: {}\nUncovered skeleton fragments: {}\n".format(
            structure_metrics["prediction_connected_components"],
            structure_metrics["uncovered_skeleton_fragment_count"],
        ))
        report_file.write("Uncovered skeleton fragment rate: {:.9f}\n".format(
            structure_metrics["vessel_break_rate"]
        ))
        report_file.write("Mean layer discrete bit-width: {:.6f}\nMean layer continuous bit-width: {:.6f}\n".format(
            mean_layer_bits, mean_continuous_bits
        ))
        report_file.write("Theoretical Conv2d weight payload: {} bytes; compression ratio vs FP32 Conv2d weights: {:.6f}x\n".format(
            theoretical_weight_bytes, fp32_conv_weight_bytes / theoretical_weight_bytes
        ))
        report_file.write("Test inference time: {:.6f} seconds\nGPU peak allocated: {:.6f} GiB\nGPU peak reserved: {:.6f} GiB\n".format(
            test_metrics["test_seconds"], test_metrics["gpu_peak_allocated_gib"], test_metrics["gpu_peak_reserved_gib"]
        ))

    structure_csv = PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/fp32_int8_int4_sa_structure_comparison.csv"
    structures = load_structure_comparison(structure_csv)
    for model_name in ("FP32", "INT8", "INT4", "SA-INT4"):
        perf_path = {
            "FP32": PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/performance.txt",
            "INT8": PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/performance.txt",
            "INT4": PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/performance.txt",
            "SA-INT4": PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/performance.txt",
        }[model_name]
        structures[model_name].update(parse_performance(perf_path))
    fp32_test_log = PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/test_log.txt"
    structures["FP32"].update(parse_performance(fp32_test_log))

    mixed_name = "Encoder INT8 + Decoder INT4"
    mixed_root = PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_LayerSensitivity/encoder_int8_decoder_int4"
    mixed_metrics = parse_performance(mixed_root / "performance.txt")
    add_group_structure_metrics(mixed_metrics, mixed_root / "performance.txt")
    structures[mixed_name] = mixed_metrics
    learned_name = "StructuralTrainableBits"
    learned_metrics = {
        "AUC": float(test_metrics["AUC_ROC"]),
        "Dice/F1": float(test_metrics["Dice"]),
        "IoU": float(test_metrics["IoU"]),
        "Accuracy": float(test_metrics["Accuracy"]),
        "Sensitivity": float(test_metrics["Sensitivity"]),
        "Specificity": float(test_metrics["Specificity"]),
        "Thin-vessel recall": structure_metrics["thin_vessel_recall"],
        "Thin-vessel Dice": structure_metrics["thin_vessel_dice"],
        "Skeleton Dice": structure_metrics["skeleton_dice"],
        "clDice": structure_metrics["clDice"],
        "Predicted connected components": structure_metrics["prediction_connected_components"],
        "Uncovered skeleton fragments": structure_metrics["uncovered_skeleton_fragment_count"],
        "Uncovered skeleton fragment rate": structure_metrics["vessel_break_rate"],
        "Test time (s)": test_metrics["test_seconds"],
        "GPU peak allocated (GiB)": test_metrics["gpu_peak_allocated_gib"],
    }
    structures[learned_name] = learned_metrics

    fp32_conv_weights = total_conv_weight_numel * 4
    sizes = {
        "FP32": (32, fp32_conv_weights),
        "INT8": (8, json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/model_size.json").read_text())["int8_weight_payload_bytes"]),
        "INT4": (4, json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/model_size.json").read_text())["int4_theoretical_weight_payload_bytes"]),
        "SA-INT4": (4, json.loads((PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_SA_Calib/model_size.json").read_text())["int4_theoretical_weight_payload_bytes"]),
    }
    assignment_path = mixed_root / "layer_assignment.json"
    assignment = json.loads(assignment_path.read_text())
    assignment_bits = assignment["layer_bits"]
    if set(assignment_bits) != set(layer_mapping):
        raise ValueError("Hand-designed mixed assignment layer names do not match the U-Net")
    mixed_weight_bits = sum(layer_mapping[name]["weight_numel"] * int(bits)
                            for name, bits in assignment_bits.items())
    mixed_weight_bytes = int(math.ceil(mixed_weight_bits / 8.0))
    sizes[mixed_name] = (float(np.mean(list(assignment_bits.values()))), mixed_weight_bytes)
    sizes[learned_name] = (mean_layer_bits, theoretical_weight_bytes)

    history = []
    for name in ("FP32", "INT8", "INT4", "SA-INT4", mixed_name, learned_name):
        bit_width, weight_bytes = sizes[name]
        metric = structures[name]
        history.append({
            "model": name,
            "AUC": metric.get("AUC"),
            "Dice/F1": metric.get("Dice/F1"),
            "IoU": metric.get("IoU"),
            "Accuracy": metric.get("Accuracy"),
            "Sensitivity": metric.get("Sensitivity"),
            "Specificity": metric.get("Specificity"),
            "Thin-vessel recall": metric.get("Thin-vessel recall"),
            "Thin-vessel Dice": metric.get("Thin-vessel Dice"),
            "Skeleton Dice": metric.get("Skeleton Dice"),
            "clDice": metric.get("clDice"),
            "Predicted connected components": metric.get("Predicted connected components"),
            "Uncovered skeleton fragments": metric.get("Uncovered skeleton fragments"),
            "Uncovered skeleton fragment rate": metric.get("Uncovered skeleton fragment rate"),
            "Mean discrete bit-width per Conv2d layer": bit_width,
            "Theoretical Conv2d weight payload bytes": weight_bytes,
            "Theoretical compression ratio vs FP32 Conv2d weights": fp32_conv_weights / weight_bytes,
            "Test time (s)": metric.get("Test time (s)"),
            "GPU peak allocated (GiB)": metric.get("GPU peak allocated (GiB)"),
        })
    comparison_path = output_dir / "comparison.csv"
    with open(comparison_path, "w", newline="") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    config["test_metrics"] = learned_metrics
    config["structure_metrics"] = structure_metrics
    config["test_seconds"] = float(test_metrics["test_seconds"])
    config["gpu_peak_allocated_gib"] = float(test_metrics["gpu_peak_allocated_gib"])
    config["gpu_peak_reserved_gib"] = float(test_metrics["gpu_peak_reserved_gib"])
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)

    print("Official DRIVE test evaluated once using best checkpoint:", best_path)
    print("StructuralTrainableBits metrics:", learned_metrics)
    print("Structural metrics:", structure_metrics)
    print("Mean discrete/continuous bits:", mean_layer_bits, mean_continuous_bits)
    print("Theoretical Conv2d weight payload bytes/compression:", theoretical_weight_bytes,
          fp32_conv_weights / theoretical_weight_bytes)
    print("Comparison CSV:", comparison_path)
    print("Prediction images:", len(list((output_dir / "result_img").glob("*.png"))))
    print("No training or recalibration performed; fake quant with FP32 arithmetic")


if __name__ == "__main__":
    main()