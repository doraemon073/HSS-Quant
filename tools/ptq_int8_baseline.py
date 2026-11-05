import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from lib.dataset import TestDataset
from lib.extract_patches import (
    get_data_test_overlap,
    kill_border,
    load_data,
    my_PreProc,
)
from lib.metrics import Evaluate
from lib.visualize import concat_result, save_img


PATCH_SIZE = 64
FP32_METRICS = {
    "AUC": 0.980615,
    "Dice/F1": 0.826093,
    "IoU": 0.703713,
    "Accuracy": 0.957390,
    "Sensitivity": 0.795075,
    "Specificity": 0.981064,
}


def parse_args():
    parser = argparse.ArgumentParser(description="DRIVE U-Net INT8 fake-quant PTQ baseline")
    parser.add_argument("--fp32-exp", default="experiments/DRIVE_Unet_FP32_full_retry")
    parser.add_argument("--output", default="experiments/DRIVE_Unet_INT8_PTQ_baseline")
    parser.add_argument("--calib-list", default="lists/DRIVE/train_images_seed2026.txt")
    parser.add_argument("--test-list", default="prepare_dataset/data_path_list/DRIVE/test.txt")
    parser.add_argument("--calib-patches", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Calibrate and smoke-test on synthetic inputs without creating output or reading test images",
    )
    return parser.parse_args()


def read_manifest(path):
    rows = []
    with open(path, "r") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError("Expected image/label/FOV triplet at {}:{}".format(path, line_number))
            rows.append(tuple(fields))
    if not rows:
        raise ValueError("Manifest is empty: {}".format(path))
    return rows


def validate_paths(rows, split):
    expected = {
        "training": ("/DRIVE/training/images/", "/DRIVE/training/1st_manual/", "/DRIVE/training/mask/"),
        "test": ("/DRIVE/test/images/", "/DRIVE/test/1st_manual/", "/DRIVE/test/mask/"),
    }[split]
    image_ids = set()
    for image_path, label_path, fov_path in rows:
        for path, marker in zip((image_path, label_path, fov_path), expected):
            if marker not in path:
                raise ValueError("{} manifest contains a non-{} path: {}".format(split, split, path))
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        image_id = Path(image_path).name.split("_", 1)[0]
        label_id = Path(label_path).name.split("_", 1)[0]
        fov_id = Path(fov_path).name.split("_", 1)[0]
        if not (image_id == label_id == fov_id):
            raise ValueError("Image, label, and FOV IDs do not match: {}".format(image_path))
        image_ids.add(image_id)
    return image_ids


def make_calibration_patches(calibration_manifest, patch_count, seed):
    raw_images, _, raw_fovs = load_data(str(calibration_manifest))
    images = my_PreProc(raw_images)
    fovs = raw_fovs // 255
    rng = np.random.RandomState(seed)
    patches = np.empty((patch_count, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    image_count, _, image_h, image_w = images.shape
    half = PATCH_SIZE // 2
    count = 0
    while count < patch_count:
        image_index = int(rng.randint(image_count))
        center_x = int(rng.randint(half, image_w - half + 1))
        center_y = int(rng.randint(half, image_h - half + 1))
        if fovs[image_index, 0, center_y, center_x] == 0:
            continue
        patches[count] = images[
            image_index,
            :,
            center_y - half:center_y + half,
            center_x - half:center_x + half,
        ]
        count += 1
    return patches


def affine_params(value_min, value_max, quant_min, quant_max):
    if value_max <= value_min:
        return 1.0, 0
    scale = max((value_max - value_min) / float(quant_max - quant_min), 1e-12)
    zero_point = int(np.clip(np.rint(quant_min - value_min / scale), quant_min, quant_max))
    return float(scale), zero_point


def collect_calibration_stats(model, patches, device, batch_size):
    activation_stats = {}
    weight_stats = {}
    hooks = []

    def capture_input(name):
        def hook(_module, inputs):
            values = inputs[0].detach().float()
            observed_min = float(values.min().item())
            observed_max = float(values.max().item())
            if name not in activation_stats:
                activation_stats[name] = {"min": observed_min, "max": observed_max}
            else:
                activation_stats[name]["min"] = min(activation_stats[name]["min"], observed_min)
                activation_stats[name]["max"] = max(activation_stats[name]["max"], observed_max)
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.detach().float()
            weight_stats[name] = {
                "min": float(weight.min().item()),
                "max": float(weight.max().item()),
            }
            hooks.append(layer.register_forward_pre_hook(capture_input(name)))

    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(patches), batch_size):
                batch = torch.from_numpy(patches[start:start + batch_size]).to(device)
                model(batch)
    finally:
        for hook in hooks:
            hook.remove()

    if not weight_stats or set(weight_stats) != set(activation_stats):
        raise RuntimeError("Calibration failed to observe every Conv2d input and weight")
    return activation_stats, weight_stats


class FakeQuantConv2d(nn.Module):
    def __init__(self, conv, input_stats, weight_stats):
        super().__init__()
        self.conv = conv
        self.activation_min = input_stats["min"]
        self.activation_max = input_stats["max"]
        self.activation_scale, self.activation_zero_point = affine_params(
            self.activation_min, self.activation_max, -128, 127
        )
        weight = conv.weight.detach().float()
        channel_absmax = weight.abs().amax(dim=(1, 2, 3))
        self.register_buffer("weight_scales", torch.clamp(channel_absmax / 127.0, min=1e-12))
        self.register_buffer("weight_zero_points", torch.zeros_like(channel_absmax, dtype=torch.int32))
        self.weight_min = weight_stats["min"]
        self.weight_max = weight_stats["max"]

    def quantized_weight(self):
        return torch.fake_quantize_per_channel_affine(
            self.conv.weight,
            self.weight_scales,
            self.weight_zero_points,
            0,
            quant_min=-127,
            quant_max=127,
        )

    def forward(self, inputs):
        quantized_inputs = torch.fake_quantize_per_tensor_affine(
            inputs,
            self.activation_scale,
            self.activation_zero_point,
            quant_min=-128,
            quant_max=127,
        )
        return F.conv2d(
            quantized_inputs,
            self.quantized_weight(),
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )

    def export(self):
        weight = self.conv.weight.detach().cpu().float()
        scales = self.weight_scales.detach().cpu().float()
        zero_points = self.weight_zero_points.detach().cpu().to(torch.int32)
        quantized_weight = torch.clamp(
            torch.round(weight / scales[:, None, None, None]), -127, 127
        ).to(torch.int8)
        return {
            "weight_int8": quantized_weight,
            "weight_scales": scales,
            "weight_zero_points": zero_points,
            "bias_fp32": None if self.conv.bias is None else self.conv.bias.detach().cpu().float(),
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "activation_min": self.activation_min,
            "activation_max": self.activation_max,
            "activation_scale": self.activation_scale,
            "activation_zero_point": self.activation_zero_point,
            "conv_parameters": {
                "stride": list(self.conv.stride),
                "padding": list(self.conv.padding),
                "dilation": list(self.conv.dilation),
                "groups": self.conv.groups,
            },
        }


def wrap_convolutions(module, activation_stats, weight_stats, prefix=""):
    wrapped = {}
    for child_name, child in list(module.named_children()):
        layer_name = "{}.{}".format(prefix, child_name) if prefix else child_name
        if isinstance(child, nn.Conv2d):
            wrapper = FakeQuantConv2d(child, activation_stats[layer_name], weight_stats[layer_name])
            setattr(module, child_name, wrapper)
            wrapped[layer_name] = wrapper
        else:
            wrapped.update(wrap_convolutions(child, activation_stats, weight_stats, layer_name))
    return wrapped


def evaluate_test_once(model, test_rows, output_dir, device, batch_size):
    test_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    test_images, test_masks, test_fovs = load_data(str(output_dir / "test_manifest_used.txt"))
    test_images = my_PreProc(test_images)
    test_masks = test_masks / 255.0
    test_fovs = test_fovs // 255

    from lib.extract_patches import paint_border_overlap, extract_ordered_overlap, recompone_overlap

    padded_images = paint_border_overlap(test_images, 96, 96, 16, 16)
    image_h, image_w = padded_images.shape[2:]
    patches = extract_ordered_overlap(padded_images, 96, 96, 16, 16)
    loader = DataLoader(TestDataset(patches), batch_size=batch_size, shuffle=False, num_workers=0)
    patch_predictions = []
    model.eval()
    with torch.inference_mode():
        for inputs in tqdm(loader, total=len(loader), desc="Final DRIVE test"):
            patch_predictions.append(model(inputs.to(device))[:, 1].cpu().numpy())
    predictions = np.expand_dims(np.concatenate(patch_predictions, axis=0), axis=1)
    pred_images = recompone_overlap(predictions, image_h, image_w, 16, 16)
    pred_images = pred_images[:, :, :test_images.shape[2], :test_images.shape[3]]

    fov_mask = test_fovs[:, 0] > 0
    y_true = test_masks[:, 0][fov_mask]
    y_scores = pred_images[:, 0][fov_mask]
    evaluator = Evaluate(save_path=str(output_dir))
    evaluator.add_batch(y_true, y_scores)
    metrics = evaluator.save_all_result(plot_curve=True, save_name="performance.txt")
    confusion, accuracy, specificity, sensitivity, precision = evaluator.confusion_matrix()
    true_positive = confusion[1, 1]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    metrics["Dice"] = 2 * true_positive / dice_denominator if dice_denominator else 0.0
    metrics["IoU"] = true_positive / iou_denominator if iou_denominator else 0.0
    metrics["Accuracy"] = accuracy
    metrics["Sensitivity"] = sensitivity
    metrics["Specificity"] = specificity
    metrics["Precision"] = precision
    with open(output_dir / "performance.txt", "a") as performance_file:
        performance_file.write(
            "\nDice: {}\nIoU: {}\nFOV pixels only: yes\nTest images: {}\n".format(
                metrics["Dice"], metrics["IoU"], len(test_images)
            )
        )
    np.save(output_dir / "result.npy", np.asarray([y_true, y_scores]))

    visible_predictions = pred_images.copy()
    kill_border(visible_predictions, test_fovs)
    prediction_dir = output_dir / "result_img"
    prediction_dir.mkdir(exist_ok=True)
    image_names = [Path(row[0]).stem for row in test_rows]
    for index, image_name in enumerate(image_names):
        composite = concat_result(test_images[index], visible_predictions[index], test_masks[index])
        save_img(composite, str(prediction_dir / "Result_{}.png".format(image_name)))

    metrics["test_images"] = len(test_images)
    metrics["test_seconds"] = time.perf_counter() - test_started
    metrics["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    metrics["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return metrics


def main():
    args = parse_args()
    project_root = PROJECT_ROOT
    os.chdir(project_root)
    fp32_experiment = Path(args.fp32_exp)
    if not fp32_experiment.is_absolute():
        fp32_experiment = project_root / fp32_experiment
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    calibration_manifest = Path(args.calib_list)
    if not calibration_manifest.is_absolute():
        calibration_manifest = project_root / calibration_manifest
    test_manifest = Path(args.test_list)
    if not test_manifest.is_absolute():
        test_manifest = project_root / test_manifest

    checkpoint_path = fp32_experiment / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("FP32 checkpoint not found: {}".format(checkpoint_path))
    if not calibration_manifest.is_file():
        raise FileNotFoundError("Calibration manifest not found: {}".format(calibration_manifest))
    if not test_manifest.is_file():
        raise FileNotFoundError("Official test manifest not found: {}".format(test_manifest))
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; refusing to overwrite: {}".format(output_dir))
    if args.calib_patches <= 0 or args.batch_size <= 0:
        raise ValueError("Calibration patch count and batch size must be positive")
    if str(output_dir.resolve()) == str(fp32_experiment.resolve()):
        raise ValueError("Output must not be the FP32 experiment directory")

    calibration_rows = read_manifest(calibration_manifest)
    test_rows = read_manifest(test_manifest)
    calibration_ids = validate_paths(calibration_rows, "training")
    test_ids = validate_paths(test_rows, "test")
    if len(calibration_rows) != 16 or len(test_rows) != 20:
        raise ValueError("Expected 16 train manifest rows and 20 official test rows")
    if calibration_ids & test_ids:
        raise ValueError("Train calibration and DRIVE test IDs overlap")
    if calibration_manifest.name != "train_images_seed2026.txt":
        raise ValueError("Calibration is restricted to train_images_seed2026.txt")
    if test_manifest.name != "test.txt" or "/data_path_list/DRIVE/" not in str(test_manifest):
        raise ValueError("Test evaluation must use the official DRIVE test manifest")
    forbidden_calibration_paths = [
        "/DRIVE/validation/",
        "/DRIVE/test/",
        "val_images_seed2026.txt",
        "test.txt",
    ]
    if any(token in str(calibration_manifest) for token in forbidden_calibration_paths):
        raise ValueError("Validation/test source detected in calibration manifest")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This baseline requires CUDA GPU 0")
    if device.index not in (None, 0):
        raise ValueError("Only GPU 0 is permitted")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch != 9:
        raise ValueError("Expected validation-best FP32 checkpoint from epoch 9; got {}".format(checkpoint_epoch))
    model = models.UNetFamily.U_Net(1, 2)
    model.load_state_dict(checkpoint["net"])
    fp32_model_state_bytes = sum(value.numel() * value.element_size() for value in checkpoint["net"].values())
    model.to(device).eval()
    del checkpoint

    print("FP32 checkpoint:", checkpoint_path)
    print("FP32 checkpoint epoch:", checkpoint_epoch)
    print("Calibration manifest:", calibration_manifest)
    print("Calibration images:", len(calibration_rows))
    print("Calibration patches:", args.calib_patches)
    print("Test manifest (reserved for final evaluation):", test_manifest)
    print("Calibration uses validation/test: no")

    calibration_started = time.perf_counter()
    patches = make_calibration_patches(calibration_manifest, args.calib_patches, args.seed)
    activation_stats, weight_stats = collect_calibration_stats(model, patches, device, args.batch_size)
    calibration_seconds = time.perf_counter() - calibration_started

    if args.calibration_only:
        quantized_model = models.UNetFamily.U_Net(1, 2)
        quantized_model.load_state_dict(model.state_dict())
        quantized_model.to(device).eval()
        wrapped = wrap_convolutions(quantized_model, activation_stats, weight_stats)
        with torch.inference_mode():
            smoke_output = quantized_model(torch.from_numpy(patches[:2]).to(device))
        print("Calibration-only smoke passed; synthetic batch output:", tuple(smoke_output.shape))
        print("Calibration Conv2d layers:", len(wrapped), "skipped: 0")
        print("Calibration seconds:", round(calibration_seconds, 2))
        return

    output_dir.mkdir(parents=True)
    quantized_layers = wrap_convolutions(model, activation_stats, weight_stats)
    if len(quantized_layers) != len(weight_stats):
        raise RuntimeError("One or more Conv2d layers were not wrapped")

    config = {
        "method": "INT8 fake quant PTQ, Conv2d input activations and weights",
        "real_int8_inference": False,
        "statement": "This is INT8 fake-quant PTQ using FP32 operations to simulate INT8 quantization error; it does not represent real hardware INT8 acceleration.",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "calibration_manifest": str(calibration_manifest),
        "calibration_images": len(calibration_rows),
        "calibration_patch_count": args.calib_patches,
        "calibration_patch_size": [PATCH_SIZE, PATCH_SIZE],
        "calibration_seed": args.seed,
        "calibration_batch_size": args.batch_size,
        "calibration_fov_rule": "sample patch center only inside training FOV",
        "validation_or_test_used_for_calibration": False,
        "test_manifest": str(test_manifest),
        "test_evaluated_once_after_calibration": True,
        "weight_quantization": "symmetric signed INT8, per-output-channel, range [-127,127]",
        "activation_quantization": "asymmetric signed INT8, per-Conv2d-input tensor, range [-128,127]",
        "other_operations": "FP32",
        "amp": False,
        "device": str(device),
        "quantized_layers": list(weight_stats),
        "skipped_layers": [],
    }
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    shutil.copy2(Path(__file__).resolve(), output_dir / "ptq_int8_baseline.py")
    np.save(output_dir / "calibration_patches.npy", patches)

    quantized_state = {name: layer.export() for name, layer in quantized_layers.items()}
    int8_state_path = output_dir / "int8_weight_state_dict.pt"
    torch.save({"checkpoint_epoch": checkpoint_epoch, "layers": quantized_state}, int8_state_path)
    ranges_report = {
        name: {
            "activation_input_min": activation_stats[name]["min"],
            "activation_input_max": activation_stats[name]["max"],
            "activation_scale": quantized_layers[name].activation_scale,
            "activation_zero_point": quantized_layers[name].activation_zero_point,
            "weight_min": weight_stats[name]["min"],
            "weight_max": weight_stats[name]["max"],
            "weight_scales_per_output_channel": quantized_layers[name].weight_scales.detach().cpu().tolist(),
            "weight_zero_points_per_output_channel": quantized_layers[name].weight_zero_points.detach().cpu().tolist(),
        }
        for name in weight_stats
    }
    with open(output_dir / "quant_layer_ranges.json", "w") as ranges_file:
        json.dump(ranges_report, ranges_file, indent=2)
    torch.save({"images": len(calibration_rows), "patches": len(patches), "patches_fp32": patches},
               output_dir / "calibration_data_info.pt")
    shutil.copy2(test_manifest, output_dir / "test_manifest_used.txt")

    fp32_checkpoint_bytes = checkpoint_path.stat().st_size
    int8_weight_bytes = sum(record["weight_int8"].numel() for record in quantized_state.values())
    int8_artifact_bytes = int8_state_path.stat().st_size
    size_report = {
        "fp32_checkpoint_file_bytes": fp32_checkpoint_bytes,
        "fp32_model_state_bytes": fp32_model_state_bytes,
        "int8_weight_payload_bytes": int8_weight_bytes,
        "int8_fake_quant_artifact_file_bytes": int8_artifact_bytes,
        "fp32_model_state_to_int8_weight_payload_ratio": fp32_model_state_bytes / int8_weight_bytes,
        "fp32_checkpoint_to_int8_artifact_ratio": fp32_checkpoint_bytes / int8_artifact_bytes,
        "note": "Fake quant retains FP32 execution. The weight-payload ratio is theoretical storage comparison, not a claim that the fake-quant model runs faster or that the artifact is a deployable integer model.",
    }
    with open(output_dir / "model_size.json", "w") as size_file:
        json.dump(size_report, size_file, indent=2)

    calibration_only_args = argparse.Namespace(**vars(args))
    calibration_only_args.test_manifest = test_manifest
    metrics = evaluate_test_once(model, test_rows, output_dir, device, args.batch_size)
    metrics["calibration_seconds"] = calibration_seconds
    comparison = []
    metric_pairs = [
        ("AUC", FP32_METRICS["AUC"], metrics["AUC_ROC"]),
        ("Dice/F1", FP32_METRICS["Dice/F1"], metrics["Dice"]),
        ("IoU", FP32_METRICS["IoU"], metrics["IoU"]),
        ("Accuracy", FP32_METRICS["Accuracy"], metrics["Accuracy"]),
        ("Sensitivity", FP32_METRICS["Sensitivity"], metrics["Sensitivity"]),
        ("Specificity", FP32_METRICS["Specificity"], metrics["Specificity"]),
    ]
    with open(output_dir / "fp32_vs_int8.csv", "w", newline="") as comparison_file:
        writer = csv.writer(comparison_file)
        writer.writerow(["metric", "FP32", "INT8_fake_quant", "delta_FP32_minus_INT8"])
        for metric_name, fp32_value, int8_value in metric_pairs:
            delta = fp32_value - int8_value
            comparison.append({"metric": metric_name, "FP32": fp32_value, "INT8": int8_value, "delta": delta})
            writer.writerow([metric_name, fp32_value, int8_value, delta])
    with open(output_dir / "comparison.json", "w") as comparison_file:
        json.dump(comparison, comparison_file, indent=2)

    with open(output_dir / "performance.txt", "a") as performance_file:
        performance_file.write("\nINT8 type: INT8 fake quant PTQ using FP32 execution; not real hardware INT8 inference or acceleration.\n")
        performance_file.write("Calibration manifest: {}\nCalibration patches: {}\n".format(
            calibration_manifest, args.calib_patches
        ))
        performance_file.write("Quantized Conv2d layers: {}; skipped: 0\n".format(len(quantized_layers)))
        performance_file.write("Calibration time (seconds): {:.2f}\nTest time (seconds): {:.2f}\n".format(
            calibration_seconds, metrics["test_seconds"]
        ))
        performance_file.write("Peak GPU memory (GiB allocated/reserved): {:.2f}/{:.2f}\n".format(
            metrics["gpu_peak_allocated_gib"], metrics["gpu_peak_reserved_gib"]
        ))
        performance_file.write("\nFP32 vs INT8 (delta = FP32 - INT8):\n")
        for row in comparison:
            performance_file.write("{}: FP32={}, INT8={}, delta={}\n".format(
                row["metric"], row["FP32"], row["INT8"], row["delta"]
            ))

    print("INT8 type: INT8 fake quant PTQ; FP32 compute, no real INT8 acceleration")
    print("Calibration patches/images:", args.calib_patches, "/", len(calibration_rows))
    print("Quantized Conv2d layers: {}; skipped: 0".format(len(quantized_layers)))
    print("Test metrics:", metrics)
    print("Model sizes:", size_report)
    print("INT8 evaluation results saved to:", output_dir)


if __name__ == "__main__":
    main()