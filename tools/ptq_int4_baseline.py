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
PATCH_SIZE = 64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from tools.ptq_int8_baseline import (
    FP32_METRICS,
    collect_calibration_stats,
    evaluate_test_once,
    make_calibration_patches,
    read_manifest,
    validate_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DRIVE U-Net INT4 fake-quant PTQ baseline")
    parser.add_argument("--fp32-exp", default="experiments/DRIVE_Unet_FP32_full_retry")
    parser.add_argument("--output", default="experiments/DRIVE_Unet_INT4_PTQ_baseline")
    parser.add_argument("--calib-list", default="lists/DRIVE/train_images_seed2026.txt")
    parser.add_argument("--test-list", default="prepare_dataset/data_path_list/DRIVE/test.txt")
    parser.add_argument("--calib-patches", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-only", action="store_true")
    return parser.parse_args()


def affine_params(value_min, value_max, quant_min=-8, quant_max=7):
    if value_max <= value_min:
        return 1.0, 0
    scale = max((value_max - value_min) / float(quant_max - quant_min), 1e-12)
    zero_point = int(np.clip(np.rint(quant_min - value_min / scale), quant_min, quant_max))
    return float(scale), zero_point


def pack_signed_int4(values):
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    n_values = flat.size
    if n_values % 2:
        flat = np.append(flat, np.int8(0))
    nibbles = (flat.astype(np.int16) & 0x0F).astype(np.uint8)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed, n_values


class FakeQuantConv2dInt4(nn.Module):
    def __init__(self, conv, activation_stats, weight_stats):
        super().__init__()
        self.conv = conv
        self.activation_min = activation_stats["min"]
        self.activation_max = activation_stats["max"]
        self.activation_scale, self.activation_zero_point = affine_params(
            self.activation_min, self.activation_max
        )
        weights = conv.weight.detach().float()
        per_channel_absmax = weights.abs().amax(dim=(1, 2, 3))
        self.register_buffer("weight_scales", torch.clamp(per_channel_absmax / 7.0, min=1e-12))
        self.register_buffer("weight_zero_points", torch.zeros_like(per_channel_absmax, dtype=torch.int32))
        self.weight_min = weight_stats["min"]
        self.weight_max = weight_stats["max"]

    def quantized_weight(self):
        return torch.fake_quantize_per_channel_affine(
            self.conv.weight,
            self.weight_scales,
            self.weight_zero_points,
            0,
            quant_min=-7,
            quant_max=7,
        )

    def forward(self, inputs):
        quantized_inputs = torch.fake_quantize_per_tensor_affine(
            inputs,
            self.activation_scale,
            self.activation_zero_point,
            quant_min=-8,
            quant_max=7,
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
        weights = self.conv.weight.detach().cpu().float()
        scales = self.weight_scales.detach().cpu().float()
        zero_points = self.weight_zero_points.detach().cpu().to(torch.int32)
        quantized = torch.clamp(torch.round(weights / scales[:, None, None, None]), -7, 7).to(torch.int8)
        packed, value_count = pack_signed_int4(quantized.numpy())
        return {
            "weight_int4_packed": torch.from_numpy(packed),
            "weight_value_count": value_count,
            "weight_shape": list(weights.shape),
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
            wrapper = FakeQuantConv2dInt4(child, activation_stats[layer_name], weight_stats[layer_name])
            setattr(module, child_name, wrapper)
            wrapped[layer_name] = wrapper
        else:
            wrapped.update(wrap_convolutions(child, activation_stats, weight_stats, layer_name))
    return wrapped


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    fp32_experiment = resolve_path(args.fp32_exp)
    output_dir = resolve_path(args.output)
    calibration_manifest = resolve_path(args.calib_list)
    test_manifest = resolve_path(args.test_list)
    checkpoint_path = fp32_experiment / "best_model.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError("FP32 checkpoint not found: {}".format(checkpoint_path))
    if not calibration_manifest.is_file() or not test_manifest.is_file():
        raise FileNotFoundError("Calibration or official DRIVE test manifest is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is non-empty; refusing to overwrite: {}".format(output_dir))
    if output_dir.resolve() in (fp32_experiment.resolve(),
                                (PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline").resolve()):
        raise ValueError("INT4 output must not overwrite FP32 or INT8 experiment directories")
    if args.calib_patches <= 0 or args.batch_size <= 0:
        raise ValueError("Calibration patch count and batch size must be positive")

    calibration_rows = read_manifest(calibration_manifest)
    test_rows = read_manifest(test_manifest)
    calibration_ids = validate_paths(calibration_rows, "training")
    test_ids = validate_paths(test_rows, "test")
    if len(calibration_rows) != 16 or len(test_rows) != 20 or calibration_ids & test_ids:
        raise ValueError("Expected disjoint 16-image train calibration and 20-image official test manifests")
    if calibration_manifest.name != "train_images_seed2026.txt":
        raise ValueError("Calibration may only use train_images_seed2026.txt")
    if any(token in str(calibration_manifest) for token in ("val_images_seed2026.txt", "test.txt", "/DRIVE/test/")):
        raise ValueError("Validation/test paths are forbidden for calibration")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or device.index not in (None, 0):
        raise RuntimeError("Use the requested CUDA GPU 0")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch != 9:
        raise ValueError("Expected the FP32 best checkpoint from epoch 9; got {}".format(checkpoint_epoch))
    model = models.UNetFamily.U_Net(1, 2)
    model.load_state_dict(checkpoint["net"])
    fp32_model_state_bytes = sum(t.numel() * t.element_size() for t in checkpoint["net"].values())
    model.to(device).eval()
    del checkpoint

    print("FP32 checkpoint:", checkpoint_path)
    print("FP32 checkpoint epoch:", checkpoint_epoch)
    print("Calibration manifest:", calibration_manifest)
    print("Calibration images/patches:", len(calibration_rows), "/", args.calib_patches)
    print("Validation/test calibration: no")
    print("Output directory (will be created only after calibration smoke):", output_dir)

    calibration_started = time.perf_counter()
    patches = make_calibration_patches(calibration_manifest, args.calib_patches, args.seed)
    activation_stats, weight_stats = collect_calibration_stats(model, patches, device, args.batch_size)
    calibration_seconds = time.perf_counter() - calibration_started

    if args.calibration_only:
        smoke_model = models.UNetFamily.U_Net(1, 2)
        smoke_model.load_state_dict(model.state_dict())
        smoke_model.to(device).eval()
        smoke_layers = wrap_convolutions(smoke_model, activation_stats, weight_stats)
        with torch.inference_mode():
            smoke_output = smoke_model(torch.from_numpy(patches[:2]).to(device))
        print("INT4 calibration smoke passed; synthetic output:", tuple(smoke_output.shape))
        print("Conv2d layers calibrated/quantized:", len(smoke_layers), "; skipped: 0")
        print("Calibration seconds:", round(calibration_seconds, 2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    quantized_layers = wrap_convolutions(model, activation_stats, weight_stats)
    if len(quantized_layers) != len(weight_stats):
        raise RuntimeError("One or more Conv2d layers were not wrapped")

    config = {
        "method": "INT4 fake quant PTQ, Conv2d input activations and weights",
        "real_int4_inference": False,
        "statement": "This is INT4 fake-quant/simulation using FP32 operations; it is not real INT4 hardware inference.",
        "structure_aware_quantization": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "calibration_manifest": str(calibration_manifest),
        "calibration_images": len(calibration_rows),
        "calibration_patch_count": args.calib_patches,
        "calibration_patch_size": [PATCH_SIZE, PATCH_SIZE],
        "calibration_seed": args.seed,
        "calibration_batch_size": args.batch_size,
        "calibration_fov_rule": "training FOV center sampling",
        "validation_or_test_used_for_calibration": False,
        "test_manifest": str(test_manifest),
        "test_evaluated_once_after_calibration": True,
        "weight_quantization": "symmetric signed INT4 fake quant, per-output-channel [-7,7]",
        "activation_quantization": "asymmetric signed INT4 fake quant, per-Conv2d-input tensor [-8,7]",
        "other_operations": "FP32",
        "amp": False,
        "device": str(device),
        "quantized_layers": list(weight_stats),
        "skipped_layers": [],
    }
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    shutil.copy2(Path(__file__).resolve(), output_dir / "ptq_int4_baseline.py")
    np.save(output_dir / "calibration_patches.npy", patches)

    int4_state = {name: layer.export() for name, layer in quantized_layers.items()}
    int4_state_path = output_dir / "int4_weight_state_dict.pt"
    torch.save({"checkpoint_epoch": checkpoint_epoch, "layers": int4_state}, int4_state_path)
    layer_ranges = {
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
    with open(output_dir / "quant_layer_ranges.json", "w") as range_file:
        json.dump(layer_ranges, range_file, indent=2)
    shutil.copy2(test_manifest, output_dir / "test_manifest_used.txt")

    fp32_checkpoint_bytes = checkpoint_path.stat().st_size
    int4_weight_bits = sum(record["weight_value_count"] * 4 for record in int4_state.values())
    int4_theoretical_weight_bytes = (int4_weight_bits + 7) // 8
    int4_packed_weight_bytes = sum(record["weight_int4_packed"].numel() for record in int4_state.values())
    size_report = {
        "fp32_checkpoint_file_bytes": fp32_checkpoint_bytes,
        "fp32_model_state_bytes": fp32_model_state_bytes,
        "int4_theoretical_weight_payload_bytes": int4_theoretical_weight_bytes,
        "int4_packed_weight_payload_bytes": int4_packed_weight_bytes,
        "int4_fake_quant_artifact_file_bytes": int4_state_path.stat().st_size,
        "fp32_model_state_to_int4_theoretical_payload_ratio": fp32_model_state_bytes / int4_theoretical_weight_bytes,
        "fp32_checkpoint_to_int4_artifact_file_ratio": fp32_checkpoint_bytes / int4_state_path.stat().st_size,
        "note": "Fake quant retains FP32 execution. Ratios compare theoretical packed weights; they do not imply hardware INT4 acceleration.",
    }
    with open(output_dir / "model_size.json", "w") as size_file:
        json.dump(size_report, size_file, indent=2)

    metrics = evaluate_test_once(model, test_rows, output_dir, device, args.batch_size)
    metrics["calibration_seconds"] = calibration_seconds
    with open(output_dir / "performance.txt", "a") as performance_file:
        performance_file.write("\nINT4 type: fake quant/simulation with FP32 operations; not real INT4 hardware inference.\n")
        performance_file.write("Structure-aware quantization: not used.\n")
        performance_file.write("Calibration: {} ({} training-only patches).\n".format(
            calibration_manifest, args.calib_patches
        ))
        performance_file.write("Quantized Conv2d layers: {}; skipped: 0.\n".format(len(quantized_layers)))
        performance_file.write("Calibration time (seconds): {:.2f}\nTest time (seconds): {:.2f}\n".format(
            calibration_seconds, metrics["test_seconds"]
        ))
        performance_file.write("Peak GPU memory (GiB allocated/reserved): {:.2f}/{:.2f}\n".format(
            metrics["gpu_peak_allocated_gib"], metrics["gpu_peak_reserved_gib"]
        ))

    print("INT4 type: fake quant/simulation; not true INT4 hardware inference")
    print("Quantized Conv2d layers: {}; skipped: 0".format(len(quantized_layers)))
    print("Test metrics:", metrics)
    print("Model sizes:", size_report)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()