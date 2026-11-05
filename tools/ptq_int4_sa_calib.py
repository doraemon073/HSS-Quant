import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from scipy import ndimage

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from skimage.morphology import skeletonize

import models
from lib.extract_patches import load_data, load_file_path_txt, my_PreProc
from tools.ptq_int4_baseline import evaluate_test_once
from tools.ptq_int4_baseline import FakeQuantConv2dInt4, pack_signed_int4, wrap_convolutions
from tools.structure_metrics_from_saved_predictions import (
    analyze_prediction,
    load_saved_prediction,
    load_test_ground_truth_and_fov,
    reconstruct_fov_images,
)
from tools.ptq_int8_baseline import collect_calibration_stats, read_manifest, validate_paths


PATCH_SIZE = 64
SCORE_WEIGHTS = {
    "vessel_ratio": 1.0,
    "thin_vessel_ratio": 3.0,
    "skeleton_ratio": 2.0,
    "boundary_ratio": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Structure-aware calibration for ordinary INT4 fake-quant PTQ")
    parser.add_argument("--fp32-exp", default="experiments/DRIVE_Unet_FP32_full_retry")
    parser.add_argument("--output", default="experiments/DRIVE_Unet_INT4_SA_Calib")
    parser.add_argument("--calib-list", default="lists/DRIVE/train_images_seed2026.txt")
    parser.add_argument("--test-list", default="prepare_dataset/data_path_list/DRIVE/test.txt")
    parser.add_argument("--candidate-patches", type=int, default=10000)
    parser.add_argument("--calib-patches", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--smoke-selected-patches", type=int, default=64)
    parser.add_argument("--finalize-structure-only", action="store_true",
                        help="Build four-way structure comparison from saved result.npy files without test inference")
    return parser.parse_args()


def build_structure_candidates(calibration_manifest, candidate_count, seed):
    rows = read_manifest(calibration_manifest)
    source_ids = []
    for image_path, _, _ in rows:
        match = re.match(r"^(\d+)_", Path(image_path).name)
        if match is None:
            raise ValueError("Cannot extract original image ID from {}".format(image_path))
        source_ids.append(match.group(1))

    raw_images, raw_masks, raw_fovs = load_data(str(calibration_manifest))
    images = my_PreProc(raw_images)
    masks = raw_masks[:, 0] > 0
    fovs = raw_fovs[:, 0] > 0
    vessel_regions = masks & fovs
    skeleton_maps = np.asarray([skeletonize(region) for region in vessel_regions])
    thin_maps = []
    boundary_maps = []
    for vessel_region, skeleton in zip(vessel_regions, skeleton_maps):
        distance_inside_vessel = ndimage.distance_transform_edt(vessel_region)
        thin_centerline = skeleton & ((2.0 * distance_inside_vessel) <= 2.0 + 1e-9)
        thin_maps.append(
            ndimage.binary_dilation(thin_centerline, structure=np.ones((3, 3), dtype=bool)) & vessel_region
        )
        eroded = ndimage.binary_erosion(
            vessel_region, structure=np.ones((3, 3), dtype=bool), border_value=0
        )
        boundary_maps.append(vessel_region & ~eroded)
    thin_maps = np.asarray(thin_maps)
    boundary_maps = np.asarray(boundary_maps)
    rng = np.random.RandomState(seed)
    half = PATCH_SIZE // 2
    patches = np.empty((candidate_count, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    records = []
    image_count, _, image_h, image_w = images.shape
    candidate_index = 0

    while candidate_index < candidate_count:
        image_index = int(rng.randint(image_count))
        center_x = int(rng.randint(half, image_w - half + 1))
        center_y = int(rng.randint(half, image_h - half + 1))
        if not fovs[image_index, center_y, center_x]:
            continue

        y0, y1 = center_y - half, center_y + half
        x0, x1 = center_x - half, center_x + half
        fov_patch = fovs[image_index, y0:y1, x0:x1]
        vessel_region = vessel_regions[image_index, y0:y1, x0:x1]
        thin_region = thin_maps[image_index, y0:y1, x0:x1]
        skeleton = skeleton_maps[image_index, y0:y1, x0:x1]
        boundary = boundary_maps[image_index, y0:y1, x0:x1]
        patch_area = float(PATCH_SIZE * PATCH_SIZE)

        vessel_ratio = float(vessel_region.sum()) / patch_area
        thin_vessel_ratio = float(thin_region.sum()) / patch_area
        skeleton_ratio = float(skeleton.sum()) / patch_area
        boundary_ratio = float(boundary.sum()) / patch_area
        score = (
            SCORE_WEIGHTS["vessel_ratio"] * vessel_ratio
            + SCORE_WEIGHTS["thin_vessel_ratio"] * thin_vessel_ratio
            + SCORE_WEIGHTS["skeleton_ratio"] * skeleton_ratio
            + SCORE_WEIGHTS["boundary_ratio"] * boundary_ratio
        )
        patches[candidate_index] = images[image_index, :, y0:y1, x0:x1]
        records.append({
            "candidate_index": candidate_index,
            "source_image_id": source_ids[image_index],
            "source_image_path": rows[image_index][0],
            "center_x": center_x,
            "center_y": center_y,
            "vessel_ratio": vessel_ratio,
            "thin_vessel_ratio": thin_vessel_ratio,
            "skeleton_ratio": skeleton_ratio,
            "boundary_ratio": boundary_ratio,
            "score": score,
        })
        candidate_index += 1

    return patches, records


def select_top_candidates(patches, records, selected_count):
    if selected_count <= 0 or selected_count > len(records):
        raise ValueError("Selected calibration patch count must be in [1, candidate_count]")
    # Stable ordering makes ties deterministic and preserves candidate generation order.
    order = sorted(range(len(records)), key=lambda index: (-records[index]["score"], index))
    selected = np.asarray(order[:selected_count], dtype=np.int64)
    selected_patches = patches[selected]
    selected_set = set(selected.tolist())
    selected_records = []
    for record in records:
        selected_record = dict(record)
        selected_record["selected_for_calibration"] = record["candidate_index"] in selected_set
        selected_records.append(selected_record)
    return selected_patches, selected_records, selected


def save_candidate_scores(path, records):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()) + ["selected_for_calibration"])
        writer.writeheader()
        writer.writerows(records)


def finalize_structure_comparison(output_dir):
    test_manifest = PROJECT_ROOT / "prepare_dataset/data_path_list/DRIVE/test.txt"
    test_rows = load_file_path_txt(str(test_manifest))
    image_paths, _, _ = test_rows
    if len(image_paths) != 20 or any("/DRIVE/test/images/" not in path for path in image_paths):
        raise ValueError("Expected 20 official DRIVE test images for cached-result alignment")
    ground_truths, fovs = load_test_ground_truth_and_fov(test_rows)
    prediction_specs = {
        "FP32": (PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/result.npy", "F"),
        "INT8": (PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/result.npy", "C"),
        "INT4": (PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/result.npy", "C"),
        "INT4_SA": (output_dir / "result.npy", "C"),
    }
    structures = {}
    reference_truth = None
    for name, (result_path, pixel_order) in prediction_specs.items():
        if not result_path.is_file():
            raise FileNotFoundError("Saved prediction result missing: {}".format(result_path))
        saved_truth, saved_scores = load_saved_prediction(result_path)
        prediction_maps, truth_maps = reconstruct_fov_images(
            saved_truth, saved_scores, ground_truths, fovs, pixel_order
        )
        if reference_truth is None:
            reference_truth = truth_maps
        elif not np.array_equal(reference_truth, truth_maps):
            raise ValueError("Cached test truth maps differ across prediction files")
        structures[name] = analyze_prediction(prediction_maps, truth_maps, fovs)

    metric_names = [
        "overall_dice", "overall_iou", "vessel_sensitivity_recall", "thin_vessel_recall",
        "thin_vessel_dice", "skeleton_dice", "clDice", "prediction_connected_components",
        "vessel_break_count", "vessel_break_rate",
    ]
    rows = []
    for metric in metric_names:
        row = {"metric": metric}
        for model_name in ("FP32", "INT8", "INT4", "INT4_SA"):
            row[model_name] = structures[model_name][metric]
        row["delta_FP32_minus_INT4_SA"] = row["FP32"] - row["INT4_SA"]
        row["delta_INT4_minus_INT4_SA"] = row["INT4"] - row["INT4_SA"]
        rows.append(row)
    comparison_path = output_dir / "fp32_int8_int4_sa_structure_comparison.csv"
    with open(comparison_path, "w", newline="") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    performance_path = output_dir / "performance.txt"
    existing_report = performance_path.read_text() if performance_path.exists() else ""
    completion_marker = "SA structure metrics finalized from cached predictions."
    if completion_marker not in existing_report:
        with open(performance_path, "a") as performance_file:
            performance_file.write("\n{}\n".format(completion_marker))
            performance_file.write("Calibration: 10000 training-only candidates, top 1024 selected; no validation/test calibration.\n")
            performance_file.write("Fake quant mode: ordinary INT4 weight/input-activation fake quant; FP32 arithmetic.\n")
            performance_file.write("Structure-aware quantization/model-structure changes: not used.\n")
            for model_name, model_metrics in structures.items():
                performance_file.write(
                    "{}: Dice={}, IoU={}, thin recall={}, thin Dice={}, skeleton Dice={}, clDice={}, components={}, uncovered fragments={}.\n".format(
                        model_name,
                        model_metrics["overall_dice"],
                        model_metrics["overall_iou"],
                        model_metrics["thin_vessel_recall"],
                        model_metrics["thin_vessel_dice"],
                        model_metrics["skeleton_dice"],
                        model_metrics["clDice"],
                        model_metrics["prediction_connected_components"],
                        model_metrics["vessel_break_count"],
                    )
                )
    print("Saved-prediction-only structure comparison complete")
    print("No model inference, training, or calibration performed")
    print("Structure comparison:", structures)
    print("Comparison CSV:", comparison_path)
    return structures


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    fp32_exp = Path(args.fp32_exp)
    if not fp32_exp.is_absolute():
        fp32_exp = PROJECT_ROOT / fp32_exp
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if args.finalize_structure_only:
        if not output_dir.is_dir():
            raise FileNotFoundError("SA-INT4 result directory not found: {}".format(output_dir))
        finalize_structure_comparison(output_dir)
        return
    calibration_manifest = Path(args.calib_list)
    if not calibration_manifest.is_absolute():
        calibration_manifest = PROJECT_ROOT / calibration_manifest

    checkpoint_path = fp32_exp / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("FP32 checkpoint not found: {}".format(checkpoint_path))
    if not calibration_manifest.is_file():
        raise FileNotFoundError("Training calibration manifest not found: {}".format(calibration_manifest))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory already exists and is non-empty: {}".format(output_dir))
    if output_dir.resolve() in {
        fp32_exp.resolve(),
        (PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline").resolve(),
        (PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline").resolve(),
    }:
        raise ValueError("Refusing to use an FP32 or baseline quantization experiment directory as output")
    if args.candidate_patches != 10000:
        raise ValueError("Structure-aware baseline requires exactly 10000 candidate patches")
    if args.calib_patches != 1024 and not args.calibration_only:
        raise ValueError("Full SA calibration must select exactly 1024 patches")
    if args.calibration_only and args.smoke_selected_patches <= 0:
        raise ValueError("Smoke selected-patch count must be positive")

    calibration_rows = read_manifest(calibration_manifest)
    training_ids = validate_paths(calibration_rows, "training")
    if len(calibration_rows) != 16:
        raise ValueError("Expected the 16-image fixed training manifest")
    if calibration_manifest.name != "train_images_seed2026.txt":
        raise ValueError("Calibration is restricted to train_images_seed2026.txt")
    forbidden_sources = ("val_images_seed2026.txt", "test.txt", "/DRIVE/test/")
    if any(token in str(calibration_manifest) for token in forbidden_sources):
        raise ValueError("Validation/test source detected in calibration path")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or device.index not in (None, 0):
        raise RuntimeError("Use CUDA GPU 0 for this PTQ run")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch != 9:
        raise ValueError("Expected FP32 best checkpoint from epoch 9")
    model = models.UNetFamily.U_Net(1, 2)
    model.load_state_dict(checkpoint["net"])
    fp32_model_state_bytes = sum(value.numel() * value.element_size() for value in checkpoint["net"].values())
    model.to(device).eval()
    del checkpoint

    print("Checkpoint:", checkpoint_path, "epoch:", checkpoint_epoch)
    print("Calibration source:", calibration_manifest)
    print("Training source IDs:", sorted(training_ids, key=int))
    print("Candidates:", args.candidate_patches, "patches; selected:",
          args.smoke_selected_patches if args.calibration_only else args.calib_patches)
    print("Validation/test used for calibration: no")

    candidate_patches, candidate_records = build_structure_candidates(
        calibration_manifest, args.candidate_patches, args.seed
    )
    selected_count = args.smoke_selected_patches if args.calibration_only else args.calib_patches
    selected_patches, candidate_records, selected_indices = select_top_candidates(
        candidate_patches, candidate_records, selected_count
    )
    if any(record["source_image_id"] not in training_ids for record in candidate_records):
        raise RuntimeError("A calibration candidate did not originate in the training images")

    calibration_started = time.perf_counter()
    activation_stats, weight_stats = collect_calibration_stats(
        model, selected_patches, device, args.batch_size
    )
    calibration_seconds = time.perf_counter() - calibration_started

    if args.calibration_only:
        smoke_model = models.UNetFamily.U_Net(1, 2)
        smoke_model.load_state_dict(model.state_dict())
        smoke_model.to(device).eval()
        smoke_layers = wrap_convolutions(smoke_model, activation_stats, weight_stats)
        with torch.inference_mode():
            smoke_output = smoke_model(torch.from_numpy(selected_patches[:2]).to(device))
        print("SA selection smoke passed; synthetic output:", tuple(smoke_output.shape))
        print("Candidates scored:", len(candidate_records), "selected training patches:", len(selected_patches))
        print("Conv2d layers calibrated/quantized:", len(smoke_layers), "; skipped: 0")
        print("Test manifest/images were not opened or evaluated")
        print("Calibration seconds:", round(calibration_seconds, 2))
        return

    # Test manifest is intentionally untouched until calibration and fake-quant conversion finish.
    output_dir.mkdir(parents=True, exist_ok=True)
    test_manifest = PROJECT_ROOT / "prepare_dataset/data_path_list/DRIVE/test.txt"
    if not test_manifest.is_file():
        raise FileNotFoundError("Official DRIVE test manifest missing after calibration")
    test_rows = read_manifest(test_manifest)
    test_ids = validate_paths(test_rows, "test")
    if len(test_rows) != 20 or training_ids & test_ids:
        raise ValueError("Expected disjoint official train and 20-image test manifests")

    quantized_layers = wrap_convolutions(model, activation_stats, weight_stats)
    if len(quantized_layers) != len(weight_stats):
        raise RuntimeError("One or more Conv2d layers were not fake quantized")
    config = {
        "method": "Structure-aware calibration, ordinary INT4 fake quant PTQ",
        "real_int4_inference": False,
        "statement": "INT4 fake-quant/simulation using FP32 arithmetic; not real INT4 hardware inference.",
        "structure_aware_quantization": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 9,
        "calibration_manifest": str(calibration_manifest),
        "calibration_source_image_ids": sorted(training_ids, key=int),
        "candidate_patch_count": args.candidate_patches,
        "selected_calibration_patch_count": args.calib_patches,
        "calibration_patch_size": [PATCH_SIZE, PATCH_SIZE],
        "calibration_seed": args.seed,
        "score_weights": SCORE_WEIGHTS,
        "thin_vessel_definition": "GT skeleton pixels with local width 2*EDT(GT vessel)<=2 px, dilated 1 pixel and clipped to GT vessel and FOV",
        "boundary_definition": "GT vessel pixels removed by one-iteration 8-neighbor binary erosion, divided by patch area",
        "calibration_patch_selection": "top score among 10000 candidates sampled only from train_images_seed2026.txt",
        "validation_or_test_used_for_calibration": False,
        "test_manifest": str(test_manifest),
        "test_evaluated_once_after_calibration": True,
        "weight_quantization": "symmetric signed INT4 fake quant, per-output-channel [-7,7]",
        "activation_quantization": "asymmetric signed INT4 fake quant at Conv2d input, [-8,7]",
        "quantized_layers": list(weight_stats),
        "skipped_layers": [],
        "other_operations": "FP32",
        "device": str(device),
    }
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    shutil.copy2(Path(__file__).resolve(), output_dir / "ptq_int4_sa_calib.py")
    np.save(output_dir / "candidate_patches.npy", candidate_patches)
    np.save(output_dir / "selected_calibration_patches.npy", selected_patches)
    with open(output_dir / "candidate_patch_scores.csv", "w", newline="") as score_file:
        writer = csv.DictWriter(score_file, fieldnames=list(candidate_records[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_records)
    selected_id_set = set(selected_indices.tolist())
    with open(output_dir / "selected_calibration_patches.csv", "w", newline="") as selected_file:
        writer = csv.DictWriter(selected_file, fieldnames=list(candidate_records[0].keys()))
        writer.writeheader()
        writer.writerows(record for record in candidate_records if record["candidate_index"] in selected_id_set)

    int4_state = {name: layer.export() for name, layer in quantized_layers.items()}
    int4_state_path = output_dir / "int4_sa_weight_state_dict.pt"
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
    with open(output_dir / "test_manifest_used.txt", "w") as manifest_file:
        for row in test_rows:
            manifest_file.write("{} {} {}\n".format(*row))

    metrics = evaluate_test_once(model, test_rows, output_dir, device, args.batch_size)
    metrics["calibration_seconds"] = calibration_seconds
    int4_weight_bytes = sum(record["weight_value_count"] * 4 for record in int4_state.values()) // 8
    with open(output_dir / "model_size.json", "w") as size_file:
        json.dump({
            "fp32_model_state_bytes": fp32_model_state_bytes,
            "int4_theoretical_weight_payload_bytes": int4_weight_bytes,
            "int4_sa_fake_quant_artifact_bytes": int4_state_path.stat().st_size,
            "theoretical_weight_compression_ratio": fp32_model_state_bytes / int4_weight_bytes,
            "note": "Fake quant executes FP32 operations; payload size is a theoretical packed-weight comparison, not hardware acceleration.",
        }, size_file, indent=2)

    structures = finalize_structure_comparison(output_dir)
    with open(output_dir / "performance.txt", "a") as performance_file:
        performance_file.write("\nSA calibration candidates: 10000; selected: 1024; source: training manifest only.\n")
        performance_file.write("Quantized Conv2d layers: {}; skipped: 0.\n".format(len(quantized_layers)))
        performance_file.write("Structure metrics computed from saved test predictions; no second inference.\n")
        performance_file.write("Calibration seconds: {:.2f}; test seconds: {:.2f}.\n".format(
            calibration_seconds, metrics["test_seconds"]
        ))
        performance_file.write("Peak GPU memory allocated/reserved GiB: {:.2f}/{:.2f}.\n".format(
            metrics["gpu_peak_allocated_gib"], metrics["gpu_peak_reserved_gib"]
        ))
    print("INT4 SA metrics:", metrics)
    print("FP32/INT8/INT4/INT4-SA structure metrics:", structures)
    print("Calibration source IDs:", sorted({record["source_image_id"] for record in candidate_records}, key=int))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()