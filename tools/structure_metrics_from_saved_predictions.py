import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from lib.common import readImg
from lib.extract_patches import load_file_path_txt


FP32_RESULT = PROJECT_ROOT / "experiments/DRIVE_Unet_FP32_full_retry/result.npy"
INT8_RESULT = PROJECT_ROOT / "experiments/DRIVE_Unet_INT8_PTQ_baseline/result.npy"
INT4_RESULT = PROJECT_ROOT / "experiments/DRIVE_Unet_INT4_PTQ_baseline/result.npy"
TEST_MANIFEST = PROJECT_ROOT / "prepare_dataset/data_path_list/DRIVE/test.txt"
OUTPUT_CSV = PROJECT_ROOT / "experiments/DRIVE_structure_metrics/fp32_int8_structure_metrics.csv"
VESSEL_THRESHOLD = 0.5
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def load_saved_prediction(path):
    saved = np.load(path, allow_pickle=False)
    if saved.shape[0] != 2:
        raise ValueError("Expected saved [ground_truth, scores] array: {}".format(path))
    return np.asarray(saved[0]).reshape(-1), np.asarray(saved[1]).reshape(-1)


def load_test_ground_truth_and_fov(manifest_rows):
    masks = []
    fovs = []
    for _, label_path, fov_path in zip(*manifest_rows):
        mask = np.asarray(readImg(label_path))
        fov = np.asarray(readImg(fov_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if fov.ndim == 3:
            fov = fov[:, :, 0]
        masks.append(mask)
        fovs.append(fov)
    ground_truths = np.expand_dims(np.asarray(masks), axis=1)
    fovs = np.expand_dims(np.asarray(fovs), axis=1)
    if np.max(ground_truths) > 1:
        ground_truths = ground_truths / 255.0
    fovs = fovs / 255.0
    return ground_truths, fovs


def reconstruct_fov_images(saved_truth, saved_scores, ground_truths, fovs, pixel_order):
    expected_pixels = int(np.count_nonzero(fovs[:, 0] > 0))
    if saved_truth.size != expected_pixels or saved_scores.size != expected_pixels:
        raise ValueError(
            "Saved prediction pixels ({}/{}) do not match test FOV pixel count ({})".format(
                saved_truth.size, saved_scores.size, expected_pixels
            )
        )

    image_count, _, image_h, image_w = fovs.shape
    prediction_images = np.zeros((image_count, image_h, image_w), dtype=np.float32)
    truth_images = np.zeros((image_count, image_h, image_w), dtype=bool)
    offset = 0
    for image_index in range(image_count):
        fov = fovs[image_index, 0] > 0
        fov_flat = fov.reshape(-1, order=pixel_order)
        pixel_count = int(fov_flat.sum())
        next_offset = offset + pixel_count
        image_scores = np.zeros(image_h * image_w, dtype=np.float32)
        image_truth = np.zeros(image_h * image_w, dtype=bool)
        image_scores[fov_flat] = saved_scores[offset:next_offset]
        image_truth[fov_flat] = saved_truth[offset:next_offset] >= 0.5
        prediction_images[image_index] = image_scores.reshape((image_h, image_w), order=pixel_order)
        image_truth_map = image_truth.reshape((image_h, image_w), order=pixel_order)
        truth_images[image_index] = image_truth_map

        manifest_truth = ground_truths[image_index, 0] > 0
        if not np.array_equal(image_truth_map[fov], manifest_truth[fov]):
            raise ValueError("Saved truth ordering does not match DRIVE test manifest at image {}".format(image_index))
        offset = next_offset
    return prediction_images, truth_images


def analyze_prediction(prediction_images, truth_images, fovs):
    total_tp = total_fp = total_fn = 0
    thin_tp = thin_fp = thin_fn = 0
    skeleton_intersection = skeleton_prediction_count = skeleton_truth_count = 0
    topology_precision_intersection = topology_sensitivity_intersection = 0
    predicted_component_count = 0
    break_count = 0

    for prediction_scores, truth, fov_raw in zip(prediction_images, truth_images, fovs[:, 0]):
        fov = fov_raw > 0
        prediction = (prediction_scores >= VESSEL_THRESHOLD) & fov
        truth = truth & fov

        total_tp += int(np.count_nonzero(prediction & truth))
        total_fp += int(np.count_nonzero(prediction & ~truth & fov))
        total_fn += int(np.count_nonzero(~prediction & truth))

        truth_skeleton = skeletonize(truth) & fov
        distance_inside_vessel = ndimage.distance_transform_edt(truth)
        thin_centerline = truth_skeleton & ((2.0 * distance_inside_vessel) <= 2.0 + 1e-9)
        thin_truth = ndimage.binary_dilation(thin_centerline, structure=CONNECTIVITY_8) & truth & fov
        thin_tp += int(np.count_nonzero(prediction & thin_truth))
        thin_fp += int(np.count_nonzero(prediction & ~thin_truth & fov))
        thin_fn += int(np.count_nonzero(~prediction & thin_truth))

        prediction_skeleton = skeletonize(prediction) & fov
        skeleton_intersection += int(np.count_nonzero(prediction_skeleton & truth_skeleton))
        skeleton_prediction_count += int(np.count_nonzero(prediction_skeleton))
        skeleton_truth_count += int(np.count_nonzero(truth_skeleton))
        topology_precision_intersection += int(np.count_nonzero(prediction_skeleton & truth))
        topology_sensitivity_intersection += int(np.count_nonzero(truth_skeleton & prediction))

        predicted_labels, predicted_components = ndimage.label(prediction, structure=CONNECTIVITY_8)
        predicted_component_count += int(predicted_components)
        gt_skeleton_labels, gt_skeleton_components = ndimage.label(truth_skeleton, structure=CONNECTIVITY_8)
        uncovered_skeleton = truth_skeleton & ~ndimage.binary_dilation(prediction, structure=CONNECTIVITY_8)
        _, uncovered_fragments = ndimage.label(uncovered_skeleton, structure=CONNECTIVITY_8)
        break_count += int(uncovered_fragments)

    dice_denominator = 2 * total_tp + total_fp + total_fn
    iou_denominator = total_tp + total_fp + total_fn
    thin_dice_denominator = 2 * thin_tp + thin_fp + thin_fn
    thin_recall_denominator = thin_tp + thin_fn
    skeleton_dice_denominator = skeleton_prediction_count + skeleton_truth_count
    skeleton_precision = (
        topology_precision_intersection / skeleton_prediction_count if skeleton_prediction_count else 0.0
    )
    skeleton_sensitivity = (
        topology_sensitivity_intersection / skeleton_truth_count if skeleton_truth_count else 0.0
    )
    cldice_denominator = skeleton_precision + skeleton_sensitivity

    return {
        "overall_dice": 2 * total_tp / dice_denominator if dice_denominator else 0.0,
        "overall_iou": total_tp / iou_denominator if iou_denominator else 0.0,
        "vessel_sensitivity_recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "thin_vessel_recall": thin_tp / thin_recall_denominator if thin_recall_denominator else 0.0,
        "thin_vessel_dice": 2 * thin_tp / thin_dice_denominator if thin_dice_denominator else 0.0,
        "skeleton_dice": (
            2 * skeleton_intersection / skeleton_dice_denominator if skeleton_dice_denominator else 0.0
        ),
        "clDice": (
            2 * skeleton_precision * skeleton_sensitivity / cldice_denominator if cldice_denominator else 0.0
        ),
        "prediction_connected_components": predicted_component_count,
        "vessel_break_count": break_count,
        "vessel_break_rate": (
            break_count / skeleton_truth_count
            if skeleton_truth_count > 0
            else 0.0
        ),
        "uncovered_skeleton_fragment_count": break_count,
        "fov_pixel_count": int(np.count_nonzero(fovs[:, 0] > 0)),
    }


def main():
    if not FP32_RESULT.is_file() or not INT8_RESULT.is_file() or not INT4_RESULT.is_file() or not TEST_MANIFEST.is_file():
        raise FileNotFoundError("Expected saved FP32/INT8/INT4 result.npy and DRIVE test manifest")
    test_rows = load_file_path_txt(str(TEST_MANIFEST))
    image_paths, _, _ = test_rows
    if len(image_paths) != 20 or any("/DRIVE/test/images/" not in path for path in image_paths):
        raise ValueError("Expected the 20-image official DRIVE test manifest")

    ground_truths, fovs = load_test_ground_truth_and_fov(test_rows)
    fp32_truth, fp32_scores = load_saved_prediction(FP32_RESULT)
    int8_truth, int8_scores = load_saved_prediction(INT8_RESULT)
    int4_truth, int4_scores = load_saved_prediction(INT4_RESULT)
    fp32_predictions, fp32_truth_images = reconstruct_fov_images(
        fp32_truth, fp32_scores, ground_truths, fovs, pixel_order="F"
    )
    int8_predictions, int8_truth_images = reconstruct_fov_images(
        int8_truth, int8_scores, ground_truths, fovs, pixel_order="C"
    )
    int4_predictions, int4_truth_images = reconstruct_fov_images(
        int4_truth, int4_scores, ground_truths, fovs, pixel_order="C"
    )
    if not np.array_equal(fp32_truth_images, int8_truth_images):
        raise ValueError("Reconstructed FP32 and INT8 truth maps differ")
    if not np.array_equal(fp32_truth_images, int4_truth_images):
        raise ValueError("Reconstructed FP32 and INT4 truth maps differ")

    fp32_metrics = analyze_prediction(fp32_predictions, fp32_truth_images, fovs)
    int8_metrics = analyze_prediction(int8_predictions, int8_truth_images, fovs)
    int4_metrics = analyze_prediction(int4_predictions, int4_truth_images, fovs)
    thin_rule = (
        "GT skeleton pixels with local width 2*EDT(GT vessel)<=2 px; thin GT region is those pixels "
        "dilated by one 8-neighbor pixel and intersected with GT vessel and FOV"
    )
    break_rule = (
        "FOV-restricted 8-connected fragments of GT skeleton pixels with no predicted vessel pixel in their "
        "immediate 8-neighborhood; break rate is fragment count / GT skeleton pixel count (fragment density)"
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "metric",
            "FP32",
            "INT8_fake_quant",
            "INT4_fake_quant",
            "delta_FP32_minus_INT8",
            "delta_FP32_minus_INT4",
            "definition",
        ])
        for metric in (
            "overall_dice",
            "overall_iou",
            "vessel_sensitivity_recall",
            "thin_vessel_recall",
            "thin_vessel_dice",
            "skeleton_dice",
            "clDice",
            "prediction_connected_components",
            "vessel_break_count",
            "vessel_break_rate",
        ):
            definition = thin_rule if metric.startswith("thin_vessel_") else (
                break_rule if metric.startswith("vessel_break_") else "FOV pixels only; threshold=0.5"
            )
            fp32_value = fp32_metrics[metric]
            int8_value = int8_metrics[metric]
            int4_value = int4_metrics[metric]
            writer.writerow([
                metric,
                fp32_value,
                int8_value,
                int4_value,
                fp32_value - int8_value,
                fp32_value - int4_value,
                definition,
            ])
        writer.writerow(["fov_pixel_count", fp32_metrics["fov_pixel_count"], int8_metrics["fov_pixel_count"],
                         int4_metrics["fov_pixel_count"], 0, 0,
                         "Pixels included in this analysis"])

    print("Saved-predictions-only structure analysis complete")
    print("FP32 saved prediction:", FP32_RESULT)
    print("INT8 saved prediction:", INT8_RESULT)
    print("INT4 saved prediction:", INT4_RESULT)
    print("Test images used for GT/FOV mapping:", len(image_paths))
    print("No inference, training, or calibration was performed")
    print("Thin-vessel rule:", thin_rule)
    print("Break-count rule:", break_rule)
    print("CSV:", OUTPUT_CSV)
    print("FP32 metrics:", fp32_metrics)
    print("INT8 fake-quant metrics:", int8_metrics)
    print("INT4 fake-quant metrics:", int4_metrics)


if __name__ == "__main__":
    main()