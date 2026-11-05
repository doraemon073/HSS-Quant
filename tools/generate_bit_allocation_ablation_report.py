import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REPORT_DIR = EXPERIMENTS / "DRIVE_Unet_BitAllocation_Ablation_Report"
LONG_DIR = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_v1_long"
METHODS = (
    "FP32",
    "Full INT8",
    "Full INT4",
    "SA-INT4",
    "Manual Encoder INT8 + Decoder INT4",
    "StructuralTrainableBits",
)
METRIC_COLUMNS = (
    "AUC", "Dice/F1", "IoU", "Accuracy", "Sensitivity", "Specificity",
    "Thin-vessel recall", "Thin-vessel Dice", "Skeleton Dice", "clDice",
)
SIZE_COLUMNS = (
    "Mean layer bitwidth",
    "Parameter-weighted mean bitwidth",
    "Theoretical Conv2d weight bytes",
    "Theoretical compression ratio vs FP32 Conv2d weights",
)


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
    }
    for line in path.read_text().splitlines():
        for metric, pattern in patterns.items():
            match = re.search(pattern, line.strip())
            if match:
                values[metric] = float(match.group(1))
    return values


def read_structure_table(path):
    values = {name: {} for name in ("FP32", "Full INT8", "Full INT4", "SA-INT4")}
    columns = {"FP32": "FP32", "Full INT8": "INT8", "Full INT4": "INT4", "SA-INT4": "INT4_SA"}
    row_to_metric = {
        "thin_vessel_recall": "Thin-vessel recall",
        "thin_vessel_dice": "Thin-vessel Dice",
        "skeleton_dice": "Skeleton Dice",
        "clDice": "clDice",
    }
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            metric = row_to_metric.get(row["metric"])
            if metric:
                for method, column in columns.items():
                    values[method][metric] = float(row[column])
    return values


def load_inputs():
    if REPORT_DIR.exists():
        generated_partial = {"bit_allocation_comparison.csv", "layer_bitwidth_comparison.csv"}
        current_files = {path.name for path in REPORT_DIR.iterdir()}
        if current_files != generated_partial:
            raise FileExistsError("Refusing to overwrite report directory: {}".format(REPORT_DIR))

    layer_map_path = LONG_DIR / "layer_mapping.json"
    best_path = LONG_DIR / "best_checkpoint.pt"
    structural_test_path = LONG_DIR / "test_results" / "quant_config.json"
    mixed_root = EXPERIMENTS / "DRIVE_Unet_INT4_LayerSensitivity" / "encoder_int8_decoder_int4"
    structure_path = EXPERIMENTS / "DRIVE_Unet_INT4_SA_Calib" / "fp32_int8_int4_sa_structure_comparison.csv"
    required = [
        EXPERIMENTS / "DRIVE_Unet_FP32_full_retry" / "performance.txt",
        EXPERIMENTS / "DRIVE_Unet_INT8_PTQ_baseline" / "performance.txt",
        EXPERIMENTS / "DRIVE_Unet_INT8_PTQ_baseline" / "model_size.json",
        EXPERIMENTS / "DRIVE_Unet_INT4_PTQ_baseline" / "performance.txt",
        EXPERIMENTS / "DRIVE_Unet_INT4_PTQ_baseline" / "model_size.json",
        EXPERIMENTS / "DRIVE_Unet_INT4_SA_Calib" / "performance.txt",
        EXPERIMENTS / "DRIVE_Unet_INT4_SA_Calib" / "model_size.json",
        structure_path,
        mixed_root / "performance.txt",
        mixed_root / "layer_assignment.json",
        layer_map_path,
        best_path,
        structural_test_path,
        EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_full_ablation" / "ablation_results.csv",
        EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_full_ablation_multiseed" / "mean_std.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cached report inputs missing: {}".format(missing))

    layer_mapping = json.loads(layer_map_path.read_text())
    if len(layer_mapping) != 23:
        raise ValueError("Expected a 23-layer Conv2d mapping")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    structural_test = json.loads(structural_test_path.read_text())
    if int(best["epoch"]) != 16 or structural_test["best_checkpoint"] != str(best_path):
        raise ValueError("StructuralTrainableBits cached test output does not match expected best checkpoint")
    if structural_test["inference_passes"] != 1 or structural_test["training_performed"]:
        raise ValueError("StructuralTrainableBits result is not a saved evaluation-only result")
    if set(best["discrete_bit_width"]) != set(layer_mapping):
        raise ValueError("Best checkpoint and layer mapping do not contain the same layer keys")
    if best["discrete_bit_width"] != structural_test["per_layer_discrete_bit_width"]:
        raise ValueError("Saved test bit assignment differs from the best checkpoint")

    manual = json.loads((mixed_root / "layer_assignment.json").read_text())
    manual_bits = {name: int(bit) for name, bit in manual["layer_bits"].items()}
    if set(manual_bits) != set(layer_mapping):
        raise ValueError("Manual mixed assignment does not map all 23 Conv2d layers")
    if any(int(bit) not in (4, 8, 32) for bit in manual_bits.values()):
        raise ValueError("Manual assignment contains unsupported precision")

    performance_paths = {
        "FP32": EXPERIMENTS / "DRIVE_Unet_FP32_full_retry" / "performance.txt",
        "Full INT8": EXPERIMENTS / "DRIVE_Unet_INT8_PTQ_baseline" / "performance.txt",
        "Full INT4": EXPERIMENTS / "DRIVE_Unet_INT4_PTQ_baseline" / "performance.txt",
        "SA-INT4": EXPERIMENTS / "DRIVE_Unet_INT4_SA_Calib" / "performance.txt",
        "Manual Encoder INT8 + Decoder INT4": mixed_root / "performance.txt",
    }
    metrics = {method: parse_performance(path) for method, path in performance_paths.items()}
    structures = read_structure_table(structure_path)
    for method in ("FP32", "Full INT8", "Full INT4", "SA-INT4"):
        metrics[method].update(structures[method])
    metrics["Manual Encoder INT8 + Decoder INT4"].update({
        name: parse_performance(mixed_root / "performance.txt")[name]
        for name in ("Thin-vessel recall", "Thin-vessel Dice", "Skeleton Dice", "clDice")
    })
    metrics["StructuralTrainableBits"] = {
        **structural_test["test_metrics"],
        **{
            "Thin-vessel recall": structural_test["structure_metrics"]["thin_vessel_recall"],
            "Thin-vessel Dice": structural_test["structure_metrics"]["thin_vessel_dice"],
            "Skeleton Dice": structural_test["structure_metrics"]["skeleton_dice"],
            "clDice": structural_test["structure_metrics"]["clDice"],
        },
    }
    for method in METHODS:
        absent = [metric for metric in METRIC_COLUMNS if metric not in metrics[method]]
        if absent:
            raise ValueError("Cached metrics missing for {}: {}".format(method, absent))

    assignments = {
        "FP32": {name: 32 for name in layer_mapping},
        "Full INT8": {name: 8 for name in layer_mapping},
        "Full INT4": {name: 4 for name in layer_mapping},
        "SA-INT4": {name: 4 for name in layer_mapping},
        "Manual Encoder INT8 + Decoder INT4": manual_bits,
        "StructuralTrainableBits": {name: int(bit) for name, bit in best["discrete_bit_width"].items()},
    }
    return layer_mapping, assignments, metrics, best, structural_test


def derive_method_tables(layer_mapping, assignments, metrics):
    total_parameters = sum(int(layer["weight_numel"]) for layer in layer_mapping.values())
    fp32_conv_bytes = total_parameters * 4
    rows = []
    method_bits = {}
    for method in METHODS:
        bits = assignments[method]
        bit_payload = sum(int(layer_mapping[name]["weight_numel"]) * bits[name] for name in layer_mapping)
        payload_bytes = int(math.ceil(bit_payload / 8.0))
        arithmetic_mean = sum(bits.values()) / len(bits)
        parameter_weighted_mean = bit_payload / total_parameters
        method_bits[method] = bits
        rows.append({
            "method": method,
            **{metric: metrics[method][metric] for metric in METRIC_COLUMNS},
            "mean_layer_bitwidth": arithmetic_mean,
            "parameter_weighted_mean_bitwidth": parameter_weighted_mean,
            "theoretical_conv_weight_bytes": payload_bytes,
            "theoretical_compression_ratio_vs_fp32_conv_weights": fp32_conv_bytes / payload_bytes,
        })
    return rows, method_bits, total_parameters, fp32_conv_bytes


def group_allocation_summary(layer_mapping, method_bits):
    report = {}
    for group in ("encoder", "bottleneck", "decoder", "output"):
        names = [name for name, item in layer_mapping.items() if item["module"] == group]
        params = sum(int(layer_mapping[name]["weight_numel"]) for name in names)
        report[group] = {"layer_count": len(names), "weight_parameters": params, "methods": {}}
        for method in METHODS:
            bits = method_bits[method]
            bit_counts = {str(value): sum(bits[name] == value for name in names) for value in (4, 8, 32)}
            param_counts = {str(value): sum(int(layer_mapping[name]["weight_numel"])
                                           for name in names if bits[name] == value)
                            for value in (4, 8, 32)}
            bit_sum = sum(int(layer_mapping[name]["weight_numel"]) * bits[name] for name in names)
            report[group]["methods"][method] = {
                "layer_count_by_precision": bit_counts,
                "weight_parameters_by_precision": param_counts,
                "mean_layer_bitwidth": sum(bits[name] for name in names) / len(names),
                "parameter_weighted_mean_bitwidth": bit_sum / params,
            }
    return report


def plot_heatmap(layer_mapping, method_bits):
    ordered_layers = []
    for group in ("encoder", "bottleneck", "decoder", "output"):
        ordered_layers.extend(name for name, item in layer_mapping.items() if item["module"] == group)
    matrix = np.asarray([[method_bits[method][name] for name in ordered_layers] for method in METHODS])
    cmap = ListedColormap(["#168C85", "#E5A33D", "#5C6470"])
    norm = BoundaryNorm([2, 6, 20, 33], cmap.N)
    fig, axis = plt.subplots(figsize=(17, 5.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_yticks(np.arange(len(METHODS)), METHODS)
    axis.set_xticks(np.arange(len(ordered_layers)), ordered_layers, rotation=65, ha="right", fontsize=8)
    axis.set_title("Conv2d precision assignment by method")
    axis.set_xlabel("Conv2d layer (grouped by U-Net stage)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, "FP32" if matrix[row, col] == 32 else str(int(matrix[row, col])),
                      ha="center", va="center", fontsize=7,
                      color="white" if matrix[row, col] in (4, 32) else "#222222")
    boundaries = []
    cursor = 0
    for group in ("encoder", "bottleneck", "decoder", "output"):
        count = sum(item["module"] == group for item in layer_mapping.values())
        cursor += count
        boundaries.append(cursor - 0.5)
        if cursor < len(ordered_layers):
            axis.axvline(cursor - 0.5, color="white", linewidth=2)
    cbar = fig.colorbar(image, ax=axis, ticks=[4, 8, 32], shrink=0.8)
    cbar.ax.set_yticklabels(["INT4", "INT8", "FP32"])
    fig.savefig(REPORT_DIR / "bitwidth_heatmap.png", dpi=180, facecolor="white")
    plt.close(fig)


def scatter_plot(rows, x_key, y_key, xlabel, ylabel, output_name, xlog=False):
    fig, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    palette = ["#4B6078", "#367FA3", "#C75B46", "#DA9B2F", "#637C4D", "#168C85"]
    label_offsets = {
        "FP32": (-42, 8),
        "Full INT8": (8, 8),
        "Full INT4": (-88, 10),
        "SA-INT4": (8, -16),
        "Manual Encoder INT8 + Decoder INT4": (8, 8),
        "StructuralTrainableBits": (-145, 8),
    }
    if y_key == "Accuracy":
        label_offsets["StructuralTrainableBits"] = (8, -20)
    for color, row in zip(palette, rows):
        x, y = float(row[x_key]), float(row[y_key])
        axis.scatter(x, y, s=90, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        axis.annotate(row["method"], (x, y), xytext=label_offsets[row["method"]],
                      textcoords="offset points", fontsize=8)
    axis.margins(x=0.20, y=0.14)
    if xlog:
        axis.set_xscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22)
    axis.set_title(ylabel + " vs " + xlabel)
    fig.savefig(REPORT_DIR / output_name, dpi=180, facecolor="white")
    plt.close(fig)


def read_multiseed_ablation_context():
    path = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_full_ablation_multiseed" / "mean_std.csv"
    summaries = {"full": {}, "no_struct": {}}
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            if row["metric"] in ("Thin-vessel recall", "clDice", "Skeleton Dice"):
                summaries[row["method"]][row["metric"]] = {
                    "mean": float(row["mean"]),
                    "std": float(row["sample_std_ddof_1"]),
                }
    return summaries


def write_readme(rows, group_summary, ablation_context):
    method_row = {row["method"]: row for row in rows}
    manual = method_row["Manual Encoder INT8 + Decoder INT4"]
    structural = method_row["StructuralTrainableBits"]
    int4 = method_row["Full INT4"]
    thin_full = ablation_context["full"]["Thin-vessel recall"]
    thin_no = ablation_context["no_struct"]["Thin-vessel recall"]
    cl_full = ablation_context["full"]["clDice"]
    cl_no = ablation_context["no_struct"]["clDice"]
    readme = """# StructuralTrainableBits 位宽分配消融报告

## 范围与来源

本报告仅汇总列出的已有缓存性能文件、结构指标 CSV、逐层 `layer_assignment.json`、StructuralTrainableBits `layer_mapping.json` 和 best checkpoint。没有运行训练、test.py、推理或量化校准。六种方法的测试指标直接取自已保存结果，均按各实验记录的 FOV 口径；StructuralTrainableBits 使用 v1_long 中 best epoch 16 的 checkpoint 和已保存 test 结果。

## 关键差异

统一 INT4 将 23 个 Conv2d 层全部压到 4 bit，理论 Conv2d payload 为 {i4_bytes:,} bytes、压缩率 {i4_ratio:.3f}x，但缓存测试中 Dice/F1 为 {i4_dice:.6f}、thin-vessel recall 为 {i4_thin:.6f}、clDice 为 {i4_cl:.6f}。SA-INT4 使用相同的全 INT4 层分配；其校准策略不同，但保存的结构指标与普通 INT4 接近。

手工 Encoder INT8 + Decoder INT4 实际上保留 Bottleneck 和 Output 为 FP32：逐层 assignment 对应 Encoder 8 层 INT8、Bottleneck 2 层 FP32、Decoder 12 层中 10 层 INT4/2 层 FP32、Output 1 层 FP32。其平均位宽 {manual_mean:.3f} bit、参数量加权位宽 {manual_weighted:.3f} bit；由于大规模 bottleneck 权重留在 FP32，理论 Conv2d payload 为 {manual_bytes:,} bytes，压缩率约 {manual_ratio:.3f}x。这个命名配置不是单纯的全网两档混合量化。

StructuralTrainableBits 按每层学习到的概率离散选择 INT4/INT8，不把层 precision 统一成固定分组。该缓存 checkpoint 的平均位宽为 {struct_mean:.3f} bit，参数量加权位宽 {struct_weighted:.3f} bit，理论 Conv2d payload {struct_bytes:,} bytes、压缩率 {struct_ratio:.3f}x。相比统一 INT4，它以更高理论权重 payload 换取明显较高的缓存 Dice/F1、thin-vessel recall 和 clDice；相比手工配置，自动分配的理论 payload 更小，同时该缓存结果的结构指标更高。以上是跨既有实验的描述性比较，不构成因果或普遍优势证明。

## 位宽分布与结构损失消融

`allocation_summary.json` 按 Encoder、Bottleneck、Decoder、Output 列出每种方法的逐组 INT4/INT8/FP32 层数、参数量和组内平均位宽。热图中的 FP32、INT8、INT4 表示配置精度，不表示实际整数硬件算子。

结构损失消融结果改善有限，应谨慎表述。seed2026 单次消融中完整方法 thin-vessel recall 为 0.590713，no_struct 为 0.571348；clDice 为 0.817262 与 0.815340，而 Skeleton Dice 略低于 no_struct。另有既有三 seed 配对缓存结果：完整方法 thin-vessel recall 均值 {thin_full_mean:.6f}±{thin_full_std:.6f}，no_struct 为 {thin_no_mean:.6f}±{thin_no_std:.6f}；clDice 分别为 {cl_full_mean:.6f}±{cl_full_std:.6f} 与 {cl_no_mean:.6f}±{cl_no_std:.6f}。三对结果方向一致，但差距较小，且基于同一固定划分；不能据此声称对所有指标、数据集或部署条件都优越。Skeleton Dice 的多 seed 均值也未显示完整方法有一致改善。

## 指标图解释

- `accuracy_vs_bitwidth.png` 横轴是 23 层 Conv2d 的算术平均 bitwidth，纵轴为缓存 Accuracy。
- `thin_vessel_vs_compression.png` 与 `cldice_vs_compression.png` 横轴为理论权重压缩率，纵轴分别为薄血管召回率与 clDice。

## 量化和存储限制

所有量化实验均为 **fake quantization**，以 FP32 算术模拟量化误差，不等于真实整数算子执行。表中权重大小和压缩率是 **理论 Conv2d 权重 payload**：按各层元素数乘分配 bitwidth 计算，排除 scale、bias、其他参数、激活、打包元数据和运行时开销。尚未验证真实 TensorRT INT4/INT8 推理速度、硬件 kernel、功耗或部署兼容性；不得将理论压缩率表述为实测模型体积或速度提升。

StructuralTrainableBits 训练期间冻结 FP32 权重，只优化 23 个 per-Conv2d bitwidth logits。所有指标为现有缓存结果，本报告未重新推理。
""".format(
        i4_bytes=int4["theoretical_conv_weight_bytes"], i4_ratio=int4["theoretical_compression_ratio_vs_fp32_conv_weights"],
        i4_dice=int4["Dice/F1"], i4_thin=int4["Thin-vessel recall"], i4_cl=int4["clDice"],
        manual_mean=manual["mean_layer_bitwidth"], manual_weighted=manual["parameter_weighted_mean_bitwidth"],
        manual_bytes=manual["theoretical_conv_weight_bytes"], manual_ratio=manual["theoretical_compression_ratio_vs_fp32_conv_weights"],
        struct_mean=structural["mean_layer_bitwidth"], struct_weighted=structural["parameter_weighted_mean_bitwidth"],
        struct_bytes=structural["theoretical_conv_weight_bytes"], struct_ratio=structural["theoretical_compression_ratio_vs_fp32_conv_weights"],
        thin_full_mean=thin_full["mean"], thin_full_std=thin_full["std"], thin_no_mean=thin_no["mean"], thin_no_std=thin_no["std"],
        cl_full_mean=cl_full["mean"], cl_full_std=cl_full["std"], cl_no_mean=cl_no["mean"], cl_no_std=cl_no["std"],
    )
    (REPORT_DIR / "README.md").write_text(readme)


def main():
    layer_mapping, assignments, metrics, best, structural_test = load_inputs()
    rows, method_bits, total_parameters, fp32_conv_bytes = derive_method_tables(layer_mapping, assignments, metrics)
    group_summary = group_allocation_summary(layer_mapping, method_bits)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bit_rows = []
    for method in METHODS:
        bit_count = method_bits[method]
        probability = best["p_l"] if method == "StructuralTrainableBits" else {}
        for layer_name, metadata in layer_mapping.items():
            bit_rows.append({
                "method": method,
                "layer": layer_name,
                "group": metadata["module"],
                "weight_numel": metadata["weight_numel"],
                "p_int8": probability.get(layer_name, ""),
                "precision": {32: "FP32", 8: "INT8", 4: "INT4"}[bit_count[layer_name]],
                "bitwidth": bit_count[layer_name],
            })

    with open(REPORT_DIR / "bit_allocation_comparison.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(REPORT_DIR / "layer_bitwidth_comparison.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(bit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bit_rows)

    ablation_path = EXPERIMENTS / "DRIVE_Unet_StructuralTrainableBits_full_ablation" / "ablation_results.csv"
    with open(ablation_path, newline="") as source:
        ablation_rows = list(csv.DictReader(source))
    full_ablation = next(row for row in ablation_rows if row["arm"] == "full")
    no_struct_ablation = next(row for row in ablation_rows if row["arm"] == "no_struct")
    multiseed_stats = read_multiseed_ablation_context()
    allocation_summary = {
        "methods": rows,
        "conv2d_layer_count": len(layer_mapping),
        "total_conv2d_weight_parameters": total_parameters,
        "fp32_conv2d_weight_payload_bytes": fp32_conv_bytes,
        "group_allocations": group_summary,
        "structural_trainable_bits_best_checkpoint": str(LONG_DIR / "best_checkpoint.pt"),
        "structural_trainable_bits_best_epoch": int(best["epoch"]),
        "full_method_cache_test_inference_passes": int(structural_test["inference_passes"]),
        "structural_loss_ablation_seed2026": {
            "full": {key: float(full_ablation[key]) for key in METRIC_COLUMNS},
            "no_struct": {key: float(no_struct_ablation[key]) for key in METRIC_COLUMNS},
            "difference_no_struct_minus_full": {
                key: float(no_struct_ablation[key]) - float(full_ablation[key]) for key in METRIC_COLUMNS
            },
            "multiseed_full_means_and_std": multiseed_stats,
            "interpretation": "Cached ablation gains for thin-vessel recall/clDice are modest; skeleton Dice is mixed. No universal superiority claim is made.",
        },
        "storage_and_quantization_caveat": "All quantized methods are fake quant with FP32 arithmetic. Compression is theoretical Conv2d weight payload only.",
        "all_test_metrics_from_existing_cache": True,
        "training_or_inference_executed": False,
    }
    (REPORT_DIR / "allocation_summary.json").write_text(json.dumps(allocation_summary, indent=2))

    plot_heatmap(layer_mapping, method_bits)
    scatter_plot(rows, "mean_layer_bitwidth", "Accuracy", "Arithmetic mean Conv2d bitwidth", "Accuracy (FOV)", "accuracy_vs_bitwidth.png")
    scatter_plot(rows, "theoretical_compression_ratio_vs_fp32_conv_weights", "Thin-vessel recall",
                 "Theoretical Conv2d weight compression ratio", "Thin-vessel recall (FOV)", "thin_vessel_vs_compression.png")
    scatter_plot(rows, "theoretical_compression_ratio_vs_fp32_conv_weights", "clDice",
                 "Theoretical Conv2d weight compression ratio", "clDice (FOV)", "cldice_vs_compression.png")
    write_readme(rows, group_summary, multiseed_stats)

    print("Bit allocation report generated from cached artifacts only:", REPORT_DIR)
    print("Methods and sizes:")
    for row in rows:
        print(row["method"], "mean_bits={:.3f}".format(row["mean_layer_bitwidth"]),
              "weighted_bits={:.3f}".format(row["parameter_weighted_mean_bitwidth"]),
              "payload_bytes={}".format(row["theoretical_conv_weight_bytes"]),
              "ratio={:.3f}x".format(row["theoretical_compression_ratio_vs_fp32_conv_weights"]))
    print("Output files:", sorted(path.name for path in REPORT_DIR.iterdir()))


if __name__ == "__main__":
    main()