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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from lib.common import setpu_seed
from lib.datasetV2 import TrainDatasetV2, create_patch_idx, data_preprocess
from lib.extract_patches import load_file_path_txt
from lib.losses.loss import CrossEntropyLoss2d
from tools.ptq_int4_baseline import affine_params, make_calibration_patches


OUTPUT_RELATIVE = Path("experiments/DRIVE_Unet_StructuralTrainableBits_v1")
CHECKPOINT_RELATIVE = Path("experiments/DRIVE_Unet_FP32_full_retry/best_model.pth")
TRAIN_MANIFEST_RELATIVE = Path("lists/DRIVE/train_images_seed2026.txt")
VAL_MANIFEST_RELATIVE = Path("lists/DRIVE/val_images_seed2026.txt")
PATCH_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Structural-aware trainable INT4/INT8 bit selection prototype")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--N-patches", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calib-patches", type=int, default=320)
    parser.add_argument("--lr-bits", type=float, default=1e-2)
    parser.add_argument("--lambda-bit", type=float, default=0.001)
    parser.add_argument("--lambda-struct", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=str(OUTPUT_RELATIVE))
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Run synthetic forward, one train batch and one validation batch without writing experiment files",
    )
    return parser.parse_args()


def layer_group(name):
    if name == "Conv_1x1":
        return "output"
    if name.startswith("Conv5."):
        return "bottleneck"
    if name.startswith(("Conv1.", "Conv2.", "Conv3.", "Conv4.")):
        return "encoder"
    if name.startswith(("Up5.", "Up4.", "Up3.", "Up2.", "Up_conv5.", "Up_conv4.", "Up_conv3.", "Up_conv2.")):
        return "decoder"
    raise ValueError("Unmapped Conv2d module: {}".format(name))


class TrainableBitConv2d(nn.Module):
    """Soft interpolation between per-channel INT4 and INT8 fake-quant Conv2d outputs."""

    def __init__(self, conv, activation_min, activation_max, initial_logit=0.0):
        super().__init__()
        self.conv = conv
        for parameter in self.conv.parameters():
            parameter.requires_grad_(False)
        self.bit_logit = nn.Parameter(torch.tensor(float(initial_logit), dtype=torch.float32))
        self.activation_min = float(activation_min)
        self.activation_max = float(activation_max)

        weight = conv.weight.detach().float()
        channel_absmax = weight.abs().amax(dim=(1, 2, 3))
        self.register_buffer("weight_scale_int4", torch.clamp(channel_absmax / 7.0, min=1e-12))
        self.register_buffer("weight_scale_int8", torch.clamp(channel_absmax / 127.0, min=1e-12))
        self.register_buffer("weight_zero_int4", torch.zeros_like(channel_absmax, dtype=torch.int32))
        self.register_buffer("weight_zero_int8", torch.zeros_like(channel_absmax, dtype=torch.int32))
        self.activation_scale_int4, self.activation_zero_int4 = affine_params(
            self.activation_min, self.activation_max, -8, 7
        )
        self.activation_scale_int8, self.activation_zero_int8 = affine_params(
            self.activation_min, self.activation_max, -128, 127
        )

    def bit_probability(self):
        return torch.sigmoid(self.bit_logit)

    def continuous_bits(self):
        return 4.0 + 4.0 * self.bit_probability()

    def forward(self, inputs):
        probability_int8 = self.bit_probability()
        input_int4 = torch.fake_quantize_per_tensor_affine(
            inputs, self.activation_scale_int4, self.activation_zero_int4, -8, 7
        )
        input_int8 = torch.fake_quantize_per_tensor_affine(
            inputs, self.activation_scale_int8, self.activation_zero_int8, -128, 127
        )
        weight_int4 = torch.fake_quantize_per_channel_affine(
            self.conv.weight, self.weight_scale_int4, self.weight_zero_int4, 0, -7, 7
        )
        weight_int8 = torch.fake_quantize_per_channel_affine(
            self.conv.weight, self.weight_scale_int8, self.weight_zero_int8, 0, -127, 127
        )
        output_int4 = F.conv2d(
            input_int4, weight_int4, self.conv.bias, self.conv.stride,
            self.conv.padding, self.conv.dilation, self.conv.groups,
        )
        output_int8 = F.conv2d(
            input_int8, weight_int8, self.conv.bias, self.conv.stride,
            self.conv.padding, self.conv.dilation, self.conv.groups,
        )
        return (1.0 - probability_int8) * output_int4 + probability_int8 * output_int8

    def discrete_bits(self):
        return 8 if float(self.bit_probability().detach()) >= 0.5 else 4


def replace_conv_layers(model, activation_ranges):
    wrappers = {}
    mapping = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            original = getattr(parent, child_name)
            wrapper = TrainableBitConv2d(
                original,
                activation_ranges[name]["min"],
                activation_ranges[name]["max"],
            )
            setattr(parent, child_name, wrapper)
            wrappers[name] = wrapper
            mapping[name] = {
                "module": layer_group(name),
                "weight_shape": list(original.weight.shape),
                "weight_numel": original.weight.numel(),
                "initial_probability_int8": float(wrapper.bit_probability().detach()),
            }
    if len(wrappers) != 23:
        raise RuntimeError("Expected exactly 23 Conv2d layers, found {}".format(len(wrappers)))
    return wrappers, mapping


def make_loaders(args):
    train_manifest = PROJECT_ROOT / TRAIN_MANIFEST_RELATIVE
    val_manifest = PROJECT_ROOT / VAL_MANIFEST_RELATIVE
    train_images, train_masks, train_fovs = data_preprocess(str(train_manifest))
    val_images, val_masks, val_fovs = data_preprocess(str(val_manifest))

    patch_args = argparse.Namespace(
        train_patch_height=PATCH_SIZE,
        train_patch_width=PATCH_SIZE,
        N_patches=args.N_patches,
        inside_FOV="center",
    )
    train_indices = create_patch_idx(train_fovs, patch_args, seed=args.seed)
    val_args = argparse.Namespace(
        train_patch_height=PATCH_SIZE,
        train_patch_width=PATCH_SIZE,
        N_patches=max(args.batch_size, int(args.N_patches * len(val_images) / len(train_images))),
        inside_FOV="center",
    )
    val_indices = create_patch_idx(val_fovs, val_args, seed=args.seed + 1)
    train_set = TrainDatasetV2(train_images, train_masks, train_fovs, train_indices, mode="val", args=patch_args)
    val_set = TrainDatasetV2(val_images, val_masks, val_fovs, val_indices, mode="val", args=val_args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def pixel_losses(probabilities, targets, fovs, criterion):
    fov_mask = fovs > 0
    target_vessel = (targets == 1) & fov_mask
    if not fov_mask.any():
        raise ValueError("Batch has no FOV pixels")
    pixel_ce = -torch.log(torch.clamp(probabilities[:, 1], min=1e-7)) * target_vessel.float()
    pixel_ce += -torch.log(torch.clamp(probabilities[:, 0], min=1e-7)) * (~(targets == 1)).float()
    segmentation_loss = criterion(
        probabilities.permute(0, 2, 3, 1)[fov_mask], targets[fov_mask]
    )

    local_vessel_density = F.avg_pool2d(
        target_vessel.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)
    thin_vessel_proxy = (targets == 1) & fov_mask & (local_vessel_density <= (4.0 / 9.0))
    pixel_weights = torch.ones_like(pixel_ce)
    pixel_weights = torch.where(target_vessel, 2.0, pixel_weights)
    pixel_weights = torch.where(thin_vessel_proxy, 4.0, pixel_weights)
    pixel_weights = pixel_weights * fov_mask.float()
    structural_loss = (pixel_ce * pixel_weights).sum() / torch.clamp(pixel_weights.sum(), min=1.0)
    return segmentation_loss, structural_loss, fov_mask, thin_vessel_proxy


def batch_metrics(probabilities, targets, fovs, thin_proxy):
    fov_mask = fovs > 0
    scores = probabilities[:, 1][fov_mask].detach().cpu().numpy()
    truth = (targets[fov_mask] == 1).detach().cpu().numpy().astype(np.uint8)
    prediction = scores >= 0.5
    tp = int(np.count_nonzero(prediction & (truth == 1)))
    fp = int(np.count_nonzero(prediction & (truth == 0)))
    fn = int(np.count_nonzero(~prediction & (truth == 1)))
    tn = int(np.count_nonzero(~prediction & (truth == 0)))
    thin_mask = thin_proxy & fov_mask
    thin_truth = (targets == 1) & thin_mask
    thin_count = int(thin_truth.sum().item())
    thin_tp = int(((probabilities[:, 1] >= 0.5) & thin_truth).sum().item())
    thin_fp = int(((probabilities[:, 1] >= 0.5) & ~thin_truth & fov_mask).sum().item())
    thin_fn = thin_count - thin_tp
    return {
        "truth": truth,
        "scores": scores,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "thin_tp": thin_tp, "thin_fp": thin_fp, "thin_fn": thin_fn,
    }


def aggregate_metrics(records):
    truth = np.concatenate([record["truth"] for record in records])
    scores = np.concatenate([record["scores"] for record in records])
    tp = sum(record["tp"] for record in records)
    fp = sum(record["fp"] for record in records)
    fn = sum(record["fn"] for record in records)
    tn = sum(record["tn"] for record in records)
    thin_tp = sum(record["thin_tp"] for record in records)
    thin_fp = sum(record["thin_fp"] for record in records)
    thin_fn = sum(record["thin_fn"] for record in records)
    try:
        auc = float(roc_auc_score(truth, scores))
    except ValueError:
        auc = float("nan")
    return {
        "val_loss": None,
        "val_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "val_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "val_auc": auc,
        "val_vessel_weighted_recall": tp / max(tp + fn, 1),
        "val_thin_vessel_recall_proxy": thin_tp / max(thin_tp + thin_fn, 1),
    }


def validate(model, val_loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    pixel_count = 0
    records = []
    with torch.inference_mode():
        for inputs, targets, fovs in val_loader:
            inputs, targets, fovs = inputs.to(device), targets.to(device), fovs.to(device)
            probabilities = model(inputs)
            seg_loss, _, fov_mask, thin_proxy = pixel_losses(probabilities, targets, fovs, criterion)
            count = int(fov_mask.sum().item())
            loss_sum += float(seg_loss.item()) * count
            pixel_count += count
            record = batch_metrics(probabilities, targets, fovs, thin_proxy)
            records.append(record)
    metrics = aggregate_metrics(records)
    metrics["val_loss"] = loss_sum / max(pixel_count, 1)
    return metrics


def save_bit_history_row(history_path, epoch, wrappers, train_loss, validation_metrics, args):
    continuous_bits = [float(layer.continuous_bits().detach()) for layer in wrappers.values()]
    discrete_bits = [layer.discrete_bits() for layer in wrappers.values()]
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": validation_metrics["val_loss"],
        "val_accuracy": validation_metrics["val_accuracy"],
        "val_f1": validation_metrics["val_f1"],
        "val_auc": validation_metrics["val_auc"],
        "val_vessel_weighted_recall": validation_metrics["val_vessel_weighted_recall"],
        "val_thin_vessel_recall_proxy": validation_metrics["val_thin_vessel_recall_proxy"],
        "lambda_bit": args.lambda_bit,
        "lambda_struct": args.lambda_struct,
        "mean_bit_width": float(np.mean(continuous_bits)),
        "mean_continuous_bit_width": float(np.mean(continuous_bits)),
        "mean_discrete_bit_width": float(np.mean(discrete_bits)),
    }
    for name, layer in wrappers.items():
        probability = float(layer.bit_probability().detach().cpu())
        row[name + ".p_int8"] = probability
        row[name + ".continuous_bits"] = 4.0 + 4.0 * probability
        row[name + ".discrete_bits"] = 8 if probability >= 0.5 else 4
    file_exists = history_path.exists()
    with open(history_path, "a", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_train_batch(model, wrappers, optimizer, criterion, batch, device, args):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    inputs, targets, fovs = (value.to(device) for value in batch)
    optimizer.zero_grad(set_to_none=True)
    probabilities = model(inputs)
    seg_loss, struct_loss, fov_mask, thin_proxy = pixel_losses(probabilities, targets, fovs, criterion)
    bit_loss = torch.stack([layer.continuous_bits() for layer in wrappers.values()]).mean()
    total_loss = seg_loss + args.lambda_bit * bit_loss + args.lambda_struct * struct_loss
    total_loss.backward()
    optimizer.step()
    return {
        "loss": float(total_loss.detach()),
        "seg_loss": float(seg_loss.detach()),
        "struct_loss": float(struct_loss.detach()),
        "bit_loss": float(bit_loss.detach()),
        "fov_mask": fov_mask,
        "thin_proxy": thin_proxy,
    }


def main():
    args = parse_args()
    if args.epochs <= 0 or args.N_patches <= 0:
        raise ValueError("Epoch count and training patch count must be positive")
    output_dir_argument = Path(args.output_dir).expanduser()
    output_dir = output_dir_argument if output_dir_argument.is_absolute() else PROJECT_ROOT / output_dir_argument
    output_dir = output_dir.resolve()
    checkpoint_path = PROJECT_ROOT / CHECKPOINT_RELATIVE
    train_manifest = PROJECT_ROOT / TRAIN_MANIFEST_RELATIVE
    val_manifest = PROJECT_ROOT / VAL_MANIFEST_RELATIVE
    if output_dir.exists():
        raise FileExistsError("Prototype output directory exists; refusing to overwrite: {}".format(output_dir))
    for required_path in (checkpoint_path, train_manifest, val_manifest):
        if not required_path.is_file():
            raise FileNotFoundError(str(required_path))
    if args.batch_size <= 0 or args.calib_patches <= 0:
        raise ValueError("Batch size and calibration patch count must be positive")

    train_rows = load_file_path_txt(str(train_manifest))
    val_rows = load_file_path_txt(str(val_manifest))
    if len(train_rows[0]) != 16 or len(val_rows[0]) != 4:
        raise ValueError("Expected the fixed 16/4 train/validation image split")
    if any("/DRIVE/test/" in path for column in train_rows + val_rows for path in column):
        raise ValueError("Test paths cannot participate in train or validation")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or device.index not in (None, 0):
        raise RuntimeError("This prototype requires CUDA GPU 0")
    device = torch.device("cuda:0")
    setpu_seed(args.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("epoch", -1)) != 9:
        raise ValueError("Expected the epoch-9 validation-best FP32 checkpoint")
    base_state = checkpoint["net"]
    model = models.UNetFamily.U_Net(1, 2)
    model.load_state_dict(base_state)
    model.to(device).eval()

    activation_calibration = make_calibration_patches(train_manifest, args.calib_patches, args.seed)
    from tools.ptq_int8_baseline import collect_calibration_stats

    activation_stats, _ = collect_calibration_stats(model, activation_calibration, device, args.batch_size)
    wrappers, layer_mapping = replace_conv_layers(model, activation_stats)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bit_parameters = []
    for wrapper in wrappers.values():
        wrapper.bit_logit.requires_grad_(True)
        bit_parameters.append(wrapper.bit_logit)
    optimizer = torch.optim.Adam(bit_parameters, lr=args.lr_bits)
    criterion = CrossEntropyLoss2d()

    train_images, train_masks, train_fovs = data_preprocess(str(train_manifest))
    val_images, val_masks, val_fovs = data_preprocess(str(val_manifest))
    train_patch_args = argparse.Namespace(
        train_patch_height=PATCH_SIZE, train_patch_width=PATCH_SIZE,
        N_patches=args.N_patches, inside_FOV="center",
    )
    train_indices = create_patch_idx(train_fovs, train_patch_args, seed=args.seed)
    val_patch_args = argparse.Namespace(
        train_patch_height=PATCH_SIZE, train_patch_width=PATCH_SIZE,
        N_patches=max(args.batch_size, args.N_patches // 4), inside_FOV="center",
    )
    val_indices = create_patch_idx(val_fovs, val_patch_args, seed=args.seed + 1)
    train_dataset = TrainDatasetV2(
        train_images, train_masks, train_fovs, train_indices, mode="val", args=train_patch_args
    )
    val_dataset = TrainDatasetV2(
        val_images, val_masks, val_fovs, val_indices, mode="val", args=val_patch_args
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.checks_only:
        with torch.inference_mode():
            synthetic_output = model(torch.randn(2, 1, PATCH_SIZE, PATCH_SIZE, device=device))
        print("Synthetic forward:", tuple(synthetic_output.shape))
        training_batch = next(iter(train_loader))
        update = run_train_batch(model, wrappers, optimizer, criterion, training_batch, device, args)
        print("One train batch:", tuple(training_batch[0].shape), "loss:", update["loss"])
        validation_metrics = validate(model, val_loader, device, criterion)
        print("One validation pass:", validation_metrics)
        print("Bit parameters changed by optimizer step:",
              any(float(layer.bit_logit.detach()) != 0.0 for layer in wrappers.values()))
        print("No experiment files written in checks-only mode")
        return

    output_dir.mkdir(parents=True)
    with open(output_dir / "layer_mapping.json", "w") as mapping_file:
        json.dump(layer_mapping, mapping_file, indent=2)
    shutil.copy2(Path(__file__).resolve(), output_dir / "structural_trainable_bits.py")
    shutil.copy2(train_manifest, output_dir / "calibration_manifest_used.txt")
    config = {
        "method": "Structural-aware trainable soft INT4/INT8 fake quant prototype",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 9,
        "training_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "test_used_for_training_validation_calibration_early_stopping": False,
        "training_images": 16,
        "validation_images": 4,
        "epochs": args.epochs,
        "training_patches": args.N_patches,
        "calibration_patches": args.calib_patches,
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
        "batch_size": args.batch_size,
        "initial_p_l": 0.0,
        "p_l": "sigmoid(logit_l)",
        "continuous_bit_width": "4 + 4 * sigmoid(logit_l)",
        "hard_bit_rule": "p_l >= 0.5 selects INT8; otherwise INT4",
        "training_estimator": "Differentiable soft interpolation: (1-p_l)*Conv(Q4(x),Q4(w)) + p_l*Conv(Q8(x),Q8(w)); fake quant uses STE.",
        "trainable_parameters": "Only per-Conv2d bit logits; FP32 checkpoint weights are frozen.",
        "weight_quantization": "INT4 [-7,7], INT8 [-127,127], per-output-channel symmetric",
        "activation_quantization": "INT4 [-8,7], INT8 [-128,127], per-layer affine ranges calibrated from training patches",
        "L_seg": "masked CrossEntropyLoss2d-style negative log probability over FOV pixels",
        "L_bit": "mean continuous bit-width across 23 Conv2d layers",
        "L_struct": "FOV-masked weighted pixel CE; background=1, vessel=2, thin-vessel proxy=4",
        "thin_vessel_proxy": "GT vessel pixels with 3x3 local vessel fraction <= 4/9; no skeleton labels fabricated",
        "lambda_bit": args.lambda_bit,
        "lambda_struct": args.lambda_struct,
        "optimizer": "Adam over 23 bit logits only",
        "learning_rate_bits": args.lr_bits,
        "amp": False,
        "device": str(device),
        "full_training": False,
    }
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    readme = """# Structural Trainable Bits Prototype

This is a controlled research prototype run, not a completed training recipe. It reuses the fixed DRIVE seed-2026 16/4 original-image split and the epoch-9 FP32 checkpoint. FP32 convolution weights are frozen; only one scalar bit logit per Conv2d is optimized.

For each layer, `p_l = sigmoid(logit_l)` and `bit_l = 4 + 4*p_l`. Training uses differentiable soft interpolation between INT4 and INT8 fake-quant Conv2d outputs with PyTorch fake-quant STE. At export/report time, `p_l >= 0.5` selects INT8; otherwise INT4. All arithmetic remains FP32, so this is simulation and does not imply integer hardware acceleration.

The structural term is a FOV-masked weighted pixel loss using existing vessel masks: background weight 1, vessel weight 2, and a thin-vessel proxy weight 4 for vessel pixels whose 3x3 local vessel fraction is at most 4/9. No skeleton target is fabricated. Validation uses only the four fixed validation originals, and the official test set is not opened by this prototype.

This differs from a generic trainable-bit-width U-Net prototype by adding an explicit retinal-vessel structural pixel-weighting term and FOV masking. It does not yet optimize skeleton Dice or clDice directly.
"""
    (output_dir / "README.md").write_text(readme)

    history_path = output_dir / "bitwidth_history.csv"
    training_started = time.perf_counter()
    epoch_metrics = []
    initial_logits = {name: float(wrapper.bit_logit.detach()) for name, wrapper in wrappers.items()}
    best_val_f1 = float("-inf")
    best_epoch = None

    def checkpoint_payload(epoch, validation_metrics):
        probabilities = {name: float(wrapper.bit_probability().detach().cpu())
                         for name, wrapper in wrappers.items()}
        discrete_bits = {name: 8 if probability >= 0.5 else 4
                         for name, probability in probabilities.items()}
        return {
            "fp32_checkpoint": str(checkpoint_path),
            "fp32_epoch": 9,
            "epoch": epoch,
            "p_l": probabilities,
            "bit_logits": {name: wrapper.bit_logit.detach().cpu() for name, wrapper in wrappers.items()},
            "continuous_bit_width": {name: 4.0 + 4.0 * probability
                                     for name, probability in probabilities.items()},
            "discrete_bit_width": discrete_bits,
            "mean_discrete_bit_width": float(np.mean(list(discrete_bits.values()))),
            "mean_continuous_bit_width": float(np.mean(
                [4.0 + 4.0 * probability for probability in probabilities.values()]
            )),
            "validation_metrics": validation_metrics,
            "activation_calibration_ranges": activation_stats,
            "note": "Load the referenced frozen FP32 checkpoint together with these learned bit logits.",
        }

    def atomic_save(payload, path):
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)

    for epoch in range(1, args.epochs + 1):
        loss_values = []
        for batch in tqdm(train_loader, desc="Train epoch {}/{}".format(epoch, args.epochs)):
            update = run_train_batch(model, wrappers, optimizer, criterion, batch, device, args)
            loss_values.append(update["loss"])
        validation_metrics = validate(model, val_loader, device, criterion)
        epoch_train_loss = float(np.mean(loss_values))
        epoch_metrics.append({"epoch": epoch, "train_loss": epoch_train_loss, **validation_metrics})
        save_bit_history_row(history_path, epoch, wrappers, epoch_train_loss, validation_metrics, args)
        payload = checkpoint_payload(epoch, validation_metrics)
        atomic_save(payload, output_dir / "latest_checkpoint.pt")
        if validation_metrics["val_f1"] > best_val_f1:
            best_val_f1 = validation_metrics["val_f1"]
            best_epoch = epoch
            payload["best_epoch"] = best_epoch
            payload["best_val_f1"] = best_val_f1
            atomic_save(payload, output_dir / "best_checkpoint.pt")
        print("Epoch {} metrics: {}".format(epoch, epoch_metrics[-1]))

    final_probabilities = {name: float(wrapper.bit_probability().detach().cpu())
                           for name, wrapper in wrappers.items()}
    final_bits = {name: 8 if probability >= 0.5 else 4
                  for name, probability in final_probabilities.items()}
    final_mean_bits = float(np.mean(list(final_bits.values())))
    final_mean_continuous_bits = float(np.mean([4.0 + 4.0 * p for p in final_probabilities.values()]))
    final_payload = checkpoint_payload(args.epochs, epoch_metrics[-1])
    final_payload["best_epoch"] = best_epoch
    final_payload["best_val_f1"] = best_val_f1
    atomic_save(final_payload, output_dir / "latest_checkpoint.pt")
    run_seconds = time.perf_counter() - training_started
    with open(output_dir / "training_log.csv", "w", newline="") as train_file:
        writer = csv.DictWriter(train_file, fieldnames=list(epoch_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_metrics)
    (output_dir / "README.md").write_text(readme + "\nSmoke wall time: {:.2f} seconds.\n".format(run_seconds))
    config["final_mean_bit_width"] = final_mean_bits
    config["final_mean_continuous_bit_width"] = final_mean_continuous_bits
    config["bit_logits_changed_from_initial"] = {
        name: final_probabilities[name] != 0.5 or initial_logits[name] != 0.0
        for name in final_probabilities
    }
    config["run_wall_seconds"] = run_seconds
    with open(output_dir / "quant_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    encoder_bits = [final_bits[name] for name, details in layer_mapping.items() if details["module"] == "encoder"]
    bottleneck_bits = [final_bits[name] for name, details in layer_mapping.items() if details["module"] == "bottleneck"]
    decoder_bits = [final_bits[name] for name, details in layer_mapping.items() if details["module"] == "decoder"]
    print("Best epoch:", best_epoch, "best validation F1:", best_val_f1)
    print("Mean final discrete bit-width:", final_mean_bits)
    print("Mean final continuous bit-width:", final_mean_continuous_bits)
    print("Final layer bit-widths:")
    for name in wrappers:
        print("  {}: INT{} (p_int8={:.6f})".format(name, final_bits[name], final_probabilities[name]))
    print("Mean group discrete bits: encoder={:.3f}, bottleneck={:.3f}, decoder={:.3f}".format(
        np.mean(encoder_bits), np.mean(bottleneck_bits), np.mean(decoder_bits)
    ))
    print("Encoder higher than bottleneck and decoder:",
          bool(np.mean(encoder_bits) > np.mean(bottleneck_bits)
               and np.mean(encoder_bits) > np.mean(decoder_bits)))
    print("Best checkpoint:", output_dir / "best_checkpoint.pt")
    print("Latest checkpoint:", output_dir / "latest_checkpoint.pt")
    print("Complete train/validation patches only; no test manifest or images were read")


if __name__ == "__main__":
    main()