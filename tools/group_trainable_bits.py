"""MVP channel-group structural mixed-precision fake quantization.

This module is intentionally independent of the existing layer-level baseline.
It provides G=8 output-channel groups, differentiable INT4/INT8 mixing, and
budget statistics. It does not implement skeleton/branch sensitivity or QAT.
"""
from collections import OrderedDict
from typing import Dict, Iterable, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _symmetric_weight_fake_quant(weight: Tensor, qmax: int) -> Tensor:
    scale = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = torch.clamp(scale / float(qmax), min=torch.finfo(weight.dtype).eps)
    q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    return (q * scale).detach() + weight - weight.detach()


def _tensor_fake_quant(value: Tensor, qmin: int, qmax: int) -> Tensor:
    scale = torch.clamp(value.detach().abs().amax(), min=torch.finfo(value.dtype).eps)
    scale = scale / float(qmax)
    q = torch.clamp(torch.round(value / scale), qmin, qmax)
    return (q * scale).detach() + value - value.detach()


class GroupTrainableBitConv2d(nn.Module):
    """Conv2d wrapper with one independent bit logit per output-channel group."""

    def __init__(self, conv: nn.Conv2d, group_size: int = 8) -> None:
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("conv must be nn.Conv2d")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.group_size = int(group_size)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.weight = conv.weight
        self.bias = conv.bias
        self.num_groups = (self.out_channels + self.group_size - 1) // self.group_size
        self.bit_logit = nn.Parameter(torch.zeros(self.num_groups))
        self.group_slices = tuple(
            (start, min(start + self.group_size, self.out_channels))
            for start in range(0, self.out_channels, self.group_size)
        )

    def bit_probability(self) -> Tensor:
        return torch.sigmoid(self.bit_logit)

    def continuous_bits(self) -> Tensor:
        return 4.0 + 4.0 * self.bit_probability()

    def discrete_bits(self) -> Tensor:
        return torch.where(self.bit_probability() >= 0.5, 8, 4).to(torch.int64)

    def parameter_weighted_average_bits(self) -> Tensor:
        counts = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        group_sizes = self.bit_logit.new_tensor(
            [end - start for start, end in self.group_slices]
        )
        return (self.continuous_bits() * group_sizes).sum() / (group_sizes.sum() + 1e-12)

    def theoretical_bit_budget(self, hard: bool = False) -> Tensor:
        bits = self.discrete_bits().to(self.bit_logit.dtype) if hard else self.continuous_bits()
        group_sizes = self.bit_logit.new_tensor(
            [end - start for start, end in self.group_slices]
        )
        elements_per_channel = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        return (bits * group_sizes).sum() * elements_per_channel

    def forward(self, inputs: Tensor) -> Tensor:
        x4 = _tensor_fake_quant(inputs, -8, 7)
        x8 = _tensor_fake_quant(inputs, -128, 127)
        w4 = _symmetric_weight_fake_quant(self.weight, 7)
        w8 = _symmetric_weight_fake_quant(self.weight, 127)
        outputs = []
        probabilities = self.bit_probability()
        for group_index, (start, end) in enumerate(self.group_slices):
            y4 = F.conv2d(
                x4, w4[start:end],
                None if self.bias is None else self.bias[start:end],
                self.stride, self.padding, self.dilation, self.groups,
            )
            y8 = F.conv2d(
                x8, w8[start:end],
                None if self.bias is None else self.bias[start:end],
                self.stride, self.padding, self.dilation, self.groups,
            )
            p = probabilities[group_index]
            outputs.append((1.0 - p) * y4 + p * y8)
        return torch.cat(outputs, dim=1)


def replace_conv_layers_with_groups(
    model: nn.Module, group_size: int = 8
) -> "OrderedDict[str, GroupTrainableBitConv2d]":
    wrappers = OrderedDict()
    for name, module in list(model.named_children()):
        full_name = name
        if isinstance(module, nn.Conv2d):
            wrapper = GroupTrainableBitConv2d(module, group_size)
            setattr(model, name, wrapper)
            wrappers[full_name] = wrapper
        else:
            child_wrappers = replace_conv_layers_with_groups(module, group_size)
            for child_name, wrapper in child_wrappers.items():
                wrappers[f"{name}.{child_name}"] = wrapper
    return wrappers


def group_statistics(
    wrappers: Iterable[GroupTrainableBitConv2d], hard: bool = False
) -> Dict[str, float]:
    wrappers = list(wrappers)
    total_params = sum(w.weight.numel() for w in wrappers)
    if total_params == 0:
        raise ValueError("no wrapped convolution parameters")
    total_bits = sum(float(w.theoretical_bit_budget(hard).detach()) for w in wrappers)
    total_channels = sum(w.out_channels for w in wrappers)
    weighted_bits = sum(
        float(w.parameter_weighted_average_bits().detach()) * w.weight.numel()
        for w in wrappers
    ) / total_params
    return {
        "group_count": float(sum(w.num_groups for w in wrappers)),
        "conv_count": float(len(wrappers)),
        "parameter_weighted_average_bits": weighted_bits,
        "theoretical_bit_budget": total_bits,
        "parameter_count": float(total_params),
        "output_channel_count": float(total_channels),
    }

