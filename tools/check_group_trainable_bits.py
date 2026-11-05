"""Synthetic sanity check for the MVP group quantization wrapper.

No DRIVE paths, manifests, checkpoints, training loops, or test data are used.
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import U_Net
from tools.group_trainable_bits import replace_conv_layers_with_groups


def main():
    torch.manual_seed(0)
    model = U_Net(img_ch=1, output_ch=2)
    model.eval()
    reference = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        reference_output = model(reference)
    wrappers = replace_conv_layers_with_groups(model, group_size=8)
    assert wrappers, "no Conv2d layers were replaced"
    assert all(wrapper.group_size == 8 for wrapper in wrappers.values())
    expected_groups = sum(
        (wrapper.out_channels + 7) // 8 for wrapper in wrappers.values()
    )
    actual_groups = sum(wrapper.num_groups for wrapper in wrappers.values())
    assert actual_groups == expected_groups
    output = model(reference)
    assert output.shape == reference_output.shape, (
        output.shape, reference_output.shape
    )
    assert torch.isfinite(output).all(), "forward produced non-finite values"
    for wrapper in wrappers.values():
        wrapper.weight.requires_grad_(False)
        if wrapper.bias is not None:
            wrapper.bias.requires_grad_(False)
    loss = output.square().mean()
    loss.backward()
    for name, wrapper in wrappers.items():
        assert wrapper.bit_logit.grad is not None, f"no bit gradient: {name}"
        assert wrapper.weight.grad is None, f"FP32 weight received grad: {name}"
        if wrapper.bias is not None:
            assert wrapper.bias.grad is None, f"bias received grad: {name}"
        probabilities = wrapper.bit_probability().detach()
        assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
    stats = {
        "conv_count": len(wrappers),
        "group_count": actual_groups,
        "output_shape": tuple(output.shape),
        "parameter_weighted_average_bits": float(
            sum(w.parameter_weighted_average_bits().detach() * w.weight.numel()
                for w in wrappers.values())
            / sum(w.weight.numel() for w in wrappers.values())
        ),
    }
    print("GROUP_TRAINABLE_BITS_SANITY_OK")
    print(stats)


if __name__ == "__main__":
    main()

