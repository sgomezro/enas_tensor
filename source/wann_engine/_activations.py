import math

import torch

# ----------------------------------------------------------------------------
# Activation functions (the WANN set)
# ----------------------------------------------------------------------------
ACT_NAMES = ["linear", "step", "sin", "gauss", "tanh",
             "sigmoid", "inverse", "abs", "relu", "cos"]
K_ACT = len(ACT_NAMES)
SHARED_WEIGHTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def apply_activations(pre: torch.Tensor, act_masks: torch.Tensor) -> torch.Tensor:
    """pre: (P,R,N); act_masks: (K,P,1,N) bool. Returns per-node activation."""
    out = pre                                                     # 0 linear
    out = torch.where(act_masks[1], (pre > 0).to(pre.dtype), out)  # 1 step
    out = torch.where(act_masks[2], torch.sin(math.pi * pre), out)  # 2 sin
    out = torch.where(act_masks[3], torch.exp(-0.5 * pre * pre), out)  # 3 gauss
    out = torch.where(act_masks[4], torch.tanh(pre), out)          # 4 tanh
    out = torch.where(act_masks[5], 0.5 * (torch.tanh(0.5 * pre) + 1), out)  # 5 sigmoid
    out = torch.where(act_masks[6], -pre, out)                     # 6 inverse
    out = torch.where(act_masks[7], pre.abs(), out)                # 7 abs
    out = torch.where(act_masks[8], torch.relu(pre), out)          # 8 relu
    out = torch.where(act_masks[9], torch.cos(math.pi * pre), out)  # 9 cos
    return out


SCALAR_ACTS = [
    lambda x: x, lambda x: float(x > 0), lambda x: math.sin(math.pi * x),
    lambda x: math.exp(-0.5 * x * x), math.tanh,
    lambda x: 0.5 * (math.tanh(0.5 * x) + 1), lambda x: -x, abs,
    lambda x: max(0.0, x), lambda x: math.cos(math.pi * x),
]
