"""Shared type aliases for the CL-Splats codebase."""

import torch

# An image tensor with shape [H, W, 3] in float32, values in [0, 1].
Image = torch.Tensor
