"""
Python fallback shim for the simple_knn C++/CUDA extension.

The original Gaussian Splatting code imports:

    from simple_knn._C import distCUDA2

This module provides a compatible distCUDA2 implemented in pure PyTorch
so that the rest of the pipeline can run without compiling the native
extension.
"""

from typing import Union

import torch


def distCUDA2(points: Union[torch.Tensor, "torch.cuda.Tensor"]) -> torch.Tensor:
    """
    Approximate replacement for the CUDA distCUDA2.

    Args:
        points: Tensor of shape (N, 3) containing 3D points.

    Returns:
        Tensor of shape (N,) with the squared distance to the nearest
        neighbor for each point (excluding self), computed with torch.cdist.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"distCUDA2 expects input of shape (N, 3), got {points.shape}")

    dists = torch.cdist(points, points, p=2.0)
    diag_mask = torch.eye(dists.shape[0], device=dists.device, dtype=torch.bool)
    dists.masked_fill_(diag_mask, float("inf"))
    nn_dists2, _ = torch.min(dists ** 2, dim=1)
    return nn_dists2

