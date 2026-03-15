"""
Lightweight fallback implementation of the simple_knn API used by the
original Gaussian Splatting code. This avoids requiring the compiled
CUDA extension while providing a compatible interface for distCUDA2.

This implementation is intentionally simple and runs on whatever device
the input tensor lives on. It is sufficient for small test scenes and
for any code paths that only need approximate pairwise distances.
"""

from typing import Union

import torch


def distCUDA2(points: Union[torch.Tensor, "torch.cuda.Tensor"]) -> torch.Tensor:
    """
    Approximate replacement for simple_knn._C.distCUDA2.

    Args:
        points: Tensor of shape (N, 3) containing 3D points.

    Returns:
        Tensor of shape (N,) with the squared distance to the nearest
        neighbor for each point (excluding self), computed with torch.cdist.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"distCUDA2 expects input of shape (N, 3), got {points.shape}")

    # Compute pairwise distances; for large N this is O(N^2) and should
    # only be used for small test scenes.
    dists = torch.cdist(points, points, p=2.0)
    # Ignore self-distance by setting diagonal to a large value
    diag_mask = torch.eye(dists.shape[0], device=dists.device, dtype=torch.bool)
    dists.masked_fill_(diag_mask, float("inf"))
    # Nearest-neighbor squared distance
    nn_dists2, _ = torch.min(dists ** 2, dim=1)
    return nn_dists2

