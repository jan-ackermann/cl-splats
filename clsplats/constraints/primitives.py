from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class SpherePrimitive:
    center: torch.Tensor  # (3,)
    radius: float


@dataclass
class OBBPrimitive:
    center: torch.Tensor      # (3,)
    rotation: torch.Tensor    # (3, 3)
    half_extents: torch.Tensor  # (3,)


Primitive = Tuple[str, object]


def _robust_center(points: torch.Tensor) -> torch.Tensor:
    """Use simple mean as a robust-enough center for small clusters."""
    return points.mean(dim=0)


def fit_sphere(points: torch.Tensor, quantile: float = 0.95, margin: float = 0.02) -> SpherePrimitive:
    center = _robust_center(points)
    dists = torch.norm(points - center[None, :], dim=-1)
    r = torch.quantile(dists, quantile).item() + margin
    return SpherePrimitive(center=center.detach(), radius=r)


def fit_obb(points: torch.Tensor, quantile_low: float = 0.05, quantile_high: float = 0.95, margin: float = 0.02) -> OBBPrimitive:
    center = _robust_center(points)
    centered = points - center[None, :]
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # Sort eigenvectors by descending eigenvalue
    order = torch.argsort(eigvals, descending=True)
    R = eigvecs[:, order]
    y = (R.T @ centered.T).T  # (N, 3)
    low = torch.quantile(y, quantile_low, dim=0) - margin
    high = torch.quantile(y, quantile_high, dim=0) + margin
    half_extents = 0.5 * (high - low)
    c_box_local = 0.5 * (low + high)
    c_box_world = center + R @ c_box_local
    return OBBPrimitive(center=c_box_world.detach(), rotation=R.detach(), half_extents=half_extents.detach())


def group_active_gaussians(positions: torch.Tensor, active_mask: torch.Tensor, radius_frac: float = 0.1) -> List[torch.Tensor]:
    """Split active Gaussians into connected components in a kNN-like graph.

    Two Gaussians are connected if their distance is below a radius threshold,
    chosen as radius_frac times the scene extent.
    """
    device = positions.device
    idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return []

    pts = positions[idx]  # (M, 3)
    extent = torch.norm(pts.max(dim=0).values - pts.min(dim=0).values)
    if extent == 0:
        # All points collapsed; single group
        return [idx]

    radius = radius_frac * extent.item()
    dists = torch.cdist(pts, pts)  # (M, M)
    adj = dists < radius

    M = pts.shape[0]
    visited = torch.zeros(M, dtype=torch.bool, device=device)
    groups: List[torch.Tensor] = []

    for i in range(M):
        if visited[i]:
            continue
        # BFS to collect a component
        queue = [i]
        component_indices = []
        visited[i] = True
        while queue:
            j = queue.pop()
            component_indices.append(j)
            neighbors = torch.nonzero(adj[j], as_tuple=False).squeeze(-1)
            for n in neighbors.tolist():
                if not visited[n]:
                    visited[n] = True
                    queue.append(n)
        groups.append(idx[torch.tensor(component_indices, device=device)])

    return groups


def fit_primitives_for_active(
    positions: torch.Tensor,
    active_mask: torch.Tensor,
    radius_frac: float = 0.1,
    anisotropy_thresh: float = 2.0,
) -> List[Tuple[torch.Tensor, Primitive]]:
    """Fit one primitive (sphere or OBB) per connected component of active Gaussians.

    Returns:
        List of (indices, primitive) where indices are the Gaussian indices
        belonging to this component and primitive is ("sphere", SpherePrimitive)
        or ("obb", OBBPrimitive).
    """
    groups = group_active_gaussians(positions, active_mask, radius_frac=radius_frac)
    primitives: List[Tuple[torch.Tensor, Primitive]] = []
    for g_idx in groups:
        pts = positions[g_idx]
        if pts.shape[0] < 4:
            prim = ("sphere", fit_sphere(pts))
        else:
            centered = pts - _robust_center(pts)[None, :]
            cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
            eigvals, _ = torch.linalg.eigh(cov)
            ratio = (eigvals.max() / (eigvals.min() + 1e-6)).item()
            if ratio > anisotropy_thresh:
                prim = ("obb", fit_obb(pts))
            else:
                prim = ("sphere", fit_sphere(pts))
        primitives.append((g_idx, prim))
    return primitives


def distance_to_primitive(points: torch.Tensor, primitive: Primitive) -> torch.Tensor:
    """Compute outside distance to a primitive for each point."""
    kind, data = primitive
    if kind == "sphere":
        center = data.center.to(points.device)
        r = data.radius
        d = torch.norm(points - center[None, :], dim=-1) - r
        return torch.relu(d)
    elif kind == "obb":
        center = data.center.to(points.device)
        R = data.rotation.to(points.device)
        e = data.half_extents.to(points.device)
        y = (R.T @ (points - center[None, :]).T).T  # (N, 3)
        q = torch.abs(y) - e[None, :]
        q_clamped = torch.relu(q)
        return torch.norm(q_clamped, dim=-1)
    else:
        raise ValueError(f"Unknown primitive kind: {kind}")


def union_distance(points: torch.Tensor, primitives: List[Primitive]) -> torch.Tensor:
    """Distance to union of primitives: min distance to any primitive."""
    if not primitives:
        return torch.zeros(points.shape[0], device=points.device)
    dists = []
    for prim in primitives:
        d = distance_to_primitive(points, prim)
        dists.append(d)
    stacked = torch.stack(dists, dim=0)  # (K, N)
    return torch.min(stacked, dim=0).values

