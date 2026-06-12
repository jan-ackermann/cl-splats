"""Depth-Anything V2 lifter.

Estimates monocular depth using Depth-Anything V2 from HuggingFace and
lifts 2-D change masks into a per-Gaussian change mask using multi-view
depth back-projection and Gaussian proximity scoring.
"""

from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image as PILImage

from clsplats.config import CLSplatsConfig
from clsplats.dataset.cameras import Camera
from clsplats.lifter.base_lifter import BaseLifter
from clsplats.representation.cl_gaussians import CLGaussians
from clsplats.utils.custom_types import Image


def align_relative_depth(
    mono: torch.Tensor,
    rendered_depth: torch.Tensor,
    change_mask: torch.Tensor,
    min_pixels: int = 200,
) -> Optional[torch.Tensor]:
    """Anchor relative monocular depth to the scene's metric scale.

    Depth-Anything (like MiDaS) predicts affine-invariant *disparity*. Fit
    ``1/z ≈ a * mono + b`` by least squares over pixels that are unchanged
    (the existing model is valid there) and covered by the render, then
    return the metric depth ``1 / (a * mono + b)`` for the whole image.

    Returns None when the fit is unusable (too few anchor pixels or a
    non-positive scale).
    """
    valid = (~change_mask) & (rendered_depth > 1e-6) & torch.isfinite(mono)
    if int(valid.sum()) < min_pixels:
        return None

    x = mono[valid].float()
    y = 1.0 / rendered_depth[valid].float()  # target disparity
    ones = torch.ones_like(x)
    A = torch.stack([x, ones], dim=-1)  # [M, 2]
    solution = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    a, b = float(solution[0]), float(solution[1])

    # Reject fits where the mono signal doesn't explain the anchor
    # disparities (R² in disparity space) — a degenerate prediction
    # collapses to the constant fit with R² ≈ 0.
    ss_res = ((a * x + b - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)
    if float(1.0 - ss_res / ss_tot) < 0.3:
        return None

    disparity = (a * mono + b).clamp_min(1e-6)
    return 1.0 / disparity


class DepthAnythingLifter(BaseLifter):
    """Lifter that uses Depth-Anything V2 for monocular depth estimation."""

    def __init__(self, cfg: CLSplatsConfig):
        super().__init__(cfg)
        from transformers import pipeline as hf_pipeline

        model_name = cfg.lifter.depth_model
        self._pipe = hf_pipeline(task="depth-estimation", model=model_name)

        # Lifting hyper-parameters — read directly from typed config
        lcfg = cfg.lifter
        self.k_nn = lcfg.k_nn
        self.local_radius_thresh = lcfg.local_radius_thresh
        self.depth_tol_abs = lcfg.depth_tol_abs
        self.depth_tol_rel = lcfg.depth_tol_rel
        self.lambda_seed = lcfg.lambda_seed
        self.lambda_neg = lcfg.lambda_neg
        self.min_visible_views = lcfg.min_visible_views
        self.min_positive_views = lcfg.min_positive_views
        self.min_seed_views = lcfg.min_seed_views
        self.min_positive_ratio = lcfg.min_positive_ratio
        self.final_thresh = lcfg.final_thresh

    @torch.no_grad()
    def estimate_depth(self, observation: Image) -> torch.Tensor:
        """Estimate depth from an observation image ``[H, W, 3]`` in ``[0, 1]``.

        Returns a depth map as a ``float32`` tensor ``[H, W]`` normalised to
        ``[0, 1]``.
        """
        obs_np = (observation.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        pil_img = PILImage.fromarray(obs_np)

        depth_output = self._pipe(pil_img)["depth"]  # PIL Image

        # Bug 2 fix: convert PIL Image → numpy → tensor properly
        depth_np = np.array(depth_output, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_np)
        depth_tensor = depth_tensor / 255.0
        return depth_tensor

    @torch.no_grad()
    def lift(
        self,
        gaussians: CLGaussians,
        cameras: "list[Camera]",
        change_masks: "list[torch.Tensor]",
        rendered_depths: "Optional[list[torch.Tensor]]" = None,
    ) -> torch.Tensor:
        """Multi-view lifting.

        For each view, estimates depth with Depth-Anything, back-projects
        changed pixels to 3-D, assigns evidence to nearby Gaussians, and
        accumulates multi-view positive/negative evidence.

        Returns:
            Boolean mask ``(N,)`` indicating which Gaussians are changed.
        """
        device = gaussians.params.positions.device
        N = gaussians.params.positions.shape[0]

        seed_score = torch.zeros(N, device=device)
        seed_votes = torch.zeros(N, device=device)
        neg_score = torch.zeros(N, device=device)
        neg_votes = torch.zeros(N, device=device)
        visible_views = torch.zeros(N, dtype=torch.int32, device=device)
        positive_views = torch.zeros(N, dtype=torch.int32, device=device)
        seed_views = torch.zeros(N, dtype=torch.int32, device=device)

        means = gaussians.params.positions  # (N, 3)
        scales = gaussians.params.scales  # (N, 3)

        skipped_views = 0
        for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
            obs = cam.original_image.permute(1, 2, 0).contiguous()
            depth = self.estimate_depth(obs).to(device)  # [H, W], relative
            mask = mask.to(device)

            # Monocular depth is relative — anchor it to the scene's metric
            # scale using the rendered depth of the current model. Without
            # the alignment, back-projections land in empty space and the
            # depth-consistency gate rejects every Gaussian.
            if rendered_depths is not None:
                aligned = align_relative_depth(
                    mono=depth,
                    rendered_depth=rendered_depths[view_id].to(device),
                    change_mask=mask > 0.5,
                )
                if aligned is None:
                    skipped_views += 1
                    continue
                depth = aligned

            H, W = depth.shape

            # --- Positive pixels (changed) ---
            pos_pixels = (mask > 0.5) & torch.isfinite(depth) & (depth > 0)
            if pos_pixels.any():
                ys, xs = torch.nonzero(pos_pixels, as_tuple=True)
                # Sub-sample to avoid OOM in the kNN distance matrix (M×N).
                max_pos = min(ys.numel(), 2048)
                if ys.numel() > max_pos:
                    perm = torch.randperm(ys.numel(), device=device)[:max_pos]
                    ys = ys[perm]
                    xs = xs[perm]
                d = depth[ys, xs]

                # Back-project to camera coordinates
                x_cam = (xs.float() - cam.cx) / cam.fx * d
                y_cam = (ys.float() - cam.cy) / cam.fy * d
                z_cam = d

                ones = torch.ones_like(z_cam)
                p_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # [M, 4]
                # Twc is stored in the 3DGS transposed convention, so row
                # vectors are transformed by right-multiplying it directly.
                Twc = cam.Twc.to(device)  # [4, 4]
                p_world_h = p_cam @ Twc  # [M, 4]
                p_world = p_world_h[..., :3] / p_world_h[..., 3:]  # [M, 3]

                # kNN in Gaussian means — cap M so cdist stays in memory
                dists = torch.cdist(p_world, means)  # [M, N]
                knn_dists, knn_idx = torch.topk(dists, k=min(self.k_nn, N), dim=-1, largest=False)
                del dists  # free immediately

                # Local scale-aware distance
                local_scales = scales[knn_idx]  # [M, k, 3]
                denom = local_scales.norm(dim=-1) + 1e-6  # [M, k]
                d_local = knn_dists / denom

                valid = d_local < self.local_radius_thresh  # [M, k]
                if valid.any():
                    # Depth consistency: project only the k neighbour means (not all N)
                    # into camera space — (M, k, 4) instead of (M, N, 4).
                    Tcw = torch.inverse(Twc)  # [4, 4], transposed convention
                    knn_means = means[knn_idx]  # [M, k, 3]
                    M, k = knn_means.shape[:2]
                    knn_means_h = torch.cat(
                        [knn_means, torch.ones(M, k, 1, device=device)], dim=-1
                    )  # [M, k, 4]
                    knn_cam = knn_means_h @ Tcw  # [M, k, 4]
                    z_knn = knn_cam[..., 2]  # [M, k]

                    depth_pix = d.unsqueeze(-1)  # [M, 1]
                    depth_ok = (z_knn - depth_pix).abs() < (
                        self.depth_tol_abs + self.depth_tol_rel * depth_pix
                    )

                    valid_final = valid & depth_ok  # [M, k]
                    if valid_final.any():
                        d_local_valid = d_local.masked_fill(~valid_final, 1e9)
                        weights = torch.exp(-0.5 * d_local_valid**2)
                        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
                        weights = weights / weights_sum  # [M, k]

                        mask_vals = mask[ys, xs].unsqueeze(-1).float()  # [M, 1]
                        contrib = mask_vals * weights  # [M, k]

                        flat_idx = knn_idx.view(-1)
                        flat_contrib = contrib.view(-1)
                        flat_valid = valid_final.view(-1)

                        flat_idx = flat_idx[flat_valid]
                        flat_contrib = flat_contrib[flat_valid]

                        seed_score.index_add_(0, flat_idx, flat_contrib)
                        seed_votes.index_add_(0, flat_idx, flat_contrib)

                        affected = torch.unique(flat_idx)
                        positive_views[affected] += 1
                        seed_views[affected] += 1
                        visible_views[affected] += 1

            # --- Weak negatives from un-masked pixels (sub-sampled) ---
            neg_pixels = (~pos_pixels) & torch.isfinite(depth) & (depth > 0)
            if neg_pixels.any():
                ys_n, xs_n = torch.nonzero(neg_pixels, as_tuple=True)
                max_neg = min(ys_n.numel(), 1024)
                perm = torch.randperm(ys_n.numel(), device=device)[:max_neg]
                ys_n = ys_n[perm]
                xs_n = xs_n[perm]
                d_n = depth[ys_n, xs_n]

                x_cam_n = (xs_n.float() - cam.cx) / cam.fx * d_n
                y_cam_n = (ys_n.float() - cam.cy) / cam.fy * d_n
                z_cam_n = d_n

                ones_n = torch.ones_like(z_cam_n)
                p_cam_n = torch.stack([x_cam_n, y_cam_n, z_cam_n, ones_n], dim=-1)
                Twc = cam.Twc.to(device)
                p_world_h_n = p_cam_n @ Twc
                # Bug 4 fix: removed incorrect .unsqueeze(-1)
                p_world_n = p_world_h_n[..., :3] / p_world_h_n[..., 3:]

                dists_n = torch.cdist(p_world_n, means)
                knn_dists_n, knn_idx_n = torch.topk(
                    dists_n, k=min(self.k_nn, N), dim=-1, largest=False
                )

                local_scales_n = scales[knn_idx_n]
                denom_n = local_scales_n.norm(dim=-1) + 1e-6
                d_local_n = knn_dists_n / denom_n

                valid_n = d_local_n < self.local_radius_thresh
                if valid_n.any():
                    d_local_valid_n = d_local_n.masked_fill(~valid_n, 1e9)
                    weights_n = torch.exp(-0.5 * d_local_valid_n**2)
                    weights_sum_n = weights_n.sum(dim=-1, keepdim=True) + 1e-8
                    weights_n = weights_n / weights_sum_n

                    mask_vals_n = mask[ys_n, xs_n].unsqueeze(-1).float()
                    contrib_n = (1.0 - mask_vals_n) * weights_n

                    flat_idx_n = knn_idx_n.view(-1)
                    flat_contrib_n = contrib_n.view(-1)
                    flat_valid_n = valid_n.view(-1)

                    flat_idx_n = flat_idx_n[flat_valid_n]
                    flat_contrib_n = flat_contrib_n[flat_valid_n]

                    neg_score.index_add_(0, flat_idx_n, flat_contrib_n)
                    neg_votes.index_add_(0, flat_idx_n, flat_contrib_n)

                    affected_n = torch.unique(flat_idx_n)
                    visible_views[affected_n] += 1

        if skipped_views:
            logger.warning(
                "Depth alignment failed for {n}/{total} views (skipped).",
                n=skipped_views,
                total=len(cameras),
            )

        # Combine evidence
        # Bug 10 fix: confidence thresholding variables were computed but unused; removed.
        pos = self.lambda_seed * seed_score
        neg = self.lambda_neg * neg_score

        score = pos / (pos + neg + 1e-8)

        # Multi-view consistency filtering
        keep = (
            (visible_views >= self.min_visible_views)
            & (positive_views >= self.min_positive_views)
            & (seed_views >= self.min_seed_views)
            & (positive_views.float() / (visible_views.float() + 1e-8) >= self.min_positive_ratio)
        )
        score = torch.where(keep, score, torch.zeros_like(score))

        changed_gaussians = score > self.final_thresh
        return changed_gaussians
