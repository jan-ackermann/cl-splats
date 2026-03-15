import torch
from transformers import pipeline
from PIL import Image as PILImage

from clsplats.lifter.base_lifter import BaseLifter
from clsplats.utils.custom_types import Image
from clsplats.dataset.cameras import Camera
from clsplats.representation.cl_gaussians import CLGaussians


class DepthAnythingLifter(BaseLifter):
    """
    Lifter that estimates monocular depth using Depth-Anything v2 from HuggingFace
    and lifts 2D change masks into a per-gaussian change mask using multi-view
    depth backprojection and simple Gaussian proximity.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = getattr(
            cfg.lifter, "depth_model", "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self._pipe = pipeline(task="depth-estimation", model=model_name)

        # Lifting hyperparameters
        self.k_nn = getattr(cfg.lifter, "k_nn", 8)
        self.local_radius_thresh = getattr(cfg.lifter, "local_radius_thresh", 2.5)
        self.depth_tol_abs = getattr(cfg.lifter, "depth_tol_abs", 0.05)
        self.depth_tol_rel = getattr(cfg.lifter, "depth_tol_rel", 0.05)
        self.lambda_seed = getattr(cfg.lifter, "lambda_seed", 2.0)
        self.lambda_neg = getattr(cfg.lifter, "lambda_neg", 0.25)
        self.min_visible_views = getattr(cfg.lifter, "min_visible_views", 2)
        self.min_positive_views = getattr(cfg.lifter, "min_positive_views", 2)
        self.min_seed_views = getattr(cfg.lifter, "min_seed_views", 1)
        self.min_positive_ratio = getattr(cfg.lifter, "min_positive_ratio", 0.3)
        self.final_thresh = getattr(cfg.lifter, "final_thresh", 0.6)

    @torch.no_grad()
    def estimate_depth(self, observation: Image) -> torch.Tensor:
        """
        Estimate depth from an observation image tensor [H, W, 3] in [0,1].
        Returns a depth map as a float32 tensor [H, W] normalized to [0,1].
        """
        obs_np = (observation.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        pil_img = PILImage.fromarray(obs_np)

        depth_output = self._pipe(pil_img)["depth"]  # PIL Image
        depth_tensor = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(depth_output.tobytes()))
            .view(depth_output.size[1], depth_output.size[0])
            .float()
        )
        depth_tensor = depth_tensor / 255.0
        return depth_tensor

    @torch.no_grad()
    def lift(
        self,
        gaussians: CLGaussians,
        cameras: "list[Camera]",
        change_masks: "list[torch.Tensor]",
    ) -> torch.Tensor:
        """
        Multi-view lifting:
        - For each view, estimate depth with Depth-Anything.
        - For masked pixels, backproject to 3D and assign evidence to nearby gaussians.
        - Accumulate multi-view positive/negative evidence.
        - Return a boolean mask over gaussians indicating which ones are changed.

        This implements a depth-seeded version of the lifting plan without
        explicit rasterization-based spreading; that can be added later.
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
        scales = gaussians.params.scales    # (N, 3)

        for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
            # Observation image [C, H, W] -> [H, W, 3]
            obs = cam.original_image.permute(1, 2, 0).contiguous()
            depth = self.estimate_depth(obs).to(device)  # [H, W]

            H, W = depth.shape
            mask = mask.to(device)

            # Track visibility simply by kNN around camera center ray endpoints
            # (approximate: treat all gaussians within a frustum as visible)
            # Here we only update per-view counts when gaussians receive evidence.

            # Positive pixels (changed)
            pos_pixels = (mask > 0.5) & torch.isfinite(depth) & (depth > 0)
            if pos_pixels.any():
                ys, xs = torch.nonzero(pos_pixels, as_tuple=True)
                d = depth[ys, xs]

                # Backproject to camera coordinates
                x_cam = (xs.float() - cam.cx) / cam.fx * d
                y_cam = (ys.float() - cam.cy) / cam.fy * d
                z_cam = d

                ones = torch.ones_like(z_cam)
                p_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # [M, 4]
                Twc = cam.Twc.to(device)  # [4, 4]
                p_world_h = p_cam @ Twc.T  # [M, 4]
                p_world = p_world_h[..., :3] / p_world_h[..., 3:].unsqueeze(-1)

                # Brute-force kNN in Gaussian means
                # distances: [M, N]
                dists = torch.cdist(p_world, means)
                knn_dists, knn_idx = torch.topk(
                    dists, k=min(self.k_nn, N), dim=-1, largest=False
                )

                # Local scale-aware distance
                local_scales = scales[knn_idx]  # [M, k, 3]
                denom = local_scales.norm(dim=-1) + 1e-6  # [M, k]
                d_local = knn_dists / denom

                valid = d_local < self.local_radius_thresh  # [M, k]
                if valid.any():
                    # Depth consistency: project means to camera space
                    means_h = torch.cat(
                        [means[None].expand(p_world.shape[0], -1, -1),
                         torch.ones(p_world.shape[0], N, 1, device=device)],
                        dim=-1,
                    )  # [M, N, 4]
                    Tcw = torch.inverse(Twc)[None]  # [1, 4, 4]
                    means_cam = means_h @ Tcw.transpose(1, 2)  # [M, N, 4]
                    z_means = means_cam[..., 2]  # [M, N]

                    # Gather candidate depths for selected knn indices
                    z_knn = torch.gather(z_means, 1, knn_idx)  # [M, k]
                    depth_pix = d.unsqueeze(-1)  # [M, 1]
                    depth_ok = (
                        (z_knn - depth_pix).abs()
                        < (self.depth_tol_abs + self.depth_tol_rel * depth_pix)
                    )

                    valid_final = valid & depth_ok  # [M, k]
                    if valid_final.any():
                        # Distance-based soft weighting
                        d_local_valid = d_local.masked_fill(~valid_final, 1e9)
                        weights = torch.exp(-0.5 * d_local_valid ** 2)
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

                        # Update per-view stats
                        affected = torch.unique(flat_idx)
                        positive_views[affected] += 1
                        seed_views[affected] += 1
                        visible_views[affected] += 1

            # Weak negatives from unmasked pixels (subsampled)
            neg_pixels = (~pos_pixels) & torch.isfinite(depth) & (depth > 0)
            if neg_pixels.any():
                ys_n, xs_n = torch.nonzero(neg_pixels, as_tuple=True)
                # Subsample to keep it cheap
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
                p_world_h_n = p_cam_n @ Twc.T
                p_world_n = p_world_h_n[..., :3] / p_world_h_n[..., 3:].unsqueeze(-1)

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
                    weights_n = torch.exp(-0.5 * d_local_valid_n ** 2)
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

        # Combine evidence
        pos = self.lambda_seed * seed_score
        pos_norm = self.lambda_seed * seed_votes + 1e-8
        pos_conf = pos / pos_norm

        neg = self.lambda_neg * neg_score
        neg_norm = self.lambda_neg * neg_votes + 1e-8
        neg_conf = neg / neg_norm

        score = pos / (pos + neg + 1e-8)

        # Multi-view consistency filtering
        keep = (
            (visible_views >= self.min_visible_views)
            & (positive_views >= self.min_positive_views)
            & (seed_views >= self.min_seed_views)
            & (positive_views.float() / (visible_views.float() + 1e-8)
               >= self.min_positive_ratio)
        )
        score = torch.where(keep, score, torch.zeros_like(score))

        changed_gaussians = score > self.final_thresh
        return changed_gaussians

