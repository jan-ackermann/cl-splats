"""CL-Splats trainer.

Orchestrates the full continual-learning Gaussian Splatting pipeline:
detect → lift → constrain → optimise → prune.
"""

from typing import List

import torch
from gsplat.rendering import rasterization
from loguru import logger
from PIL import Image as PILImage

from clsplats.change_detection.dinov2_detector import DinoV2Detector
from clsplats.config import CLSplatsConfig
from clsplats.constraints.primitives import fit_primitives_for_active, union_distance
from clsplats.dataset.cameras import Camera
from clsplats.dataset.dataset_reader import SceneInfo
from clsplats.lifter.depth_anything_lifter import DepthAnythingLifter
from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams
from clsplats.utils.custom_types import Image


class CLSplatsTrainer:
    """Minimal gsplat-backed trainer for continual-learning scene editing.

    Sets up Gaussians from a ``SceneInfo`` point cloud, renders with gsplat,
    and routes images through the DINOv2 change detector.
    """

    def __init__(self, cfg: CLSplatsConfig, scene: SceneInfo):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestep = 0
        self.active_mask = None
        self._primitives: list = []
        self._outside_counts = None
        self._global_step = 0

        # 1) Initialise Gaussians from scene point cloud
        pcd = scene.point_cloud
        xyz = torch.from_numpy(pcd.points).float().to(self.device)
        rgb = torch.from_numpy(pcd.colors).float().to(self.device)

        n_coeffs = (cfg.model.sh_degree + 1) ** 2
        N = xyz.shape[0]

        sh_features = torch.zeros((N, 3, n_coeffs), device=self.device)
        sh_features[:, :, 0] = rgb  # DC term encodes RGB

        scales = torch.full((N, 3), cfg.model.init_scale, device=self.device)
        quats = torch.zeros((N, 4), device=self.device)
        quats[:, 0] = 1.0
        opacity = torch.full((N, 1), cfg.model.init_opacity, device=self.device)

        params = GaussianParams(
            positions=xyz,
            scales=scales,
            quats=quats,
            sh_features=sh_features,
            opacity=opacity,
        )
        self.gaussians = CLGaussians(cfg, params)

        # 2) Cameras and change detector
        self.train_cameras: List[Camera] = []
        for uid, cam_info in enumerate(scene.train_cameras):
            img = PILImage.open(cam_info.image_path)
            resolution = (cam_info.width, cam_info.height)
            cam = Camera(
                resolution=resolution,
                colmap_id=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                depth_params=cam_info.depth_params,
                image=img,
                invdepthmap=None,
                image_name=cam_info.image_name,
                uid=uid,
                data_device="cpu",
                train_test_exp=False,
                is_test_dataset=False,
                is_test_view=cam_info.is_test,
            )
            self.train_cameras.append(cam)

        self.detector = DinoV2Detector(cfg.change)
        self.lifter = DepthAnythingLifter(cfg)

    def prepare_timestep(self, timestep: int) -> None:
        """Set up the trainer for optimising at the given *timestep*.

        At ``start_time`` all Gaussians are active (standard 3DGS).  From
        ``start_time + 1`` onwards, change detection and lifting select the
        subset to optimise.
        """
        assert timestep < self.cfg.train.num_times, "timestep >= num_times"
        self.timestep = timestep

        if len(self.train_cameras) == 0:
            self.active_mask = None
            return

        start_time = self.cfg.train.start_time
        if timestep == start_time:
            N = self.gaussians.params.positions.shape[0]
            self.active_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            self._primitives = []
            self._outside_counts = None
            return

        # Detect → lift
        change_masks = []
        for cam in self.train_cameras:
            rendered = self._render_camera(cam)
            gt = cam.original_image.permute(1, 2, 0).contiguous()
            with torch.no_grad():
                change_mask_2d = self.detector.predict_change_mask(
                    rendered_image=rendered,
                    observation=gt,
                )
            change_masks.append(change_mask_2d)

        self.active_mask = self.lifter.lift(
            gaussians=self.gaussians,
            cameras=self.train_cameras,
            change_masks=change_masks,
        )

        # Fit geometric primitives around active Gaussians
        if self.active_mask is not None and self.active_mask.any():
            self._primitives = [
                prim
                for _, prim in fit_primitives_for_active(
                    positions=self.gaussians.params.positions.detach(),
                    active_mask=self.active_mask.detach(),
                    radius_frac=self.cfg.constraints.group_radius_frac,
                )
            ]
        else:
            self._primitives = []

    def _render_camera(self, cam: Camera) -> Image:
        """Render a single camera view using gsplat rasterisation."""
        device = self.device
        means = self.gaussians.params.positions.to(device)
        quats = self.gaussians.params.quats.to(device)
        scales = self.gaussians.params.scales.to(device)
        opacities = self.gaussians.params.opacity.squeeze(-1).to(device)

        # SH features: [N, 3, K] → gsplat expects [N, K, 3]
        sh_feats = self.gaussians.params.sh_features.to(device)
        colors = sh_feats.permute(0, 2, 1).contiguous()

        viewmats = cam.world_view_transform.to(device).unsqueeze(0)
        Ks = torch.zeros(1, 3, 3, device=device, dtype=means.dtype)
        Ks[..., 0, 0] = cam.fx
        Ks[..., 1, 1] = cam.fy
        Ks[..., 0, 2] = cam.cx
        Ks[..., 1, 2] = cam.cy
        Ks[..., 2, 2] = 1.0

        width = cam.image_width
        height = cam.image_height

        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.cfg.model.sh_degree,
            packed=False,
            distributed=False,
        )

        img = render_colors[0]  # [H, W, 3]
        return img.contiguous()

    def _train_step(self, cam: Camera) -> dict:
        """Execute one optimisation step for a single camera view.

        Computes photometric loss, optionally adds a geometric constraint
        loss, masks gradients for inactive Gaussians, steps the optimiser,
        and optionally prunes outliers.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        rendered = self._render_camera(cam)
        gt = cam.original_image.permute(1, 2, 0).contiguous()

        photometric_loss = (rendered - gt).pow(2).mean()

        d_union_full = None
        start_time = self.cfg.train.start_time
        if self.timestep > start_time and self.active_mask is not None and self._primitives:
            pts = self.gaussians.params.positions
            d_union_full = union_distance(pts, self._primitives)
            mask = self.active_mask.to(self.device)
            if mask.any():
                loss_bound = d_union_full[mask].mean()
                total_loss = photometric_loss + self.cfg.constraints.lambda_bound * loss_bound
            else:
                total_loss = photometric_loss
        else:
            total_loss = photometric_loss
        total_loss.backward()

        # Zero gradients for inactive Gaussians (from t1+)
        if self.timestep > start_time and self.active_mask is not None:
            mask = self.active_mask.to(self.device)

            def apply_mask(param, extra_dims: int = 0):
                if param.grad is None:
                    return
                view_shape = (mask.shape[0],) + (1,) * extra_dims
                param.grad *= mask.view(*view_shape)

            apply_mask(self.gaussians.params.positions, extra_dims=1)
            apply_mask(self.gaussians.params.scales, extra_dims=1)
            apply_mask(self.gaussians.params.quats, extra_dims=1)
            apply_mask(self.gaussians.params.sh_features, extra_dims=2)
            apply_mask(self.gaussians.params.opacity, extra_dims=1)

        self.gaussians.step_optimizer()

        # Hard pruning with hysteresis
        if (
            self.timestep > start_time
            and d_union_full is not None
            and self.active_mask is not None
            and self._primitives
        ):
            prune_every = self.cfg.constraints.prune_every
            prune_dist = self.cfg.constraints.prune_dist_thresh
            prune_consec = self.cfg.constraints.prune_consecutive

            if self._global_step % prune_every == 0:
                N = self.gaussians.params.positions.shape[0]
                if self._outside_counts is None or self._outside_counts.shape[0] != N:
                    self._outside_counts = torch.zeros(N, dtype=torch.int64, device=self.device)
                outside = d_union_full > prune_dist
                self._outside_counts[outside] += 1
                self._outside_counts[~outside] = 0
                prune_mask = self._outside_counts >= prune_consec
                if prune_mask.any():
                    keep = self.gaussians.prune_gaussians(prune_mask)
                    self.active_mask = self.active_mask[keep]
                    self._outside_counts = self._outside_counts[keep]
                    # Re-fit primitives on remaining active Gaussians
                    if self.active_mask.any():
                        self._primitives = [
                            prim
                            for _, prim in fit_primitives_for_active(
                                positions=self.gaussians.params.positions.detach(),
                                active_mask=self.active_mask.detach(),
                                radius_frac=self.cfg.constraints.group_radius_frac,
                            )
                        ]
                    else:
                        self._primitives = []

        self._global_step += 1
        return {"loss": float(total_loss.detach().cpu())}

    def train(self) -> None:
        """Run the training loop for the current timestep."""
        num_iters = self.cfg.train.iters_per_timestep
        log_interval = self.cfg.train.log_interval

        for it in range(num_iters):
            cam = self.train_cameras[it % len(self.train_cameras)]
            stats = self._train_step(cam)

            if (it + 1) % log_interval == 0:
                logger.info(
                    "[time={time} it={it}/{num_iters}] loss={loss:.4f}",
                    time=self.timestep,
                    it=it + 1,
                    num_iters=num_iters,
                    loss=stats["loss"],
                )

    def log_history(self) -> None:
        """Export Gaussians and log metrics."""
        from pathlib import Path

        # We export the point cloud to the current working directory, which
        # will typically be managed by Hydra's output directory system, or ./outputs
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save a .ply file for the current timestep
        ply_path = out_dir / f"gaussians_time_{self.timestep:04d}.ply"
        self.gaussians.export_ply(str(ply_path))
        logger.info("Exported optimized Gaussians to {path}", path=ply_path)
