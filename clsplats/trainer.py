from typing import List

import omegaconf
import torch

import gsplat.rendering as rendering

from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams
from clsplats.change_detection.dinov2_detector import DinoV2Detector
from clsplats.utils.custom_types import Image
from clsplats.dataset.cameras import Camera
from clsplats.dataset.dataset_reader import SceneInfo
from clsplats.lifter.depth_anything_lifter import DepthAnythingLifter


class CLSplatsTrainer:
    """
    Minimal gsplat-backed trainer skeleton.
    This sets up gaussians from a SceneInfo point cloud, renders with gsplat,
    and routes images through the DINOv2 change detector.
    """

    def __init__(self, cfg: omegaconf.DictConfig, scene: SceneInfo):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestep = 0
        self.active_mask = None  # mask of gaussians affected by current scene change

        # 1) Initialize gaussians from scene point cloud
        pcd = scene.point_cloud
        xyz = torch.from_numpy(pcd.points).float().to(self.device)  # (N, 3)
        rgb = torch.from_numpy(pcd.colors).float().to(self.device)  # (N, 3) in [0,1]

        sh_degree = getattr(cfg.model, "sh_degree", 0)
        n_coeffs = (sh_degree + 1) ** 2
        N = xyz.shape[0]

        sh_features = torch.zeros((N, 3, n_coeffs), device=self.device)
        sh_features[:, :, 0] = rgb  # DC term encodes RGB

        init_scale = getattr(cfg.model, "init_scale", 0.01)
        init_opacity = getattr(cfg.model, "init_opacity", 0.5)
        scales = torch.full((N, 3), init_scale, device=self.device)
        quats = torch.zeros((N, 4), device=self.device)
        quats[:, 0] = 1.0
        opacity = torch.full((N, 1), init_opacity, device=self.device)

        params = GaussianParams(
            positions=xyz,
            scales=scales,
            quats=quats,
            sh_features=sh_features,
            opacity=opacity,
        )
        self.gaussians = CLGaussians(cfg, params)

        # 2) Cameras and change detector
        self.train_cameras: List[Camera] = scene.train_cameras
        self.detector = DinoV2Detector(cfg.change)
        self.lifter = DepthAnythingLifter(cfg)

    def prepare_timestep(self, timestep: int):
        assert timestep < self.cfg.train.num_times, "Timestep must be less than num_times"
        self.timestep = timestep
        if len(self.train_cameras) == 0:
            self.active_mask = None
            return
        # At a scene change, detect which region changed once per view and
        # derive the subset of gaussians to optimize using the lifter.
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

    def _render_camera(self, cam: Camera) -> Image:
        """
        Render using gsplat for a single camera.
        Assumes Camera exposes world_view_transform and projection_matrix
        as (4, 4) CUDA tensors.
        """
        world_view = cam.world_view_transform  # (4, 4)
        proj = cam.projection_matrix           # (4, 4)

        rendered = rendering.render(
            positions=self.gaussians.params.positions,
            scales=self.gaussians.params.scales,
            quats=self.gaussians.params.quats,
            sh_features=self.gaussians.params.sh_features,
            opacity=self.gaussians.params.opacity,
            world_to_cam=world_view,
            projection=proj,
            image_height=cam.image_height,
            image_width=cam.image_width,
        )
        rendered = rendered.permute(1, 2, 0).contiguous()  # [H, W, 3]
        return rendered

    def _train_step(self, cam: Camera):
        rendered = self._render_camera(cam)
        gt = cam.original_image.permute(1, 2, 0).contiguous()  # [H, W, 3]

        photometric_loss = (rendered - gt).pow(2).mean()
        total_loss = photometric_loss
        total_loss.backward()

        # TODO: restrict optimization to gaussians selected by self.active_mask by
        # masking gradients or using separate parameter groups.
        self.gaussians.step_optimizer()

        return {"loss": float(total_loss.detach().cpu())}

    def train(self):
        num_iters = getattr(self.cfg.train, "iters_per_timestep", 100)
        log_interval = getattr(self.cfg.train, "log_interval", 10)

        for it in range(num_iters):
            cam = self.train_cameras[it % len(self.train_cameras)]
            stats = self._train_step(cam)

            if (it + 1) % log_interval == 0:
                print(
                    f"[time={self.timestep} it={it+1}/{num_iters}] "
                    f"loss={stats['loss']:.4f}"
                )

    def log_history(self):
        # Placeholder for exporting gaussians / logging to wandb.
        return