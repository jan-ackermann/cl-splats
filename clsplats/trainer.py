from typing import List

import omegaconf
import torch
from PIL import Image as PILImage

from gsplat.rendering import rasterization

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
        # Wrap CameraInfo objects into our own Camera instances on CPU.
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
        Render using gsplat's 3DGS rasterization for a single camera.
        """
        device = self.device
        means = self.gaussians.params.positions.to(device)          # [N, 3]
        quats = self.gaussians.params.quats.to(device)              # [N, 4]
        scales = self.gaussians.params.scales.to(device)            # [N, 3]
        opacities = self.gaussians.params.opacity.squeeze(-1).to(device)  # [N]

        # SH features are stored as [N, 3, K]; gsplat expects [N, K, 3]
        sh_feats = self.gaussians.params.sh_features.to(device)     # [N, 3, K]
        colors = sh_feats.permute(0, 2, 1).contiguous()             # [N, K, 3]

        # Single-camera batch: viewmats [1, 4, 4], Ks [1, 3, 3]
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
            viewmats=viewmats,   # [1,4,4]
            Ks=Ks,               # [1,3,3]
            width=width,
            height=height,
             sh_degree=getattr(self.cfg.model, "sh_degree", 0),
            packed=False,
            distributed=False,
        )

        # render_colors: [1, H, W, 3] for RGB mode
        img = render_colors[0]  # [H, W, 3]
        return img.contiguous()

    def _train_step(self, cam: Camera):
        rendered = self._render_camera(cam)
        gt = cam.original_image.permute(1, 2, 0).contiguous()  # [H, W, 3]

        photometric_loss = (rendered - gt).pow(2).mean()
        total_loss = photometric_loss
        total_loss.backward()

        # Restrict optimization to gaussians selected by active_mask:
        # we keep all gaussians in the forward pass, but only this subset
        # receives gradient updates.
        if self.active_mask is not None:
            mask = self.active_mask.to(self.device)
            # Broadcast mask to parameter shapes and zero out grads for inactive gaussians
            def apply_mask(param, extra_dims: int = 0):
                if param.grad is None:
                    return
                view_shape = (mask.shape[0],) + (1,) * extra_dims
                param.grad *= mask.view(*view_shape)

            apply_mask(self.gaussians.params.positions, extra_dims=1)      # (N, 3)
            apply_mask(self.gaussians.params.scales, extra_dims=1)         # (N, 3)
            apply_mask(self.gaussians.params.quats, extra_dims=1)          # (N, 4)
            apply_mask(self.gaussians.params.sh_features, extra_dims=2)    # (N, C, K)
            apply_mask(self.gaussians.params.opacity, extra_dims=1)        # (N, 1)

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