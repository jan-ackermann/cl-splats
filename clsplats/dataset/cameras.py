# Adapted from https://github.com/graphdeco-inria/gaussian-splat-pytorch
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Camera representations for the CL-Splats pipeline."""

import math

import cv2
import numpy as np
import torch
from loguru import logger
from torch import nn

from clsplats.utils.general_utils import PILtoTorch
from clsplats.utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    """Pinhole camera with pre-computed view and projection matrices.

    Attributes:
        Twc: Cached camera-to-world transform (inverse of ``world_view_transform``).
        fx, fy: Focal lengths derived from the field-of-view and image resolution.
        cx, cy: Principal point at pixel centre.
    """

    def __init__(
        self,
        resolution,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        depth_params,
        image,
        invdepthmap,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
        timestep=0,
    ):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestep = timestep

        try:
            self.data_device = torch.device(data_device)
        except RuntimeError as e:
            logger.warning(
                "Custom device {dev} failed ({e}), falling back to cuda.", dev=data_device, e=e
            )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., : self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2 :] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if (
                    depth_params["scale"] < 0.2 * depth_params["med_scale"]
                    or depth_params["scale"] > 5 * depth_params["med_scale"]
                ):
                    self.depth_reliable = False
                    self.depth_mask *= 0

                if depth_params["scale"] > 0:
                    self.invdepthmap = (
                        self.invdepthmap * depth_params["scale"] + depth_params["offset"]
                    )

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        w2v = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        proj = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1)
        self.world_view_transform = w2v.to(self.data_device)
        self.projection_matrix = proj.to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Bug 7 fix: cache Twc once instead of recomputing on every access
        self._Twc = torch.inverse(self.world_view_transform)

    @property
    def Twc(self) -> torch.Tensor:
        """Camera-to-world transform (cached inverse of ``world_view_transform``)."""
        return self._Twc

    @property
    def fx(self) -> float:
        """Focal length in x derived from FoVx and image width."""
        return 0.5 * self.image_width / math.tan(self.FoVx * 0.5)

    @property
    def fy(self) -> float:
        """Focal length in y derived from FoVy and image height."""
        return 0.5 * self.image_height / math.tan(self.FoVy * 0.5)

    @property
    def cx(self) -> float:
        """Principal point x (pixel centre)."""
        return (self.image_width - 1) * 0.5

    @property
    def cy(self) -> float:
        """Principal point y (pixel centre)."""
        return (self.image_height - 1) * 0.5


class MiniCam:
    """Lightweight camera representation for evaluation / visualisation."""

    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
