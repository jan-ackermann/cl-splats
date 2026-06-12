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

import numpy as np
import torch
from loguru import logger
from torch import nn

from clsplats.utils.general_utils import PILtoTorch
from clsplats.utils.graphics_utils import getWorld2View2


class Camera(nn.Module):
    """Pinhole camera with a pre-computed view matrix.

    Attributes:
        world_view_transform: World-to-camera matrix in the transposed
            (column-major) 3DGS convention.
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
        image,
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

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        w2v = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.world_view_transform = w2v.to(self.data_device)

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
