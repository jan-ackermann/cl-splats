"""Abstract base class for lifters.

A lifter receives Gaussian parameters, camera views, and 2-D change masks
and produces a boolean per-Gaussian mask indicating which Gaussians are
located in changed regions.
"""

import abc
from typing import TYPE_CHECKING, List

import torch

from clsplats.config import CLSplatsConfig

if TYPE_CHECKING:
    from clsplats.dataset.cameras import Camera
    from clsplats.representation.cl_gaussians import CLGaussians


class BaseLifter(abc.ABC):
    def __init__(self, cfg: CLSplatsConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def lift(
        self,
        gaussians: "CLGaussians",
        cameras: List["Camera"],
        change_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Lift 2-D change masks into a per-Gaussian boolean mask.

        Args:
            gaussians: The current set of Gaussians.
            cameras: Camera objects for each view.
            change_masks: Per-view 2-D change masks ``[H, W]``.

        Returns:
            Boolean mask of shape ``(N,)`` over Gaussians.
        """
        raise NotImplementedError
