import abc

import omegaconf
import torch

from clsplats.utils.custom_types import Image


class BaseLifter(abc.ABC):

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def lift(self, rendered_image: Image, observation: Image) -> torch.Tensor:
        """
        Given a rendered image and an observation image, return a boolean
        mask over gaussians indicating which ones should be updated.
        Concrete implementations may internally estimate depth and use
        change masks to perform 3D lifting.
        """
        raise NotImplementedError