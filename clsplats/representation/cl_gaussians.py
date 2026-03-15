import dataclasses
from typing import Dict

import omegaconf
import torch
import gsplat


@dataclasses.dataclass
class GaussianParams:
    positions: torch.Tensor      # (N, 3)
    scales: torch.Tensor         # (N, 3) or (N, 1)
    quats: torch.Tensor          # (N, 4)
    sh_features: torch.Tensor    # (N, C, K)
    opacity: torch.Tensor        # (N, 1)


class CLGaussians:
    """
    CL-specific wrapper around gsplat Gaussian parameters and strategy.
    This intentionally does not subclass the legacy GaussianModel; instead
    it uses gsplat as the underlying backend.
    """

    def __init__(self, cfg: omegaconf.DictConfig, params: GaussianParams):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core parameters (all on same device)
        self.params = GaussianParams(
            positions=params.positions.to(self.device),
            scales=params.scales.to(self.device),
            quats=params.quats.to(self.device),
            sh_features=params.sh_features.to(self.device),
            opacity=params.opacity.to(self.device),
        )

        # Register as optimizable tensors
        self.params.positions.requires_grad_(True)
        self.params.scales.requires_grad_(True)
        self.params.quats.requires_grad_(True)
        self.params.sh_features.requires_grad_(True)
        self.params.opacity.requires_grad_(True)

        # Strategy & optimizer (hyperparameters can later be taken from cfg)
        self.strategy: gsplat.Strategy = gsplat.DefaultStrategy()

        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.params.positions], "name": "xyz"},
                {"params": [self.params.scales], "name": "scales"},
                {"params": [self.params.quats], "name": "quats"},
                {"params": [self.params.sh_features], "name": "sh"},
                {"params": [self.params.opacity], "name": "opacity"},
            ],
            lr=getattr(cfg.train, "lr", 1e-3),
        )

        self._extra_state: Dict[str, torch.Tensor] = {}

    def step_optimizer(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def prune_gaussians(self, prune_mask: torch.Tensor) -> None:
        """
        Remove gaussians where prune_mask[i] is True.
        """
        prune_mask = prune_mask.to(self.device)
        keep = ~prune_mask

        def _filter(t: torch.Tensor) -> torch.Tensor:
            return t[keep]

        self.params.positions = _filter(self.params.positions)
        self.params.scales = _filter(self.params.scales)
        self.params.quats = _filter(self.params.quats)
        self.params.sh_features = _filter(self.params.sh_features)
        self.params.opacity = _filter(self.params.opacity)

    def split_gaussians(self, active_mask: torch.Tensor) -> None:
        """
        Placeholder for densification/splitting logic guided by active_mask.
        This will be wired to gsplat strategies once thresholds and policies
        are finalized.
        """
        # For now this is a no-op to keep the API stable.
        _ = active_mask

    def unify_gaussians(self) -> None:
        """
        Placeholder for any bookkeeping after local edits (e.g. merging
        static/dynamic sets). Currently a no-op.
        """
        return

    @torch.no_grad()
    def export_ply(self, path: str) -> None:
        """
        Optional helper to export current gaussians via gsplat's exporter.
        """
        import gsplat.exporter as exporter

        exporter.export_splats(
            path,
            positions=self.params.positions,
            scales=self.params.scales,
            quats=self.params.quats,
            sh_features=self.params.sh_features,
            opacity=self.params.opacity,
        )