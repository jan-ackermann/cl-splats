"""CL-specific Gaussian parameter wrapper.

Provides a lightweight container for Gaussian Splatting parameters and
an Adam optimizer with correct bookkeeping for pruning and densification.
"""

import dataclasses
import logging
from typing import Dict, List

import gsplat
import omegaconf
import torch

from clsplats.config import CLSplatsConfig

log = logging.getLogger(__name__)


@dataclasses.dataclass
class GaussianParams:
    """Container for the core per-Gaussian tensors."""

    positions: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3) or (N, 1)
    quats: torch.Tensor  # (N, 4)
    sh_features: torch.Tensor  # (N, C, K)
    opacity: torch.Tensor  # (N, 1)


class CLGaussians:
    """CL-specific wrapper around gsplat Gaussian parameters and strategy.

    This intentionally does not subclass the legacy ``GaussianModel``; instead
    it uses gsplat as the underlying backend.
    """

    def __init__(self, cfg: CLSplatsConfig, params: GaussianParams):
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

        self.optimizer = self._build_optimizer()
        self._extra_state: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _param_groups(self) -> List[dict]:
        """Return the optimizer parameter groups for the current params."""
        lr = self.cfg.train.lr if hasattr(self.cfg, "train") else 1e-3
        return [
            {"params": [self.params.positions], "name": "xyz", "lr": lr},
            {"params": [self.params.scales], "name": "scales", "lr": lr},
            {"params": [self.params.quats], "name": "quats", "lr": lr},
            {"params": [self.params.sh_features], "name": "sh", "lr": lr},
            {"params": [self.params.opacity], "name": "opacity", "lr": lr},
        ]

    def _build_optimizer(self) -> torch.optim.Adam:
        """Create a fresh Adam optimizer for the current parameters."""
        return torch.optim.Adam(self._param_groups())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step_optimizer(self) -> None:
        """Perform one optimiser step and zero gradients."""
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def prune_gaussians(self, prune_mask: torch.Tensor) -> torch.Tensor:
        """Remove Gaussians where ``prune_mask[i]`` is True.

        This also **rebuilds the optimizer** so that Adam's internal state
        (momentum / variance buffers) stays consistent with the new parameter
        shapes.

        Returns:
            ``keep`` — boolean mask indicating which Gaussians remain.
        """
        prune_mask = prune_mask.to(self.device)
        keep = ~prune_mask

        def _filter(t: torch.Tensor) -> torch.Tensor:
            return t[keep].detach().requires_grad_(True)

        self.params.positions = _filter(self.params.positions)
        self.params.scales = _filter(self.params.scales)
        self.params.quats = _filter(self.params.quats)
        self.params.sh_features = _filter(self.params.sh_features)
        self.params.opacity = _filter(self.params.opacity)

        # Bug 5 fix: rebuild optimizer so state matches pruned params
        self.optimizer = self._build_optimizer()
        log.debug(
            "Pruned %d Gaussians (%d remain).",
            prune_mask.sum().item(),
            keep.sum().item(),
        )
        return keep

    def split_gaussians(self, active_mask: torch.Tensor) -> None:
        """Placeholder for densification/splitting logic guided by *active_mask*.

        Will be wired to gsplat strategies once thresholds and policies
        are finalised.
        """
        _ = active_mask

    def unify_gaussians(self) -> None:
        """Placeholder for bookkeeping after local edits.

        Currently a no-op.
        """
        return

    @torch.no_grad()
    def export_ply(self, path: str) -> None:
        """Export current Gaussians to a ``.ply`` file via gsplat."""
        import gsplat.exporter as exporter

        # sh_features is (N, K, 3). sh0 is (N, 1, 3), shN is (N, K-1, 3).
        exporter.export_splats(
            means=self.params.positions,
            scales=self.params.scales,
            quats=self.params.quats,
            opacities=self.params.opacity,
            sh0=self.params.sh_features[:, 0:1, :],
            shN=self.params.sh_features[:, 1:, :],
            save_to=path,
        )
