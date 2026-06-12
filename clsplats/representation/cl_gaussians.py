"""CL-specific Gaussian parameter wrapper.

Provides a lightweight container for Gaussian Splatting parameters and
an Adam optimizer with correct bookkeeping for pruning and densification.
"""

import dataclasses
import logging
import math
from typing import Dict, Optional, Tuple

import gsplat
import torch
import torch.nn.functional as F
from torch import nn

from clsplats.config import CLSplatsConfig

log = logging.getLogger(__name__)

# Sentinel for "opacity reset disabled": gsplat fires the reset whenever
# ``step % reset_every == 0 and step > 0``, so any value above the longest
# conceivable timestep never triggers.
_NEVER = 2**31 - 1

# Keys that carry CL bookkeeping (not real Gaussian parameters). They live in
# ``strategy_params`` so gsplat's duplicate/split/remove ops keep them
# index-aligned with the Gaussians; they never receive gradients.
_FLAG_KEYS = ("cl_active", "cl_outside_count")


@dataclasses.dataclass
class GaussianParams:
    """Container for the core per-Gaussian tensors."""

    positions: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3) or (N, 1)
    quats: torch.Tensor  # (N, 4)
    sh_features: torch.Tensor  # (N, C, K)
    opacity: torch.Tensor  # (N, 1)


@dataclasses.dataclass
class DensificationResult:
    """Bookkeeping needed by the trainer after densification."""

    keep_mask: torch.Tensor
    new_parent_indices: torch.Tensor


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(x / (1.0 - x))


class GaussianParamView:
    """Compatibility view exposing activated tensors with the old attribute names."""

    def __init__(self, owner: "CLGaussians"):
        self._owner = owner

    @property
    def positions(self) -> torch.Tensor:
        return self._owner.strategy_params["means"]

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._owner.strategy_params["scales"])

    @property
    def quats(self) -> torch.Tensor:
        return F.normalize(self._owner.strategy_params["quats"], dim=-1)

    @property
    def sh_features(self) -> torch.Tensor:
        # sh0/shN are stored gsplat-style as (N, K, 3); expose (N, 3, K).
        sh = torch.cat(
            [self._owner.strategy_params["sh0"], self._owner.strategy_params["shN"]], dim=1
        )
        return sh.permute(0, 2, 1).contiguous()

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._owner.strategy_params["opacities"]).unsqueeze(-1)


class CLGaussians:
    """CL-specific wrapper around gsplat Gaussian parameters and strategy.

    This intentionally does not subclass the legacy ``GaussianModel``; instead
    it uses gsplat as the underlying backend.
    """

    def __init__(
        self,
        cfg: CLSplatsConfig,
        params: GaussianParams,
        spatial_lr_scale: float = 1.0,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_sh_degree = 0
        self.max_sh_degree = cfg.model.sh_degree
        # 3DGS scales the position learning rate by the camera extent
        # (``spatial_lr_scale`` in the reference implementation).
        self.spatial_lr_scale = spatial_lr_scale

        self.strategy_params = self._make_param_dict(params)
        self.params = GaussianParamView(self)

        self.optimizers = self._build_optimizers()
        self.optimizer = self.optimizers["means"]
        self._extra_state: Dict[str, torch.Tensor] = {}
        self.strategy = self._build_strategy()
        self.strategy.check_sanity(self.strategy_params, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_param_dict(
        self,
        params: GaussianParams,
        cl_active: Optional[torch.Tensor] = None,
        cl_outside_count: Optional[torch.Tensor] = None,
    ) -> nn.ParameterDict:
        n = params.positions.shape[0]
        # sh_features is (N, 3, K) → gsplat layout (N, K, 3), split into the
        # DC term and the higher-order coefficients (separate learning rates).
        sh = params.sh_features.detach().to(self.device).permute(0, 2, 1).contiguous()
        if cl_active is None:
            cl_active = torch.ones(n, device=self.device)
        if cl_outside_count is None:
            cl_outside_count = torch.zeros(n, device=self.device)
        return nn.ParameterDict(
            {
                "means": nn.Parameter(params.positions.detach().to(self.device)),
                "scales": nn.Parameter(
                    torch.log(params.scales.detach().to(self.device).clamp_min(1e-8))
                ),
                "quats": nn.Parameter(params.quats.detach().to(self.device)),
                "opacities": nn.Parameter(
                    _inverse_sigmoid(params.opacity.detach().to(self.device).squeeze(-1))
                ),
                "sh0": nn.Parameter(sh[:, :1, :]),
                "shN": nn.Parameter(sh[:, 1:, :]),
                "cl_active": nn.Parameter(cl_active.detach().float().to(self.device)),
                "cl_outside_count": nn.Parameter(
                    cl_outside_count.detach().float().to(self.device)
                ),
            }
        )

    def _learning_rates(self) -> Dict[str, float]:
        train_cfg = self.cfg.train
        lr = train_cfg.lr if hasattr(train_cfg, "lr") else 1e-3
        return {
            "means": getattr(train_cfg, "position_lr_init", lr) * self.spatial_lr_scale,
            "scales": getattr(train_cfg, "scaling_lr", lr),
            "quats": getattr(train_cfg, "rotation_lr", lr),
            "sh0": getattr(train_cfg, "feature_lr", lr),
            # 3DGS trains the higher-order SH coefficients at feature_lr / 20.
            "shN": getattr(train_cfg, "feature_lr", lr) / 20.0,
            "opacities": getattr(train_cfg, "opacity_lr", lr),
            "cl_active": 0.0,
            "cl_outside_count": 0.0,
        }

    def _build_optimizers(self) -> Dict[str, torch.optim.Adam]:
        """Create one Adam optimizer per tensor, as required by gsplat strategies."""
        lrs = self._learning_rates()
        param_group_names = {
            "means": "xyz",
            "scales": "scaling",
            "quats": "rotation",
            "sh0": "f_dc",
            "shN": "f_rest",
            "opacities": "opacity",
            "cl_active": "cl_active",
            "cl_outside_count": "cl_outside_count",
        }
        return {
            name: torch.optim.Adam(
                [{"params": [param], "lr": lrs[name], "name": param_group_names[name]}],
                lr=0.0,
                eps=1e-15,
            )
            for name, param in self.strategy_params.items()
        }

    def _build_strategy(self, cl_phase: bool = False):
        """Build the gsplat strategy; the CL phase must not touch inactive Gaussians.

        During continual learning the opacity reset is disabled (inactive
        Gaussians could never recover through masked gradients) and the
        strategy's *global* opacity pruning is disabled too — it would
        delete frozen scene rows and break exact history recovery. The
        trainer prunes low-opacity *active* Gaussians instead.
        """
        train_cfg = self.cfg.train
        return gsplat.DefaultStrategy(
            prune_opa=0.0 if cl_phase else train_cfg.densify_prune_opa,
            grow_grad2d=train_cfg.densify_grad_threshold,
            grow_scale3d=train_cfg.percent_dense,
            prune_scale3d=train_cfg.densify_prune_scale3d,
            refine_start_iter=train_cfg.densify_from_iter,
            refine_stop_iter=train_cfg.densify_until_iter,
            refine_every=train_cfg.densification_interval,
            reset_every=_NEVER if cl_phase else train_cfg.opacity_reset_interval,
            absgrad=train_cfg.densify_absgrad,
            verbose=train_cfg.densify_verbose,
            key_for_gradient="means2d",
        )

    def initialize_strategy_state(self, scene_scale: float = 1.0) -> None:
        """Reset gsplat strategy state using the scene scale."""
        self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    def setup_timestep(self, scene_scale: float, cl_phase: bool = False) -> None:
        """(Re)build the strategy for a new timestep with a fresh step counter.

        The trainer drives the strategy with a per-timestep iteration so the
        densification window applies at every timestep. With ``cl_phase``
        the strategy is configured to never modify inactive Gaussians (no
        opacity reset, no global opacity pruning) — see ``_build_strategy``.
        """
        self.strategy = self._build_strategy(cl_phase=cl_phase)
        self.strategy.check_sanity(self.strategy_params, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    @property
    def num_gaussians(self) -> int:
        """Current number of Gaussian primitives."""
        return self.strategy_params["means"].shape[0]

    def update_learning_rate(self, step: int) -> float:
        """Update the position learning rate with the 3DGS exponential schedule.

        Mirrors the reference ``get_expon_lr_func``: log-linear interpolation
        from ``position_lr_init`` to ``position_lr_final``, both scaled by
        ``spatial_lr_scale``. The reference constructs the schedule with
        ``lr_delay_steps=0``, so the sine delay ramp never applies — we
        deliberately omit it here (``position_lr_delay_mult`` is inert, as in
        3DGS).
        """
        train_cfg = self.cfg.train
        base_lr = getattr(train_cfg, "lr", 1e-3)
        lr_init = getattr(train_cfg, "position_lr_init", base_lr) * self.spatial_lr_scale
        lr_final = getattr(train_cfg, "position_lr_final", lr_init) * self.spatial_lr_scale
        max_steps = max(getattr(train_cfg, "position_lr_max_steps", 0), 1)

        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            lr = 0.0
        else:
            t = min(max(step / max_steps, 0.0), 1.0)
            lr = math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)

        self.optimizers["means"].param_groups[0]["lr"] = lr
        return lr

    def update_sh_degree(self, iteration: int) -> int:
        """Match 3DGS progressive SH activation: one degree every 1000 iterations."""
        self.active_sh_degree = min(self.max_sh_degree, max(iteration, 0) // 1000)
        return self.active_sh_degree

    def oneupSHdegree(self) -> None:
        """Compatibility with the original 3DGS method name."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step_optimizer(self) -> None:
        """Perform one optimiser step and zero gradients."""
        for name, optimizer in self.optimizers.items():
            if name in _FLAG_KEYS:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def step_pre_backward(self, step: int, info: Dict) -> None:
        """Run gsplat strategy's pre-backward hook."""
        self.strategy.step_pre_backward(
            self.strategy_params,
            self.optimizers,
            self.strategy_state,
            step,
            info,
        )

    def step_post_backward(
        self,
        step: int,
        info: Dict,
        active_mask: Optional[torch.Tensor] = None,
        outside_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the gsplat strategy post-backward hook.

        The CL bookkeeping (``active_mask``, ``outside_counts``) is written
        into strategy-managed flag tensors before the hook runs, so gsplat's
        duplicate/split/remove ops keep it index-aligned with the Gaussians
        (split removes parents mid-array and appends children — a plain
        append/truncate of the mask would scramble it). The realigned values
        are returned.
        """
        device = self.strategy_params["means"].device
        n = self.num_gaussians
        if active_mask is not None:
            self.strategy_params["cl_active"].data.copy_(active_mask.to(device).float())
        else:
            self.strategy_params["cl_active"].data.copy_(torch.ones(n, device=device))
        if outside_counts is not None:
            self.strategy_params["cl_outside_count"].data.copy_(
                outside_counts.to(device).float()
            )
        else:
            self.strategy_params["cl_outside_count"].data.copy_(torch.zeros(n, device=device))

        if active_mask is not None and "means2d" in info:
            means2d = info["means2d"]
            if means2d.grad is not None:
                # means2d grad is [C, N, 2] (packed=False) — align the [N]
                # mask with the Gaussian dimension, not the trailing one.
                grad = means2d.grad
                mask = active_mask.to(means2d.device)
                shape = [1] * grad.ndim
                shape[-2] = mask.shape[0]
                grad *= mask.view(*shape).to(grad.dtype)

        self.strategy.step_post_backward(
            self.strategy_params,
            self.optimizers,
            self.strategy_state,
            step,
            info,
            packed=False,
        )
        self.params = GaussianParamView(self)

        new_active = (
            self.strategy_params["cl_active"].detach() > 0.5 if active_mask is not None else None
        )
        new_counts = (
            self.strategy_params["cl_outside_count"].detach().round().to(torch.int64)
            if outside_counts is not None
            else None
        )
        return new_active, new_counts

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

        # Subset the raw tensors directly. Round-tripping through the
        # activated view (exp/log, sigmoid/logit, quat normalisation) would
        # perturb every surviving row numerically and break exact history
        # recovery of the frozen scene.
        self.strategy_params = nn.ParameterDict(
            {
                name: nn.Parameter(param.detach()[keep].clone())
                for name, param in self.strategy_params.items()
            }
        )
        self.params = GaussianParamView(self)
        self.optimizers = self._build_optimizers()
        self.optimizer = self.optimizers["means"]
        self.strategy.check_sanity(self.strategy_params, self.optimizers)
        self.initialize_strategy_state(self.strategy_state.get("scene_scale", 1.0))
        log.debug(
            "Pruned %d Gaussians (%d remain).",
            prune_mask.sum().item(),
            keep.sum().item(),
        )
        return keep

    def mask_inactive_gradients(self, active_mask: Optional[torch.Tensor]) -> None:
        """Zero parameter gradients for inactive Gaussians."""
        if active_mask is not None:
            mask = active_mask.to(self.device)
            for name, param in self.strategy_params.items():
                if param.grad is None:
                    continue
                view_shape = (mask.shape[0],) + (1,) * (param.grad.ndim - 1)
                param.grad *= mask.view(*view_shape)

    def reset_inactive_optimizer_state(self, active_mask: Optional[torch.Tensor]) -> None:
        """Zero Adam moments for inactive Gaussians.

        Masking gradients alone does not freeze parameters: Adam keeps moving
        them with momentum accumulated in earlier timesteps. Combined with
        per-step gradient masking, zeroed moments keep inactive Gaussians
        exactly fixed.
        """
        if active_mask is None:
            return
        inactive = ~active_mask.to(self.device)
        if not bool(inactive.any()):
            return
        for name, optimizer in self.optimizers.items():
            param = self.strategy_params[name]
            state = optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                buf = state.get(key)
                if buf is not None and buf.shape[0] == inactive.shape[0]:
                    buf[inactive] = 0.0

    def split_gaussians(self, active_mask: torch.Tensor) -> Optional[DensificationResult]:
        """Compatibility no-op; densification is handled by gsplat.DefaultStrategy."""
        _ = active_mask
        return None

    def unify_gaussians(self) -> None:
        """Placeholder for bookkeeping after local edits.

        Currently a no-op.
        """
        return

    @torch.no_grad()
    def export_ply(self, path: str) -> None:
        """Export current Gaussians to a ``.ply`` file via gsplat."""
        import gsplat.exporter as exporter

        # export_splats writes values verbatim in the standard 3DGS PLY
        # layout: log-scales, logit-opacities, raw quats, SH as (N, K, 3).
        exporter.export_splats(
            means=self.strategy_params["means"],
            scales=self.strategy_params["scales"],
            quats=self.strategy_params["quats"],
            opacities=self.strategy_params["opacities"],
            sh0=self.strategy_params["sh0"],
            shN=self.strategy_params["shN"],
            save_to=path,
        )
