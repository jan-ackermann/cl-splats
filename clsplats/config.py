"""Structured configuration for CL-Splats.

Provides typed dataclasses with sensible defaults for every component.
These replace the scattered ``getattr(cfg.*, ..., default)`` pattern
throughout the codebase with a single source of truth.

Register with Hydra via the ``register_configs()`` call in ``train.py``.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Gaussian model hyper-parameters."""

    sh_degree: int = 0
    init_scale: float = 0.01
    init_opacity: float = 0.5


@dataclass
class TrainConfig:
    """Training loop settings."""

    lr: float = 1e-3
    iters_per_timestep: int = 100
    log_interval: int = 10
    num_times: int = 1
    start_time: int = 0


@dataclass
class ChangeDetectionConfig:
    """DINOv2 change-detection settings."""

    threshold: float = 0.8
    dilate_mask: bool = False
    dilate_kernel_size: int = 31
    upsample: bool = True


@dataclass
class LifterConfig:
    """Depth-Anything lifter settings."""

    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    k_nn: int = 8
    local_radius_thresh: float = 2.5
    depth_tol_abs: float = 0.05
    depth_tol_rel: float = 0.05
    lambda_seed: float = 2.0
    lambda_neg: float = 0.25
    min_visible_views: int = 2
    min_positive_views: int = 2
    min_seed_views: int = 1
    min_positive_ratio: float = 0.3
    final_thresh: float = 0.6


@dataclass
class ConstraintsConfig:
    """Geometric constraint and pruning settings."""

    group_radius_frac: float = 0.1
    lambda_bound: float = 0.0
    prune_every: int = 50
    prune_dist_thresh: float = 0.02
    prune_consecutive: int = 3


@dataclass
class HistoryConfig:
    """Logging and history export settings."""

    log_history: bool = False


@dataclass
class CLSplatsConfig:
    """Top-level configuration for the CL-Splats pipeline."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    change: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    lifter: LifterConfig = field(default_factory=LifterConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)

    # Dataset arguments (usually overridden CLI or via basic YAML)
    data_path: str = "."
    images: str = "images"
    depths: str = ""
    eval: bool = False
    train_test_exp: bool = False

    # Wandb integration
    wandb_project: str = "cl-splats"
    wandb_run_name: str = ""
    wandb_mode: str = "online"
