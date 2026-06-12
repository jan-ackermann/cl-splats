"""Regression tests for 3DGS-aligned loading and optimisation behavior."""

import sys
import types
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _mock_gsplat():
    if "gsplat" not in sys.modules:
        mock_gsplat = types.ModuleType("gsplat")
        mock_gsplat.Strategy = type("Strategy", (), {})  # type: ignore
        mock_gsplat.DefaultStrategy = _MockDefaultStrategy  # type: ignore
        sys.modules["gsplat"] = mock_gsplat
    if "gsplat.rendering" not in sys.modules:
        mock_rendering = types.ModuleType("gsplat.rendering")
        mock_rendering.rasterization = mock.MagicMock()  # type: ignore
        sys.modules["gsplat.rendering"] = mock_rendering
    if "gsplat.exporter" not in sys.modules:
        mock_exporter = types.ModuleType("gsplat.exporter")
        mock_exporter.export_splats = mock.MagicMock()  # type: ignore
        sys.modules["gsplat.exporter"] = mock_exporter


class _MockDefaultStrategy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pre_calls = []
        self.post_calls = []

    def check_sanity(self, params, optimizers):
        assert set(params.keys()) == set(optimizers.keys())

    def initialize_state(self, scene_scale=1.0):
        return {"scene_scale": scene_scale}

    def step_pre_backward(self, *args, **kwargs):
        self.pre_calls.append((args, kwargs))

    def step_post_backward(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))


class _PointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


class _CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    timestep: int = 0


class _SceneInfo(NamedTuple):
    point_cloud: _PointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def _make_gaussians(n=3, cfg=None, spatial_lr_scale=1.0, sh_k=1):
    from clsplats.config import CLSplatsConfig
    from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams

    cfg = cfg or CLSplatsConfig()
    params = GaussianParams(
        positions=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        scales=torch.full((n, 3), 0.01),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        sh_features=torch.zeros(n, 3, sh_k),
        opacity=torch.full((n, 1), 0.1),
    )
    return CLGaussians(cfg, params, spatial_lr_scale=spatial_lr_scale)


def test_initial_scales_follow_nearest_neighbor_spacing():
    _mock_gsplat()
    from clsplats.trainer import _initial_scales_from_points

    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    scales = _initial_scales_from_points(xyz, fallback_scale=0.01)

    assert scales.shape == (3, 3)
    assert torch.allclose(scales[0], torch.full((3,), 2.0))
    assert torch.allclose(scales[1], torch.full((3,), 2.0))
    assert torch.allclose(scales[2], torch.full((3,), 8.0))


def test_trainer_initializes_sh_from_rgb_and_scales_from_spacing(tmp_path, monkeypatch):
    _mock_gsplat()
    import clsplats.trainer as trainer_mod
    from clsplats.config import CLSplatsConfig
    from clsplats.trainer import CLSplatsTrainer
    from clsplats.utils.sh_utils import RGB2SH

    monkeypatch.setattr(trainer_mod, "DinoV2Detector", mock.Mock())
    monkeypatch.setattr(trainer_mod, "DepthAnythingLifter", mock.Mock())

    img_path = tmp_path / "img.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(img_path)

    points = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32)
    scene = _SceneInfo(
        point_cloud=_PointCloud(points=points, colors=colors, normals=np.zeros_like(points)),
        train_cameras=[
            _CameraInfo(
                uid=0,
                R=np.eye(3),
                T=np.zeros(3),
                FovY=1.0,
                FovX=1.0,
                image_path=str(img_path),
                image_name="img.png",
                width=2,
                height=2,
                is_test=False,
            )
        ],
        test_cameras=[],
        nerf_normalization={"radius": 1.0},
        ply_path="",
        is_nerf_synthetic=False,
    )

    trainer = CLSplatsTrainer(CLSplatsConfig(), scene)  # type: ignore[arg-type]

    expected_dc = RGB2SH(torch.from_numpy(colors))
    assert torch.allclose(trainer.gaussians.params.sh_features[:, :, 0].cpu(), expected_dc)
    assert torch.allclose(trainer.gaussians.params.scales.cpu(), torch.full((2, 3), 4.0))


def test_optimizer_uses_3dgs_like_param_lrs_and_xyz_schedule():
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig

    cfg = CLSplatsConfig()
    gaussians = _make_gaussians(cfg=cfg)
    lrs = {
        name: optimizer.param_groups[0]["lr"]
        for name, optimizer in gaussians.optimizers.items()
    }

    assert lrs["means"] == cfg.train.position_lr_init
    assert lrs["sh0"] == cfg.train.feature_lr
    # 3DGS trains the higher-order SH coefficients at feature_lr / 20.
    assert lrs["shN"] == cfg.train.feature_lr / 20.0
    assert lrs["opacities"] == cfg.train.opacity_lr
    assert lrs["scales"] == cfg.train.scaling_lr
    assert lrs["quats"] == cfg.train.rotation_lr
    assert gaussians.optimizers["means"].param_groups[0]["name"] == "xyz"
    assert gaussians.optimizers["sh0"].param_groups[0]["name"] == "f_dc"
    assert gaussians.optimizers["shN"].param_groups[0]["name"] == "f_rest"
    assert gaussians.optimizers["opacities"].param_groups[0]["name"] == "opacity"
    assert gaussians.optimizers["scales"].param_groups[0]["name"] == "scaling"
    assert gaussians.optimizers["quats"].param_groups[0]["name"] == "rotation"

    # The reference builds get_expon_lr_func with lr_delay_steps=0, so the
    # sine delay never applies: the schedule starts at lr_init exactly.
    first_lr = gaussians.update_learning_rate(0)
    final_lr = gaussians.update_learning_rate(cfg.train.position_lr_max_steps)
    assert first_lr == cfg.train.position_lr_init
    assert final_lr == pytest.approx(cfg.train.position_lr_final)


def test_position_lr_scales_with_scene_extent():
    """3DGS multiplies the position LR by spatial_lr_scale (camera extent)."""
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig

    cfg = CLSplatsConfig()
    gaussians = _make_gaussians(cfg=cfg, spatial_lr_scale=2.5)

    assert gaussians.optimizers["means"].param_groups[0]["lr"] == pytest.approx(
        cfg.train.position_lr_init * 2.5
    )
    assert gaussians.update_learning_rate(0) == pytest.approx(
        cfg.train.position_lr_init * 2.5
    )
    assert gaussians.update_learning_rate(cfg.train.position_lr_max_steps) == pytest.approx(
        cfg.train.position_lr_final * 2.5
    )
    # Non-position learning rates are not scaled.
    assert gaussians.optimizers["scales"].param_groups[0]["lr"] == cfg.train.scaling_lr


def test_render_camera_passes_row_major_world_to_camera(monkeypatch):
    """Camera stores the 3DGS transposed convention; gsplat needs W2C."""
    _mock_gsplat()
    import clsplats.trainer as trainer_mod
    from clsplats.config import CLSplatsConfig
    from clsplats.trainer import CLSplatsTrainer

    captured = {}

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        h, w = kwargs["height"], kwargs["width"]
        return torch.zeros(1, h, w, 3), torch.zeros(1, h, w, 1), {}

    monkeypatch.setattr(trainer_mod, "rasterization", fake_rasterization)

    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.cfg = CLSplatsConfig()
    trainer.device = torch.device("cpu")
    trainer.gaussians = _make_gaussians(n=2)

    # A realistic stored matrix: W2C transposed (translation in bottom row).
    w2c = torch.eye(4)
    w2c[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    cam = mock.Mock()
    cam.world_view_transform = w2c.transpose(0, 1)
    cam.fx, cam.fy, cam.cx, cam.cy = 10.0, 10.0, 1.5, 1.5
    cam.image_width, cam.image_height = 4, 4

    trainer._render_camera(cam)

    # gsplat must receive the plain row-major world-to-camera matrix.
    assert torch.allclose(captured["viewmats"][0], w2c)


def test_photometric_loss_uses_l1_when_dssim_disabled():
    _mock_gsplat()
    from clsplats.trainer import _photometric_loss

    rendered = torch.zeros(16, 16, 3)
    gt = torch.ones(16, 16, 3)

    loss = _photometric_loss(rendered, gt, lambda_dssim=0.0)

    assert loss.item() == 1.0


def test_train_step_applies_alpha_mask_before_loss(monkeypatch):
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig
    from clsplats.trainer import CLSplatsTrainer

    cfg = CLSplatsConfig()
    cfg.train.lambda_dssim = 0.0

    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.timestep = 0
    trainer.active_mask = None
    trainer._primitives = []
    trainer._global_step = 0
    trainer._timestep_iter = 0
    trainer._outside_counts = None

    rendered_param = torch.nn.Parameter(torch.ones(4, 4, 3))
    trainer._render_camera = mock.Mock(  # type: ignore[method-assign]
        return_value=(rendered_param, {"means2d": torch.zeros(1, 4, 2, requires_grad=True)})
    )
    trainer.gaussians = mock.Mock()
    trainer.gaussians.num_gaussians = 4
    trainer.gaussians.update_learning_rate = mock.Mock()
    trainer.gaussians.update_sh_degree = mock.Mock()
    trainer.gaussians.step_pre_backward = mock.Mock()
    trainer.gaussians.mask_inactive_gradients = mock.Mock()
    trainer.gaussians.step_post_backward = mock.Mock(return_value=(None, None))
    trainer.gaussians.step_optimizer = mock.Mock()

    cam = mock.Mock()
    cam.original_image = torch.zeros(3, 4, 4)
    cam.alpha_mask = torch.zeros(1, 4, 4)

    stats = trainer._train_step(cam)

    assert stats["loss"] == 0.0
    trainer.gaussians.update_learning_rate.assert_called_once_with(1)
    trainer.gaussians.update_sh_degree.assert_called_once_with(1)
    trainer.gaussians.step_pre_backward.assert_called_once()
    trainer.gaussians.step_post_backward.assert_called_once()
    trainer.gaussians.step_optimizer.assert_called_once()
    assert trainer._timestep_iter == 1


def test_gsplat_strategy_uses_original_3dgs_gradient_key_and_schedule():
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig

    cfg = CLSplatsConfig()
    gaussians = _make_gaussians(n=2, cfg=cfg)

    assert gaussians.strategy.kwargs["key_for_gradient"] == "means2d"
    assert gaussians.strategy.kwargs["grow_grad2d"] == cfg.train.densify_grad_threshold
    assert gaussians.strategy.kwargs["refine_start_iter"] == cfg.train.densify_from_iter
    assert gaussians.strategy.kwargs["refine_every"] == cfg.train.densification_interval
    assert gaussians.strategy.kwargs["reset_every"] == cfg.train.opacity_reset_interval


def test_setup_timestep_cl_phase_never_touches_inactive_gaussians():
    """The CL-phase strategy must not opacity-reset (inactive Gaussians can't
    recover) nor opacity-prune globally (would delete frozen rows and break
    exact history recovery)."""
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig

    cfg = CLSplatsConfig()
    gaussians = _make_gaussians(n=2, cfg=cfg)

    gaussians.setup_timestep(scene_scale=2.5, cl_phase=False)
    assert gaussians.strategy.kwargs["reset_every"] == cfg.train.opacity_reset_interval
    assert gaussians.strategy.kwargs["prune_opa"] == cfg.train.densify_prune_opa
    assert gaussians.strategy_state["scene_scale"] == 2.5

    gaussians.setup_timestep(scene_scale=2.5, cl_phase=True)
    assert gaussians.strategy.kwargs["reset_every"] > 10**9
    assert gaussians.strategy.kwargs["prune_opa"] == 0.0


def test_cl_masks_realign_through_strategy_split():
    """gsplat's split removes parents mid-array and appends children; the CL
    masks travel through the ops as flag tensors and stay index-aligned."""
    _mock_gsplat()
    from torch import nn

    gaussians = _make_gaussians(n=3)
    active = torch.tensor([True, False, True])
    counts = torch.tensor([1, 2, 3])

    class _SplittingStrategy(_MockDefaultStrategy):
        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            # Mimic gsplat splitting index 0: parent removed, 2 children appended.
            for name in list(params.keys()):
                p = params[name].detach()
                params[name] = nn.Parameter(torch.cat([p[1:], p[0:1], p[0:1]], dim=0))

    gaussians.strategy = _SplittingStrategy()
    new_active, new_counts = gaussians.step_post_backward(0, {}, active, counts)

    assert new_active is not None and new_counts is not None
    assert new_active.tolist() == [False, True, True, True]
    assert new_counts.tolist() == [2, 3, 1, 1]
    assert gaussians.num_gaussians == 4


def test_cl_masks_realign_through_strategy_prune():
    _mock_gsplat()
    from torch import nn

    gaussians = _make_gaussians(n=4)
    active = torch.tensor([True, False, True, False])
    counts = torch.tensor([1, 2, 3, 4])

    class _PruningStrategy(_MockDefaultStrategy):
        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            # Mimic gsplat pruning index 1 (mid-array removal).
            keep = torch.tensor([True, False, True, True])
            for name in list(params.keys()):
                params[name] = nn.Parameter(params[name].detach()[keep])

    gaussians.strategy = _PruningStrategy()
    new_active, new_counts = gaussians.step_post_backward(0, {}, active, counts)

    assert new_active is not None and new_counts is not None
    assert new_active.tolist() == [True, True, False]
    assert new_counts.tolist() == [1, 3, 4]


def test_prune_preserves_surviving_raw_values_bitwise():
    """Pruning must subset raw tensors, not round-trip them through the
    activated view (normalize/sigmoid/exp) — survivors stay bit-identical,
    which exact history recovery depends on."""
    _mock_gsplat()

    gaussians = _make_gaussians(n=4)
    device = gaussians.device
    # Un-normalised quats and extreme opacity logits are exactly the values
    # an activation round-trip would corrupt.
    gaussians.strategy_params["quats"].data = torch.randn(4, 4, device=device) * 3.0
    gaussians.strategy_params["opacities"].data = torch.tensor(
        [20.0, -20.0, 0.3, 5.0], device=device
    )
    before = {k: v.detach().clone() for k, v in gaussians.strategy_params.items()}

    keep = gaussians.prune_gaussians(torch.tensor([False, True, False, False]))

    surviving = torch.nonzero(keep).squeeze(-1)
    for key, old in before.items():
        new = gaussians.strategy_params[key].detach()
        assert torch.equal(new, old[surviving]), key


def test_trainer_prune_keeps_cl_flags_aligned():
    _mock_gsplat()

    gaussians = _make_gaussians(n=4)
    active = torch.tensor([True, False, True, False])
    counts = torch.tensor([1, 2, 3, 4])
    gaussians.strategy_params["cl_active"].data.copy_(active.float())
    gaussians.strategy_params["cl_outside_count"].data.copy_(counts.float())

    keep = gaussians.prune_gaussians(torch.tensor([False, True, False, False]))

    assert keep.tolist() == [True, False, True, True]
    assert gaussians.strategy_params["cl_active"].detach().tolist() == [1.0, 1.0, 0.0]
    assert gaussians.strategy_params["cl_outside_count"].detach().tolist() == [1.0, 3.0, 4.0]


def test_means2d_gradient_masking_aligns_with_gaussian_dim():
    """means2d grad is [C, N, 2]; the [N] mask must broadcast on dim -2."""
    _mock_gsplat()

    gaussians = _make_gaussians(n=3)
    means2d = torch.zeros(1, 3, 2, requires_grad=True)
    means2d.grad = torch.ones(1, 3, 2)
    active = torch.tensor([True, False, True])

    gaussians.step_post_backward(0, {"means2d": means2d}, active, None)

    assert torch.equal(means2d.grad[0, :, 0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(means2d.grad[0, :, 1], torch.tensor([1.0, 0.0, 1.0]))


def test_reset_inactive_optimizer_state_zeros_adam_moments():
    """Masked grads alone don't freeze params — stale Adam momentum moves them."""
    _mock_gsplat()

    gaussians = _make_gaussians(n=2)
    means = gaussians.strategy_params["means"]
    loss = means.sum()
    loss.backward()
    gaussians.step_optimizer()

    state = gaussians.optimizers["means"].state[means]
    assert state["exp_avg"].abs().sum() > 0

    gaussians.reset_inactive_optimizer_state(torch.tensor([True, False]))

    assert state["exp_avg"][1].abs().sum() == 0
    assert state["exp_avg_sq"][1].abs().sum() == 0
    assert state["exp_avg"][0].abs().sum() > 0


def test_sh_degree_progresses_like_original_3dgs():
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig

    cfg = CLSplatsConfig()
    cfg.model.sh_degree = 3
    gaussians = _make_gaussians(n=2, cfg=cfg, sh_k=16)

    assert gaussians.active_sh_degree == 0
    assert gaussians.update_sh_degree(999) == 0
    assert gaussians.update_sh_degree(1000) == 1
    assert gaussians.update_sh_degree(3000) == 3
    assert gaussians.update_sh_degree(4000) == 3


def test_export_ply_passes_raw_log_scales_and_logit_opacities():
    """export_splats writes values verbatim: log-scales and logit-opacities."""
    _mock_gsplat()

    gaussians = _make_gaussians(n=2)
    exporter = sys.modules["gsplat.exporter"]
    exporter.export_splats.reset_mock()

    gaussians.export_ply("/tmp/test_export.ply")

    kwargs = exporter.export_splats.call_args.kwargs
    assert torch.allclose(kwargs["scales"], gaussians.strategy_params["scales"].detach())
    assert torch.allclose(kwargs["opacities"], gaussians.strategy_params["opacities"].detach())
    # Log scales of 0.01 are negative — i.e. NOT the activated linear values.
    assert (kwargs["scales"] < 0).all()
    assert kwargs["sh0"].shape == (2, 1, 3)


def test_relative_depth_alignment_recovers_metric_scale():
    """Mono depth is affine-invariant disparity; alignment must recover metric z."""
    _mock_gsplat()
    from clsplats.lifter.depth_anything_lifter import align_relative_depth

    torch.manual_seed(0)
    z_true = torch.rand(64, 64) * 2.0 + 1.5  # metric depth in [1.5, 3.5]
    a_true, b_true = 0.7, 0.05
    mono = (1.0 / z_true - b_true) / a_true  # what a MiDaS-style model returns

    change = torch.zeros(64, 64, dtype=torch.bool)
    change[20:30, 20:30] = True  # changed region: rendered depth unreliable
    rendered = z_true.clone()
    rendered[change] = 0.0  # pretend nothing rendered there

    aligned = align_relative_depth(mono, rendered, change)

    assert aligned is not None
    # Metric depth recovered everywhere, including inside the changed region.
    assert torch.allclose(aligned, z_true, rtol=1e-3)


def test_relative_depth_alignment_rejects_garbage_fit():
    _mock_gsplat()
    from clsplats.lifter.depth_anything_lifter import align_relative_depth

    torch.manual_seed(0)
    rendered = torch.rand(64, 64) * 2.0 + 1.5
    mono = torch.rand(64, 64) * 1000.0  # uncorrelated noise
    change = torch.zeros(64, 64, dtype=torch.bool)

    assert align_relative_depth(mono, rendered, change) is None
    # Too few anchor pixels must also be rejected.
    all_changed = torch.ones(64, 64, dtype=torch.bool)
    assert align_relative_depth(mono, rendered, all_changed) is None


def test_gt_image_composited_onto_background(tmp_path):
    _mock_gsplat()
    from clsplats.trainer import _load_gt_image

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[0, 0] = [200, 30, 30, 255]  # opaque red-ish pixel
    rgba[1, 1] = [70, 70, 70, 0]  # fully transparent pixel with junk colour
    img_path = tmp_path / "img.png"
    Image.fromarray(rgba, "RGBA").save(img_path)

    white = np.array(_load_gt_image(str(img_path), white_background=True, composite_alpha=True))
    black = np.array(_load_gt_image(str(img_path), white_background=False, composite_alpha=True))
    untouched = _load_gt_image(str(img_path), white_background=True, composite_alpha=False)

    assert tuple(white[0, 0]) == (200, 30, 30)
    assert tuple(white[1, 1]) == (255, 255, 255)
    assert tuple(black[1, 1]) == (0, 0, 0)
    assert untouched.mode == "RGBA"


def test_trainer_samples_cameras_random_without_replacement(monkeypatch):
    _mock_gsplat()
    import clsplats.trainer as trainer_mod
    from clsplats.trainer import CLSplatsTrainer

    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.train_cameras = ["cam0", "cam1", "cam2"]
    trainer._viewpoint_stack = []
    trainer._viewpoint_indices = []
    trainer._reset_viewpoint_stack()

    picks = iter([1, 0, 0])
    monkeypatch.setattr(trainer_mod, "randint", lambda _low, _high: next(picks))

    assert trainer._pop_random_train_camera() == "cam1"
    assert trainer._pop_random_train_camera() == "cam0"
    assert trainer._pop_random_train_camera() == "cam2"
    assert trainer._viewpoint_stack == []


def test_constraint_pruning_never_touches_inactive_gaussians():
    """The frozen scene must be preserved no matter how far it is from the
    change-region primitives; only active Gaussians may be pruned."""
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig
    from clsplats.constraints.primitives import SpherePrimitive
    from clsplats.trainer import CLSplatsTrainer

    cfg = CLSplatsConfig()
    cfg.constraints.prune_every = 1
    cfg.constraints.prune_consecutive = 2
    cfg.constraints.prune_dist_thresh = 0.1

    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.cfg = cfg
    trainer.gaussians = _make_gaussians(n=3)
    device = trainer.gaussians.device
    trainer.device = device
    trainer.timestep = 1
    # Gaussian 0: active, inside primitive. Gaussian 1: active, far outside.
    # Gaussian 2: INACTIVE, far outside — must survive.
    trainer.gaussians.strategy_params["means"].data = torch.tensor(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 9.0, 9.0]], device=device
    )
    trainer.active_mask = torch.tensor([True, True, False], device=device)
    trainer._outside_counts = None
    trainer._primitives = [
        ("sphere", SpherePrimitive(center=torch.zeros(3, device=device), radius=1.0))
    ]

    trainer._constraint_prune_step(1)
    trainer._constraint_prune_step(2)

    assert trainer.gaussians.num_gaussians == 2
    assert trainer.active_mask.tolist() == [True, False]
    positions = trainer.gaussians.params.positions.detach().cpu()
    assert torch.allclose(positions[1], torch.tensor([9.0, 9.0, 9.0]))


def test_sync_after_strategy_no_longer_truncates_masks():
    """Masks are realigned by step_post_backward; _sync must not touch them."""
    _mock_gsplat()
    from clsplats.config import CLSplatsConfig
    from clsplats.trainer import CLSplatsTrainer

    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.cfg = CLSplatsConfig()
    trainer.device = torch.device("cpu")
    trainer.timestep = 0
    # Already realigned by step_post_backward (e.g. after a mid-array prune).
    trainer.active_mask = torch.tensor([True, False, True])
    trainer._outside_counts = torch.tensor([1, 2, 3])
    trainer.gaussians = mock.Mock()
    trainer.gaussians.num_gaussians = 3

    trainer._sync_after_strategy(old_count=4)

    assert torch.equal(trainer.active_mask, torch.tensor([True, False, True]))
    assert torch.equal(trainer._outside_counts, torch.tensor([1, 2, 3]))
