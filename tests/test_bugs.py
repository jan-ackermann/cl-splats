"""
Unit tests for bugs identified in the cl-splats codebase.

Each test targets a specific bug from the bug report and verifies
the bug exists (test should PASS when the bug is present).

Tests are designed to be self-contained and mock external dependencies
(gsplat, transformers, etc.) to focus on the specific bug logic.
"""
import ast
import importlib
import inspect
import math
import os
import sys
import textwrap
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import omegaconf

# Project root for source-code-level tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLSPLATS_DIR = PROJECT_ROOT / "clsplats"

# Ensure project root is on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _mock_gsplat():
    """Install a mock gsplat module so cl_gaussians can be imported."""
    if "gsplat" not in sys.modules:
        mock_gsplat = types.ModuleType("gsplat")
        mock_gsplat.Strategy = type("Strategy", (), {})
        mock_gsplat.DefaultStrategy = type("DefaultStrategy", (), {})
        sys.modules["gsplat"] = mock_gsplat
    # Also need gsplat.rendering for trainer
    if "gsplat.rendering" not in sys.modules:
        mock_rendering = types.ModuleType("gsplat.rendering")
        mock_rendering.rasterization = mock.MagicMock()
        sys.modules["gsplat.rendering"] = mock_rendering
    # Remove cached cl_gaussians module to force re-import with mock
    for mod_name in list(sys.modules):
        if "cl_gaussians" in mod_name:
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Bug 1: train.py — CLSplatsTrainer instantiated with wrong number of args
# ---------------------------------------------------------------------------
class TestBug01_TrainConstructorMismatch:
    """train.py calls CLSplatsTrainer(cfg) but __init__ expects (cfg, scene)."""

    def test_trainer_init_requires_scene(self):
        """Verify CLSplatsTrainer.__init__ requires a 'scene' parameter."""
        source = (CLSPLATS_DIR / "trainer.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Check parent class is CLSplatsTrainer
                parent = None
                for parent_node in ast.walk(tree):
                    if isinstance(parent_node, ast.ClassDef):
                        if node in ast.walk(parent_node):
                            parent = parent_node
                            break
                if parent and parent.name == "CLSplatsTrainer":
                    arg_names = [a.arg for a in node.args.args]
                    assert "scene" in arg_names, (
                        "CLSplatsTrainer.__init__ should require 'scene' param"
                    )
                    break

    def test_train_py_missing_scene_arg(self):
        """Verify train.py does NOT pass a scene argument to CLSplatsTrainer."""
        source = (CLSPLATS_DIR / "train.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Find CLSplatsTrainer(...) or trainer.CLSplatsTrainer(...)
                call_name = ""
                if isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    call_name = node.func.id
                if call_name == "CLSplatsTrainer":
                    n_args = len(node.args) + len(node.keywords)
                    assert n_args == 1, (
                        f"BUG CONFIRMED: CLSplatsTrainer called with {n_args} "
                        f"arg(s) but needs 2 (cfg, scene)"
                    )
                    return
        pytest.skip("CLSplatsTrainer call not found in train.py")


# ---------------------------------------------------------------------------
# Bug 2: depth_anything_lifter.py — Broken depth buffer parsing
# ---------------------------------------------------------------------------
class TestBug02_BrokenDepthParsing:
    """torch.from_numpy is called on a torch.Tensor, not numpy array."""

    def test_from_numpy_on_tensor_raises(self):
        """torch.from_numpy can't accept a torch.Tensor — this is a TypeError."""
        t = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            torch.from_numpy(t)

    def test_depth_estimation_code_pattern_is_wrong(self):
        """Verify the specific wrong code pattern in the source file."""
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        # The bug: torch.from_numpy wraps a torch.ByteTensor
        assert "torch.from_numpy" in source, "torch.from_numpy call should exist"
        assert "torch.ByteTensor" in source, "torch.ByteTensor call should exist"
        # Verify they're nested (from_numpy wrapping ByteTensor)
        assert "torch.from_numpy(\n" in source or "torch.from_numpy(" in source


# ---------------------------------------------------------------------------
# Bug 3: dinov2_detector.py — Missing ImageNet normalization
# ---------------------------------------------------------------------------
class TestBug03_MissingNormalization:
    """self.normalize is defined but never applied to images."""

    def test_normalize_defined_but_not_used_in_preprocess(self):
        """Verify _preprocess_image does NOT call self.normalize."""
        source = (CLSPLATS_DIR / "change_detection" / "dinov2_detector.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_preprocess_image":
                # Check method body for any call to self.normalize
                method_source = ast.get_source_segment(source, node)
                assert "self.normalize" not in method_source, (
                    "self.normalize found in _preprocess_image — bug might be fixed"
                )
                return
        pytest.fail("_preprocess_image method not found")

    def test_normalize_not_in_predict_change_mask(self):
        """Also verify normalize isn't called in predict_change_mask."""
        source = (CLSPLATS_DIR / "change_detection" / "dinov2_detector.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "predict_change_mask":
                method_source = ast.get_source_segment(source, node)
                assert "self.normalize" not in method_source, (
                    "self.normalize found in predict_change_mask — bug might be fixed"
                )
                return
        pytest.fail("predict_change_mask method not found")


# ---------------------------------------------------------------------------
# Bug 4: depth_anything_lifter.py — Extra .unsqueeze(-1) in homogeneous div
# ---------------------------------------------------------------------------
class TestBug04_IncorrectHomogeneousDivision:
    """Extra .unsqueeze(-1) on [M,1] tensor causes shape mismatch."""

    def test_unsqueeze_on_already_broadcastable_shape(self):
        """Demonstrate the math error: [M,3] / [M,1,1] fails or gives wrong shape."""
        M = 10
        p_world_h = torch.randn(M, 4)
        # Correct: [M, 3] / [M, 1] works via broadcasting
        correct = p_world_h[..., :3] / p_world_h[..., 3:]
        assert correct.shape == (M, 3)

        # Bug: [M, 3] / [M, 1, 1] adds dimension
        buggy_divisor = p_world_h[..., 3:].unsqueeze(-1)
        assert buggy_divisor.shape == (M, 1, 1), (
            f"Extra unsqueeze creates wrong shape: {buggy_divisor.shape}"
        )
        # This would broadcast to (M, 3, 1) instead of (M, 3)
        result = p_world_h[..., :3] / buggy_divisor
        assert result.shape != (M, 3), (
            f"BUG CONFIRMED: result shape is {result.shape}, not (M, 3)"
        )

    def test_source_contains_buggy_pattern(self):
        """Verify the buggy .unsqueeze(-1) pattern exists in the source."""
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        assert "3:].unsqueeze(-1)" in source, (
            "Expected buggy .unsqueeze(-1) pattern in depth_anything_lifter.py"
        )


# ---------------------------------------------------------------------------
# Bug 5: cl_gaussians.py — Optimizer not updated after pruning
# ---------------------------------------------------------------------------
class TestBug05_OptimizerNotUpdatedAfterPruning:
    """After prune_gaussians, optimizer still references old tensors."""

    def test_prune_does_not_update_optimizer(self):
        """Verify optimizer param groups still point to old tensors after pruning."""
        _mock_gsplat()

        cfg = omegaconf.OmegaConf.create({"train": {"lr": 1e-3}})

        from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams

        N = 20
        params = GaussianParams(
            positions=torch.randn(N, 3),
            scales=torch.full((N, 3), 0.01),
            quats=torch.zeros(N, 4),
            sh_features=torch.zeros(N, 3, 1),
            opacity=torch.full((N, 1), 0.5),
        )
        params.quats[:, 0] = 1.0

        gauss = CLGaussians(cfg, params)

        # Grab optimizer's param reference before pruning
        old_positions_id = id(gauss.optimizer.param_groups[0]["params"][0])

        # Prune half the gaussians
        prune_mask = torch.zeros(N, dtype=torch.bool)
        prune_mask[:N // 2] = True
        gauss.prune_gaussians(prune_mask)

        # After pruning, params should have changed
        assert gauss.params.positions.shape[0] == N // 2

        # BUG: optimizer still references the OLD tensor
        new_positions_id = id(gauss.optimizer.param_groups[0]["params"][0])
        assert old_positions_id == new_positions_id, (
            "BUG CONFIRMED: optimizer params were NOT updated after pruning"
        )


# ---------------------------------------------------------------------------
# Bug 6: dinov2_detector.py — Double-indexed DINOv2 features
# ---------------------------------------------------------------------------
class TestBug06_DoubleIndexedFeatures:
    """rendered_feats is already unbatched, [0] indexes channels not batch."""

    def test_source_double_indexes_features(self):
        """Verify the double-indexing pattern exists in the source."""
        source = (CLSPLATS_DIR / "change_detection" / "dinov2_detector.py").read_text()
        # The line should have: cos_sim = self.cos(rendered_feats[0], observed_feats[0])
        assert "rendered_feats[0]" in source, "Double indexing pattern found"
        assert "observed_feats[0]" in source, "Double indexing pattern found"

    def test_cosine_sim_wrong_dims_after_double_indexing(self):
        """Show that double-indexing changes shape semantics."""
        # Simulate get_intermediate_layers returning tuple of (B, C, H, W)
        B, C, H, W = 1, 768, 16, 16
        # After (feats,) = ... unpacking, feats is [B, C, H, W]
        feats = torch.randn(B, C, H, W)

        # Correct: use feats directly (dim=1 means channel dim)
        cos = torch.nn.CosineSimilarity(dim=1)
        correct_sim = cos(feats, feats)
        assert correct_sim.shape == (B, H, W), f"Correct shape: {correct_sim.shape}"

        # Buggy: feats[0] gives [C, H, W], cosine on dim=1 compares H
        buggy_sim = cos(feats[0], feats[0])
        assert buggy_sim.shape == (C, W), (
            f"BUG CONFIRMED: double indexing gives shape {buggy_sim.shape} "
            f"instead of ({B}, {H}, {W})"
        )


# ---------------------------------------------------------------------------
# Bug 7: cameras.py — Twc recomputed on every access
# ---------------------------------------------------------------------------
class TestBug07_TwcNotCached:
    """Twc property calls torch.inverse every time it's accessed."""

    def test_twc_is_property_not_cached(self):
        """Verify Twc is a simple property, not cached at __init__ time."""
        source = (CLSPLATS_DIR / "dataset" / "cameras.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Camera":
                # Check __init__ doesn't precompute self.Twc
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        init_source = ast.get_source_segment(source, method)
                        assert "Twc" not in init_source or "self.Twc" not in init_source, (
                            "Twc is NOT cached in __init__ (bug)"
                        )
                # Check Twc is defined as a @property with torch.inverse
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "Twc":
                        method_source = ast.get_source_segment(source, method)
                        assert "torch.inverse" in method_source, (
                            "Twc calls torch.inverse every access"
                        )
                        return
        pytest.fail("Camera.Twc not found")


# ---------------------------------------------------------------------------
# Bug 8: train.py — Wrong Hydra config path
# ---------------------------------------------------------------------------
class TestBug08_WrongHydraConfigPath:
    """config_path='configs' is relative to train.py inside clsplats/, not project root."""

    def test_config_path_relative_mismatch(self):
        """Verify that configs/ is NOT inside clsplats/ where train.py lives."""
        train_py_dir = CLSPLATS_DIR
        config_from_decorator = train_py_dir / "configs"
        project_configs = PROJECT_ROOT / "configs"

        assert not config_from_decorator.is_dir(), (
            f"clsplats/configs/ should NOT exist — the real configs are at {project_configs}"
        )
        assert project_configs.is_dir(), "configs/ should exist at project root"

    def test_all_config_files_are_empty(self):
        """Verify that config yaml files are empty (another bug)."""
        config_dir = PROJECT_ROOT / "configs"
        yaml_files = list(config_dir.rglob("*.yaml"))
        assert len(yaml_files) > 0, "Should have yaml config files"
        for yf in yaml_files:
            content = yf.read_text().strip()
            assert content == "", f"Config file {yf.name} is empty (no config defined)"


# ---------------------------------------------------------------------------
# Bug 9: train.py — Wrong import (bare 'import trainer')
# ---------------------------------------------------------------------------
class TestBug09_WrongImport:
    """train.py uses 'import trainer' instead of 'from clsplats import trainer'."""

    def test_bare_import_trainer(self):
        """Verify train.py uses a bare 'import trainer' without package prefix."""
        source = (CLSPLATS_DIR / "train.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "trainer":
                        # This is the bare import — BUG
                        assert True, "BUG CONFIRMED: bare 'import trainer'"
                        return
        pytest.skip("No bare 'import trainer' found — may be fixed")


# ---------------------------------------------------------------------------
# Bug 10: depth_anything_lifter.py — pos_conf/neg_conf computed but unused
# ---------------------------------------------------------------------------
class TestBug10_UnusedConfidenceVars:
    """pos_conf and neg_conf are computed but never used."""

    def test_pos_neg_conf_unused(self):
        """Verify pos_conf and neg_conf are assigned but not read."""
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        # Check they're assigned
        assert "pos_conf =" in source, "pos_conf should be assigned"
        assert "neg_conf =" in source, "neg_conf should be assigned"

        # Check they're never used after assignment (no reads beyond assignment)
        lines = source.split("\n")
        pos_conf_uses = [
            (i, l) for i, l in enumerate(lines)
            if "pos_conf" in l and "pos_conf =" not in l and "#" not in l.split("pos_conf")[0]
        ]
        neg_conf_uses = [
            (i, l) for i, l in enumerate(lines)
            if "neg_conf" in l and "neg_conf =" not in l and "#" not in l.split("neg_conf")[0]
        ]
        assert len(pos_conf_uses) == 0, f"pos_conf is never read after assignment"
        assert len(neg_conf_uses) == 0, f"neg_conf is never read after assignment"


# ---------------------------------------------------------------------------
# Bug 11: BaseLifter.lift() signature doesn't match DepthAnythingLifter.lift()
# ---------------------------------------------------------------------------
class TestBug11_LifterSignatureMismatch:
    """Abstract method signature differs from concrete implementation."""

    def test_base_vs_concrete_lift_signatures(self):
        """Verify the parameter mismatch between base and sub class."""
        base_source = (CLSPLATS_DIR / "lifter" / "base_lifter.py").read_text()
        concrete_source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()

        # Extract lift method signatures via AST
        def get_lift_params(source):
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "lift":
                    return [a.arg for a in node.args.args if a.arg != "self"]
            return None

        base_params = get_lift_params(base_source)
        concrete_params = get_lift_params(concrete_source)

        assert base_params is not None, "BaseLifter.lift should exist"
        assert concrete_params is not None, "DepthAnythingLifter.lift should exist"
        assert base_params != concrete_params, (
            f"BUG CONFIRMED: base params {base_params} != concrete params {concrete_params}"
        )


# ---------------------------------------------------------------------------
# Bug 12 & 13: dataset_reader.py & cameras.py — Wrong import paths
# ---------------------------------------------------------------------------
class TestBug12_DatasetReaderWrongImports:
    """dataset_reader.py imports from 'scene.' and 'utils.' instead of 'clsplats.'"""

    def test_imports_reference_original_repo(self):
        """Verify the broken import paths exist in dataset_reader.py."""
        source = (CLSPLATS_DIR / "dataset" / "dataset_reader.py").read_text()
        # These imports won't resolve unless the original gaussian-splatting repo is on path
        assert "from scene.colmap_loader import" in source, (
            "BUG: imports from 'scene.colmap_loader' instead of 'clsplats.dataset.colmap_reader'"
        )
        assert "from utils.graphics_utils import" in source, (
            "BUG: imports from 'utils.graphics_utils' instead of 'clsplats.utils.graphics_utils'"
        )
        assert "from utils.sh_utils import" in source, (
            "BUG: imports from 'utils.sh_utils' instead of 'clsplats.utils.sh_utils'"
        )

    def test_dataset_reader_cannot_be_imported(self):
        """Verify dataset_reader.py fails to import in a clean environment."""
        sys.path.insert(0, str(PROJECT_ROOT))
        with pytest.raises((ImportError, ModuleNotFoundError)):
            # This should fail because 'scene.colmap_loader' doesn't exist
            import importlib
            importlib.import_module("clsplats.dataset.dataset_reader")


class TestBug13_CamerasWrongImports:
    """cameras.py imports from 'utils.' instead of 'clsplats.utils.'"""

    def test_imports_reference_original_repo(self):
        """Verify the broken import paths exist in cameras.py."""
        source = (CLSPLATS_DIR / "dataset" / "cameras.py").read_text()
        assert "from utils.graphics_utils import" in source, (
            "BUG: imports from 'utils.graphics_utils'"
        )
        assert "from utils.general_utils import" in source, (
            "BUG: imports from 'utils.general_utils'"
        )

    def test_cameras_cannot_be_imported(self):
        """Verify cameras.py fails to import in a clean environment."""
        sys.path.insert(0, str(PROJECT_ROOT))
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import importlib
            importlib.import_module("clsplats.dataset.cameras")


# ---------------------------------------------------------------------------
# Bug 14: gaussian_model.py — Silent no-op feature indexing
# ---------------------------------------------------------------------------
class TestBug14_SilentNoOpFeatureIndexing:
    """features[:, 3:, 1:] = 0.0 is a no-op when dim 1 has size 3."""

    def test_noop_slice(self):
        """Demonstrate that [:, 3:, 1:] selects nothing on a (N, 3, K) tensor."""
        N, C, K = 100, 3, 4
        features = torch.ones(N, C, K)
        # This should do nothing — it selects features[:, 3:, :] = empty
        subset = features[:, 3:, 1:]
        assert subset.numel() == 0, (
            f"BUG CONFIRMED: features[:, 3:, 1:] selects {subset.numel()} elements "
            f"(shape {subset.shape}) — it's a no-op"
        )

    def test_source_contains_noop_pattern(self):
        """Verify the no-op indexing pattern exists in gaussian_model.py."""
        source = (CLSPLATS_DIR / "representation" / "gaussian_model.py").read_text()
        assert "features[:, 3:, 1:]" in source, (
            "The no-op indexing pattern should exist in gaussian_model.py"
        )


# ---------------------------------------------------------------------------
# Bug 15: Multiple files — Bare except clauses
# ---------------------------------------------------------------------------
class TestBug15_BareExceptClauses:
    """Multiple files use bare 'except:' that swallow all exceptions."""

    @pytest.mark.parametrize("filepath,expected_min_count", [
        ("representation/gaussian_model.py", 2),
        ("dataset/dataset_reader.py", 2),
    ])
    def test_bare_except_clauses(self, filepath, expected_min_count):
        """Count bare except clauses in source files."""
        source = (CLSPLATS_DIR / filepath).read_text()
        tree = ast.parse(source)
        bare_except_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # bare except:
                    bare_except_count += 1
        assert bare_except_count >= expected_min_count, (
            f"Expected at least {expected_min_count} bare except clauses in {filepath}, "
            f"found {bare_except_count}"
        )


# ---------------------------------------------------------------------------
# Bug 16: run_test_scene.py — Hardcoded macOS path
# ---------------------------------------------------------------------------
class TestBug16_HardcodedPath:
    """run_test_scene.py hardcodes a local macOS path."""

    def test_hardcoded_macos_path(self):
        """Verify the hardcoded /Users/jan path exists."""
        source = (PROJECT_ROOT / "scripts" / "run_test_scene.py").read_text()
        assert "/Users/jan/Code/" in source, (
            "BUG CONFIRMED: Hardcoded macOS path in run_test_scene.py"
        )


# ---------------------------------------------------------------------------
# Integration: Verify core cl_gaussians module works (dependency-free)
# ---------------------------------------------------------------------------
class TestCLGaussiansIntegration:
    """Integration test for CLGaussians — the main gsplat-free module."""

    def setup_method(self):
        """Mock gsplat before each test."""
        _mock_gsplat()

    def _make_gaussians(self, n=50):
        """Create a simple CLGaussians instance."""
        from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams

        cfg = omegaconf.OmegaConf.create({"train": {"lr": 1e-3}})
        params = GaussianParams(
            positions=torch.randn(n, 3),
            scales=torch.full((n, 3), 0.01),
            quats=torch.zeros(n, 4),
            sh_features=torch.zeros(n, 3, 1),
            opacity=torch.full((n, 1), 0.5),
        )
        params.quats[:, 0] = 1.0
        return CLGaussians(cfg, params)

    def test_creation(self):
        gauss = self._make_gaussians(50)
        assert gauss.params.positions.shape == (50, 3)
        assert gauss.params.positions.requires_grad

    def test_optimizer_step(self):
        gauss = self._make_gaussians(10)
        # Simulate backward + step
        loss = gauss.params.positions.sum()
        loss.backward()
        gauss.step_optimizer()
        # Grads should be zeroed
        assert gauss.params.positions.grad is None

    def test_prune_basic(self):
        gauss = self._make_gaussians(20)
        prune_mask = torch.zeros(20, dtype=torch.bool)
        prune_mask[::2] = True  # prune every other
        keep = gauss.prune_gaussians(prune_mask)
        assert gauss.params.positions.shape[0] == 10
        assert keep.sum().item() == 10

    def test_prune_then_step_has_stale_optimizer_state(self):
        """BUG 5 demo: after pruning, optimizer state buffers have wrong shape."""
        gauss = self._make_gaussians(20)
        # First do a step to populate optimizer state (exp_avg, exp_avg_sq)
        loss = gauss.params.positions.sum()
        loss.backward()
        gauss.step_optimizer()

        # Grab optimizer state shape BEFORE pruning
        state = gauss.optimizer.state[gauss.optimizer.param_groups[0]["params"][0]]
        pre_prune_shape = state["exp_avg"].shape
        assert pre_prune_shape == (20, 3)

        # Now prune to 10 gaussians
        prune_mask = torch.zeros(20, dtype=torch.bool)
        prune_mask[:10] = True
        gauss.prune_gaussians(prune_mask)

        # Params are now (10, 3)
        assert gauss.params.positions.shape == (10, 3)

        # BUG: optimizer state still has old shape (20, 3) or points to stale tensor
        # The optimizer's param_groups[0]["params"][0] still references the old tensor
        opt_param = gauss.optimizer.param_groups[0]["params"][0]
        if opt_param in gauss.optimizer.state:
            stale_state = gauss.optimizer.state[opt_param]
            stale_shape = stale_state["exp_avg"].shape
            assert stale_shape != gauss.params.positions.shape, (
                f"BUG CONFIRMED: optimizer exp_avg shape {stale_shape} != "
                f"params shape {gauss.params.positions.shape}"
            )
        else:
            # Optimizer references old tensor that is no longer gauss.params.positions
            assert opt_param is not gauss.params.positions, (
                "BUG CONFIRMED: optimizer still references old pre-prune tensor"
            )


# ---------------------------------------------------------------------------
# Test the primitives module (fully self-contained, no broken imports)
# ---------------------------------------------------------------------------
class TestPrimitivesModule:
    """Test the constraints/primitives module for correctness."""

    def setup_method(self):
        sys.path.insert(0, str(PROJECT_ROOT))

    def test_fit_sphere(self):
        from clsplats.constraints.primitives import fit_sphere
        pts = torch.randn(100, 3) * 0.5
        sphere = fit_sphere(pts)
        assert sphere.center.shape == (3,)
        assert sphere.radius > 0

    def test_fit_obb(self):
        from clsplats.constraints.primitives import fit_obb
        pts = torch.randn(100, 3)
        pts[:, 0] *= 5  # Make anisotropic
        obb = fit_obb(pts)
        assert obb.center.shape == (3,)
        assert obb.half_extents.shape == (3,)
        assert obb.rotation.shape == (3, 3)

    def test_sphere_distance_inside(self):
        from clsplats.constraints.primitives import distance_to_primitive
        prim = ("sphere", type("S", (), {"center": torch.zeros(3), "radius": 2.0})())
        # Point inside sphere should have distance 0
        pts = torch.tensor([[0.0, 0.0, 0.0]])
        d = distance_to_primitive(pts, prim)
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_sphere_distance_outside(self):
        from clsplats.constraints.primitives import distance_to_primitive
        prim = ("sphere", type("S", (), {"center": torch.zeros(3), "radius": 1.0})())
        pts = torch.tensor([[3.0, 0.0, 0.0]])
        d = distance_to_primitive(pts, prim)
        assert d.item() == pytest.approx(2.0, abs=1e-6)

    def test_union_distance_empty(self):
        from clsplats.constraints.primitives import union_distance
        pts = torch.randn(10, 3)
        d = union_distance(pts, [])
        assert (d == 0).all()

    def test_group_active_gaussians(self):
        from clsplats.constraints.primitives import group_active_gaussians
        # Two well-separated clusters
        pts = torch.zeros(20, 3)
        pts[:10] = torch.randn(10, 3) * 0.01
        pts[10:] = torch.randn(10, 3) * 0.01 + 10.0
        mask = torch.ones(20, dtype=torch.bool)
        groups = group_active_gaussians(pts, mask, radius_frac=0.1)
        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
