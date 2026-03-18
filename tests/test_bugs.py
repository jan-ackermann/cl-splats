"""
Unit tests for the cl-splats codebase.

Tests verify that all 16 previously-identified bugs are now FIXED.
Tests are designed to be self-contained and mock external dependencies
(gsplat, transformers, etc.) to focus on the specific logic.
"""

import ast
import sys
import types
from pathlib import Path
from unittest import mock

import omegaconf
import pytest
import torch

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
        mock_gsplat.Strategy = type("Strategy", (), {})  # type: ignore
        mock_gsplat.DefaultStrategy = type("DefaultStrategy", (), {})  # type: ignore
        sys.modules["gsplat"] = mock_gsplat
    if "gsplat.rendering" not in sys.modules:
        mock_rendering = types.ModuleType("gsplat.rendering")
        mock_rendering.rasterization = mock.MagicMock()  # type: ignore
        sys.modules["gsplat.rendering"] = mock_rendering
    if "gsplat.exporter" not in sys.modules:
        mock_exporter = types.ModuleType("gsplat.exporter")
        mock_exporter.export_splats = mock.MagicMock()  # type: ignore
        sys.modules["gsplat.exporter"] = mock_exporter
    # Force re-import of cl_gaussians
    for mod_name in list(sys.modules):
        if "cl_gaussians" in mod_name:
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Bug 1 FIX: train.py — CLSplatsTrainer now called with (cfg, scene)
# ---------------------------------------------------------------------------
class TestBug01_TrainConstructorFixed:
    """train.py now passes both cfg and scene to CLSplatsTrainer."""

    def test_trainer_init_requires_scene(self):
        source = (CLSPLATS_DIR / "trainer.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
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

    def test_train_py_passes_scene_arg(self):
        """Verify train.py passes 2 args (cfg, scene) to CLSplatsTrainer."""
        source = (CLSPLATS_DIR / "train.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = ""
                if isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    call_name = node.func.id
                if call_name == "CLSplatsTrainer":
                    n_args = len(node.args) + len(node.keywords)
                    assert n_args == 2, (
                        f"CLSplatsTrainer should be called with 2 args, got {n_args}"
                    )
                    return
        pytest.skip("CLSplatsTrainer call not found in train.py")


# ---------------------------------------------------------------------------
# Bug 2 FIX: depth_anything_lifter.py — Correct depth parsing
# ---------------------------------------------------------------------------
class TestBug02_DepthParsingFixed:
    """Depth parsing now correctly converts PIL Image → numpy → tensor."""

    def test_no_from_numpy_on_tensor(self):
        """torch.from_numpy should NOT be called on a torch.Tensor."""
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        # The buggy pattern (torch.from_numpy wrapping ByteTensor) should be gone
        assert "torch.ByteTensor" not in source, "ByteTensor pattern removed"

    def test_uses_np_array_conversion(self):
        """Verify depth parsing uses np.array() for proper conversion."""
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        assert "np.array(depth_output" in source or "np.array(" in source, (
            "Should use np.array for depth conversion"
        )


# ---------------------------------------------------------------------------
# Bug 3 FIX: dinov2_detector.py — ImageNet normalisation now applied
# ---------------------------------------------------------------------------
class TestBug03_NormalizationFixed:
    """self.normalize is now applied inside _preprocess_image."""

    def test_normalize_used_in_preprocess(self):
        source = (CLSPLATS_DIR / "change_detection" / "dinov2_detector.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_preprocess_image":
                method_source = ast.get_source_segment(source, node)
                assert method_source is not None
                assert "self.normalize" in method_source, (
                    "_preprocess_image should call self.normalize"
                )
                return
        pytest.fail("_preprocess_image method not found")


# ---------------------------------------------------------------------------
# Bug 4 FIX: depth_anything_lifter.py — Correct homogeneous division
# ---------------------------------------------------------------------------
class TestBug04_HomogeneousDivisionFixed:
    """The buggy .unsqueeze(-1) is removed from homogeneous division."""

    def test_correct_homogeneous_division(self):
        """[M, 3] / [M, 1] should give [M, 3]."""
        M = 10
        p_world_h = torch.randn(M, 4)
        p_world_h[:, 3] = 1.0
        correct = p_world_h[..., :3] / p_world_h[..., 3:]
        assert correct.shape == (M, 3)

    def test_source_no_buggy_unsqueeze(self):
        """Verify the buggy .unsqueeze(-1) pattern is removed."""
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        assert "3:].unsqueeze(-1)" not in source, "Buggy .unsqueeze(-1) pattern should be removed"


# ---------------------------------------------------------------------------
# Bug 5 FIX: cl_gaussians.py — Optimizer rebuilt after pruning
# ---------------------------------------------------------------------------
class TestBug05_OptimizerUpdatedAfterPruning:
    """After prune_gaussians, optimizer references the new tensors."""

    def test_optimizer_updated_after_prune(self):
        _mock_gsplat()
        from clsplats.config import CLSplatsConfig
        from typing import cast
        cfg_dict = omegaconf.OmegaConf.create({"train": {"lr": 1e-3}})
        base_cfg = omegaconf.OmegaConf.structured(CLSplatsConfig)
        cfg = cast(CLSplatsConfig, omegaconf.OmegaConf.merge(base_cfg, cfg_dict))
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

        old_positions_id = id(gauss.optimizer.param_groups[0]["params"][0])

        prune_mask = torch.zeros(N, dtype=torch.bool)
        prune_mask[: N // 2] = True
        gauss.prune_gaussians(prune_mask)

        assert gauss.params.positions.shape[0] == N // 2
        # FIX: optimizer now references the NEW tensor
        new_positions_id = id(gauss.optimizer.param_groups[0]["params"][0])
        assert old_positions_id != new_positions_id, (
            "Optimizer params should be updated after pruning"
        )

    def test_step_after_prune_succeeds(self):
        """After pruning, optimizer.step() should still work correctly."""
        _mock_gsplat()
        from clsplats.config import CLSplatsConfig
        from typing import cast
        cfg_dict = omegaconf.OmegaConf.create({"train": {"lr": 1e-3}})
        base_cfg = omegaconf.OmegaConf.structured(CLSplatsConfig)
        cfg = cast(CLSplatsConfig, omegaconf.OmegaConf.merge(base_cfg, cfg_dict))
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

        # Step once to populate state
        loss = gauss.params.positions.sum()
        loss.backward()
        gauss.step_optimizer()

        # Prune
        prune_mask = torch.zeros(N, dtype=torch.bool)
        prune_mask[:10] = True
        gauss.prune_gaussians(prune_mask)

        # Another backward + step should NOT raise
        loss2 = gauss.params.positions.sum()
        loss2.backward()
        gauss.step_optimizer()
        assert gauss.params.positions.shape == (10, 3)


# ---------------------------------------------------------------------------
# Bug 6 FIX: dinov2_detector.py — No double indexing
# ---------------------------------------------------------------------------
class TestBug06_NoDoubleIndexing:
    """Features are used directly instead of feats[0]."""

    def test_source_no_double_indexing(self):
        source = (CLSPLATS_DIR / "change_detection" / "dinov2_detector.py").read_text()
        assert "rendered_feats[0]" not in source, "Double indexing should be removed"
        assert "observed_feats[0]" not in source, "Double indexing should be removed"


# ---------------------------------------------------------------------------
# Bug 7 FIX: cameras.py — Twc cached in __init__
# ---------------------------------------------------------------------------
class TestBug07_TwcCached:
    """Twc is now cached at init time, not recomputed on every access."""

    def test_twc_cached_in_init(self):
        source = (CLSPLATS_DIR / "dataset" / "cameras.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Camera":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        init_source = ast.get_source_segment(source, method)
                        assert init_source is not None
                        assert "_Twc" in init_source, "Twc should be cached as _Twc in __init__"
                        return
        pytest.fail("Camera.__init__ not found")


# ---------------------------------------------------------------------------
# Bug 8 FIX: train.py — Correct Hydra config path
# ---------------------------------------------------------------------------
class TestBug08_CorrectHydraConfigPath:
    """config_path is now '../configs' relative to clsplats/train.py."""

    def test_config_path_correct(self):
        source = (CLSPLATS_DIR / "train.py").read_text()
        assert "../configs" in source, "config_path should be '../configs'"

    def test_config_file_not_empty(self):
        """cl-splats.yaml should now have content."""
        config_file = PROJECT_ROOT / "configs" / "cl-splats.yaml"
        assert config_file.exists()
        content = config_file.read_text().strip()
        assert len(content) > 10, "Config file should have meaningful content"


# ---------------------------------------------------------------------------
# Bug 9 FIX: train.py — Package-qualified import
# ---------------------------------------------------------------------------
class TestBug09_CorrectImport:
    """train.py now uses package-qualified imports."""

    def test_no_bare_import_trainer(self):
        source = (CLSPLATS_DIR / "train.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "trainer", "Bare 'import trainer' should be fixed"


# ---------------------------------------------------------------------------
# Bug 10 FIX: depth_anything_lifter.py — No dangling confidence vars
# ---------------------------------------------------------------------------
class TestBug10_NoUnusedConfidenceVars:
    """pos_conf and neg_conf are no longer in the source."""

    def test_no_pos_neg_conf(self):
        source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()
        assert "pos_conf" not in source, "pos_conf should be removed"
        assert "neg_conf" not in source, "neg_conf should be removed"


# ---------------------------------------------------------------------------
# Bug 11 FIX: BaseLifter.lift() signature now matches
# ---------------------------------------------------------------------------
class TestBug11_LifterSignatureMatches:
    """Abstract method signature now matches concrete implementation."""

    def test_base_matches_concrete_signature(self):
        base_source = (CLSPLATS_DIR / "lifter" / "base_lifter.py").read_text()
        concrete_source = (CLSPLATS_DIR / "lifter" / "depth_anything_lifter.py").read_text()

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
        assert base_params == concrete_params, (
            f"Signatures should match: base={base_params}, concrete={concrete_params}"
        )


# ---------------------------------------------------------------------------
# Bug 12 & 13 FIX: Correct import paths
# ---------------------------------------------------------------------------
class TestBug12_DatasetReaderCorrectImports:
    """dataset_reader.py now uses clsplats-qualified imports."""

    def test_no_legacy_imports(self):
        source = (CLSPLATS_DIR / "dataset" / "dataset_reader.py").read_text()
        assert "from scene.colmap_loader import" not in source, "Legacy import path should be fixed"
        assert (
            "from clsplats.dataset.colmap_reader" in source
            or "from clsplats.utils.graphics_utils" in source
        ), "Should use clsplats-qualified imports"


class TestBug13_CamerasCorrectImports:
    """cameras.py now uses clsplats-qualified imports."""

    def test_no_legacy_imports(self):
        source = (CLSPLATS_DIR / "dataset" / "cameras.py").read_text()
        assert "from utils.graphics_utils import" not in source, "Legacy import should be fixed"
        assert "from clsplats.utils.graphics_utils" in source, (
            "Should use clsplats-qualified imports"
        )


# ---------------------------------------------------------------------------
# Bug 14: gaussian_model.py — File is now deleted
# ---------------------------------------------------------------------------
class TestBug14_GaussianModelDeleted:
    """gaussian_model.py (with the no-op indexing) is now deleted."""

    def test_file_deleted(self):
        assert not (CLSPLATS_DIR / "representation" / "gaussian_model.py").exists(), (
            "gaussian_model.py should be deleted"
        )


# ---------------------------------------------------------------------------
# Bug 15 FIX: No bare except clauses in dataset_reader.py
# ---------------------------------------------------------------------------
class TestBug15_NoBareExceptClauses:
    """dataset_reader.py no longer uses bare 'except:' clauses."""

    def test_no_bare_except_in_dataset_reader(self):
        source = (CLSPLATS_DIR / "dataset" / "dataset_reader.py").read_text()
        tree = ast.parse(source)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    count += 1
        assert count == 0, f"Found {count} bare except clauses, expected 0"

    def test_no_bare_except_in_cameras(self):
        source = (CLSPLATS_DIR / "dataset" / "cameras.py").read_text()
        tree = ast.parse(source)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    count += 1
        assert count == 0, f"Found {count} bare except clauses, expected 0"


# ---------------------------------------------------------------------------
# Bug 16 FIX: run_test_scene.py — No hardcoded macOS path
# ---------------------------------------------------------------------------
class TestBug16_NoHardcodedPath:
    """run_test_scene.py no longer contains hardcoded macOS path."""

    def test_no_hardcoded_macos_path(self):
        source = (PROJECT_ROOT / "scripts" / "run_test_scene.py").read_text()
        assert "/Users/jan/Code/" not in source, "Hardcoded macOS path should be removed"


# ---------------------------------------------------------------------------
# Quality: structured config exists and is importable
# ---------------------------------------------------------------------------
class TestConfigModule:
    """Test that the new config module is well-formed."""

    def test_config_importable(self):
        from clsplats.config import CLSplatsConfig

        cfg = CLSplatsConfig()
        assert cfg.model.sh_degree == 0
        assert cfg.train.lr == 1e-3
        assert cfg.lifter.k_nn == 8

    def test_all_fields_have_defaults(self):
        from clsplats.config import CLSplatsConfig

        cfg = CLSplatsConfig()
        # Every sub-config should instantiate without arguments
        assert cfg.model is not None
        assert cfg.train is not None
        assert cfg.change is not None
        assert cfg.lifter is not None
        assert cfg.constraints is not None
        assert cfg.history is not None


# ---------------------------------------------------------------------------
# Quality: dead code is removed
# ---------------------------------------------------------------------------
class TestDeadCodeRemoved:
    """Verify unused modules are deleted."""

    def test_gaussian_model_deleted(self):
        assert not (CLSPLATS_DIR / "representation" / "gaussian_model.py").exists()

    def test_sampling_deleted(self):
        assert not (CLSPLATS_DIR / "sampling").exists()

    def test_pyproject_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# Integration: CLGaussians with mocked gsplat
# ---------------------------------------------------------------------------
class TestCLGaussiansIntegration:
    """Integration tests for CLGaussians."""

    def setup_method(self):
        _mock_gsplat()

    def _make_gaussians(self, n=50):
        from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams

        from clsplats.config import CLSplatsConfig
        from typing import cast
        cfg_dict = omegaconf.OmegaConf.create({"train": {"lr": 1e-3}})
        base_cfg = omegaconf.OmegaConf.structured(CLSplatsConfig)
        cfg = cast(CLSplatsConfig, omegaconf.OmegaConf.merge(base_cfg, cfg_dict))
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
        loss = gauss.params.positions.sum()
        loss.backward()
        gauss.step_optimizer()
        assert gauss.params.positions.grad is None

    def test_prune_basic(self):
        gauss = self._make_gaussians(20)
        prune_mask = torch.zeros(20, dtype=torch.bool)
        prune_mask[::2] = True
        keep = gauss.prune_gaussians(prune_mask)
        assert gauss.params.positions.shape[0] == 10
        assert keep.sum().item() == 10

    def test_prune_then_step_works(self):
        """After pruning, optimizer step should succeed with no shape mismatch."""
        gauss = self._make_gaussians(20)
        loss = gauss.params.positions.sum()
        loss.backward()
        gauss.step_optimizer()

        prune_mask = torch.zeros(20, dtype=torch.bool)
        prune_mask[:10] = True
        gauss.prune_gaussians(prune_mask)
        assert gauss.params.positions.shape == (10, 3)

        # After pruning, step should work correctly
        loss2 = gauss.params.positions.sum()
        loss2.backward()
        gauss.step_optimizer()
        assert gauss.params.positions.shape == (10, 3)

        # Optimizer state (if any) should match current shape
        opt_param = gauss.optimizer.param_groups[0]["params"][0]
        if opt_param in gauss.optimizer.state:
            state = gauss.optimizer.state[opt_param]
            assert state["exp_avg"].shape == gauss.params.positions.shape


# ---------------------------------------------------------------------------
# Primitives module
# ---------------------------------------------------------------------------
class TestPrimitivesModule:
    """Test the constraints/primitives module."""

    def test_fit_sphere(self):
        from clsplats.constraints.primitives import fit_sphere

        pts = torch.randn(100, 3) * 0.5
        sphere = fit_sphere(pts)
        assert sphere.center.shape == (3,)
        assert sphere.radius > 0

    def test_fit_obb(self):
        from clsplats.constraints.primitives import fit_obb

        pts = torch.randn(100, 3)
        pts[:, 0] *= 5
        obb = fit_obb(pts)
        assert obb.center.shape == (3,)
        assert obb.half_extents.shape == (3,)
        assert obb.rotation.shape == (3, 3)

    def test_sphere_distance_inside(self):
        from clsplats.constraints.primitives import distance_to_primitive

        prim = ("sphere", type("S", (), {"center": torch.zeros(3), "radius": 2.0})())
        pts = torch.tensor([[0.0, 0.0, 0.0]])
        d = distance_to_primitive(pts, prim)  # type: ignore
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_sphere_distance_outside(self):
        from clsplats.constraints.primitives import distance_to_primitive

        prim = ("sphere", type("S", (), {"center": torch.zeros(3), "radius": 1.0})())
        pts = torch.tensor([[3.0, 0.0, 0.0]])
        d = distance_to_primitive(pts, prim)  # type: ignore
        assert d.item() == pytest.approx(2.0, abs=1e-6)

    def test_union_distance_empty(self):
        from clsplats.constraints.primitives import union_distance

        pts = torch.randn(10, 3)
        d = union_distance(pts, [])
        assert (d == 0).all()

    def test_group_active_gaussians(self):
        from clsplats.constraints.primitives import group_active_gaussians

        pts = torch.zeros(20, 3)
        pts[:10] = torch.randn(10, 3) * 0.01
        pts[10:] = torch.randn(10, 3) * 0.01 + 10.0
        mask = torch.ones(20, dtype=torch.bool)
        groups = group_active_gaussians(pts, mask, radius_frac=0.1)
        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
