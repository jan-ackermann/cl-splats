"""Manual test script for the CL-Splats pipeline.

Usage:
    python scripts/run_test_scene.py
"""

import os
import sys

from omegaconf import OmegaConf

# Ensure project root is on sys.path when executing as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Bug 16 fix: use env var or project root instead of hardcoded macOS path
GSPLAT_ROOT = os.environ.get("GSPLAT_ROOT", "")
for p in (PROJECT_ROOT, GSPLAT_ROOT):
    if p and p not in sys.path:
        sys.path.insert(0, p)

from clsplats.dataset.dataset_reader import readColmapSceneInfo  # noqa: E402
from clsplats.trainer import CLSplatsTrainer  # noqa: E402


def build_test_cfg():
    cfg_dict = {
        "model": {
            "sh_degree": 0,
            "init_scale": 0.01,
            "init_opacity": 0.5,
        },
        "train": {
            "lr": 1e-3,
            "iters_per_timestep": 10,
            "log_interval": 1,
            "num_times": 1,
            "start_time": 0,
        },
        "change": {
            "threshold": 0.8,
            "dilate_mask": False,
            "upsample": True,
        },
        "history": {
            "log_history": False,
        },
        "lifter": {
            "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
            "k_nn": 8,
            "local_radius_thresh": 2.5,
            "depth_tol_abs": 0.05,
            "depth_tol_rel": 0.05,
            "lambda_seed": 2.0,
            "lambda_neg": 0.25,
            "min_visible_views": 2,
            "min_positive_views": 2,
            "min_seed_views": 1,
            "min_positive_ratio": 0.3,
            "final_thresh": 0.6,
        },
        "constraints": {
            "group_radius_frac": 0.1,
            "lambda_bound": 0.0,
            "prune_every": 50,
            "prune_dist_thresh": 0.02,
            "prune_consecutive": 3,
        },
    }
    return OmegaConf.create(cfg_dict)


def main():
    workspace_root = "test_data/colmap_input/colmap_workspace/undistorted"
    if not os.path.isdir(workspace_root):
        raise RuntimeError(f"Undistorted COLMAP workspace not found at {workspace_root}")

    scene = readColmapSceneInfo(
        path=workspace_root,
        images="images",
        depths="",
        eval=False,
        train_test_exp=False,
    )

    from clsplats.config import CLSplatsConfig
    from typing import cast
    base_cfg = OmegaConf.structured(CLSplatsConfig)
    cfg_raw = OmegaConf.merge(base_cfg, build_test_cfg())
    cfg = cast(CLSplatsConfig, cfg_raw)
    
    trainer = CLSplatsTrainer(cfg, scene)

    print("Starting test training run...")
    trainer.prepare_timestep(0)
    trainer.train()
    print("Finished test training run.")


if __name__ == "__main__":
    main()
