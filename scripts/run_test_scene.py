"""Manual test script for the CL-Splats pipeline.

Usage:
    python scripts/run_test_scene.py
"""

import os
import sys

# Ensure project root is on sys.path when executing as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clsplats.config import CLSplatsConfig  # noqa: E402
from clsplats.dataset.dataset_reader import readColmapSceneInfo  # noqa: E402
from clsplats.trainer import CLSplatsTrainer  # noqa: E402


def main():
    workspace_root = "test_data/colmap_input/colmap_workspace/undistorted"
    if not os.path.isdir(workspace_root):
        raise RuntimeError(f"Undistorted COLMAP workspace not found at {workspace_root}")

    scene = readColmapSceneInfo(
        path=workspace_root,
        images="images",
        eval=False,
        train_test_exp=False,
    )

    cfg = CLSplatsConfig()
    cfg.train.iters_per_timestep = 10
    cfg.train.log_interval = 1

    trainer = CLSplatsTrainer(cfg, scene)

    print("Starting test training run...")
    trainer.prepare_timestep(0)
    trainer.train()
    print("Finished test training run.")


if __name__ == "__main__":
    main()
