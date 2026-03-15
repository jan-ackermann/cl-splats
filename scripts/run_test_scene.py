import os
import sys

from omegaconf import OmegaConf

# Ensure project root is on sys.path when executing as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
GSPLAT_ROOT = "/Users/jan/Code/gaussian-splatting"
for p in (PROJECT_ROOT, GSPLAT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from clsplats.dataset.dataset_reader import readColmapSceneInfo
from clsplats.trainer import CLSplatsTrainer


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
        },
    }
    return OmegaConf.create(cfg_dict)


def main():
    # Use the undistorted COLMAP workspace root; our loader now supports
    # both "sparse/0" and "sparse" layouts, and the undistorted model
    # has PINHOLE/SIMPLE_PINHOLE cameras as required downstream.
    workspace_root = "test_data/colmap_input/colmap_workspace/undistorted"
    if not os.path.isdir(workspace_root):
        raise RuntimeError(f"Undistorted COLMAP workspace not found at {workspace_root}")

    # Load scene info from COLMAP outputs
    scene = readColmapSceneInfo(
        path=workspace_root,
        images="images",
        depths="",
        eval=False,
        train_test_exp=False,
    )

    cfg = build_test_cfg()
    trainer = CLSplatsTrainer(cfg, scene)

    print("Starting test training run...")
    trainer.prepare_timestep(0)
    trainer.train()
    print("Finished test training run.")


if __name__ == "__main__":
    main()

