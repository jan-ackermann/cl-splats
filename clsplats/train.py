"""CL-Splats entry point.

Launches the CL-Splats training pipeline using Hydra for configuration
and Weights & Biases for experiment tracking.
"""

import os
import sys
from typing import Optional, cast

import hydra
import omegaconf
import typer
import wandb
from hydra.core.global_hydra import GlobalHydra
from loguru import logger

# Loguru defaults to stderr; redirect to stdout so XCloud doesn't flag
# every INFO log as an error.
logger.remove()
logger.add(sys.stdout, colorize=False)

from clsplats.config import CLSplatsConfig
from clsplats.dataset.dataset_reader import readColmapSceneInfo, readNerfSyntheticInfo
from clsplats.trainer import CLSplatsTrainer

app = typer.Typer(
    help="CL-Splats: Continual Learning with 3D Gaussian Splatting",
    pretty_exceptions_show_locals=False,
)


def setup_wandb(cfg: "CLSplatsConfig") -> None:
    """Initialise a Weights & Biases run from *cfg*."""
    if not wandb.run and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            mode=cfg.wandb_mode,  # type: ignore
        )


def _detect_format(data_path: str) -> str:
    """Return ``'Blender'`` or ``'Colmap'`` based on files present in *data_path*."""
    if os.path.exists(os.path.join(data_path, "transforms_train.json")):
        return "Blender"
    if os.path.exists(os.path.join(data_path, "sparse")):
        return "Colmap"
    raise FileNotFoundError(
        f"Could not detect dataset format at '{data_path}'. "
        "Expected either 'transforms_train.json' (NeRF-Synthetic) or 'sparse/' (COLMAP)."
    )


@app.command()
def main(
    data_path: str = typer.Option(
        ".", "--data-path", "-d", help="Path to the dataset directory (e.g., COLMAP workspace)."
    ),
    images: str = typer.Option("images", help="Name of the images subdirectory."),
    depths: str = typer.Option("", help="Name of the depths subdirectory (optional)."),
    eval_split: bool = typer.Option(False, "--eval", help="Whether to evaluate on a test split."),
    change_type: Optional[str] = typer.Option(
        None,
        "--change-type",
        "-c",
        help="Change type for Blender CL datasets (e.g., add, delete, move, multi).",
    ),
    offline: bool = typer.Option(
        False,
        "--offline/--no-offline",
        help=(
            "Run in fully offline mode (no internet). "
            "Sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, and forces wandb offline. "
            "Use --offline on XCloud; omit (or --no-offline) for local dev."
        ),
    ),
    config_name: str = typer.Option(
        "cl-splats", help="Name of the Hydra configuration file to use."
    ),
    overrides: list[str] = typer.Argument(
        default=None, help="Additional Hydra configuration overrides (e.g., train.num_times=100)."
    ),
) -> None:
    """Launch the CL-Splats training pipeline."""
    if offline:
        # Must be set before any HuggingFace / torch.hub imports resolve.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", "/workdir/hf_cache")
        os.environ.setdefault("TORCH_HOME", "/workdir/torch_cache")
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.info("Offline mode enabled — using cached models only.")
    else:
        logger.info("Online mode — models and W&B will use network as needed.")
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg_overrides = []
        if data_path != ".":
            cfg_overrides.append(f"data_path={data_path}")
        if images != "images":
            cfg_overrides.append(f"images={images}")
        if depths != "":
            cfg_overrides.append(f"depths={depths}")
        if eval_split:
            cfg_overrides.append("eval=True")

        if isinstance(overrides, list) and overrides:
            cfg_overrides.extend(overrides)

        cfg_dict = hydra.compose(config_name=config_name, overrides=cfg_overrides)

        # Merge incoming config with structured config to maintain type safety
        base_cfg = omegaconf.OmegaConf.structured(CLSplatsConfig)
        cfg_merged = omegaconf.OmegaConf.merge(base_cfg, cfg_dict)

        cfg = cast(CLSplatsConfig, cfg_merged)

    setup_wandb(cfg)

    # Auto-detect dataset format
    fmt = _detect_format(cfg.data_path)
    logger.info("Detected dataset format: {fmt}", fmt=fmt)

    if fmt == "Blender":
        scene = readNerfSyntheticInfo(
            path=cfg.data_path,
            white_background=False,
            depths=cfg.depths,
            eval=cfg.eval,
        )
    else:
        scene = readColmapSceneInfo(
            path=cfg.data_path,
            images=cfg.images,
            depths=cfg.depths,
            eval=cfg.eval,
            train_test_exp=cfg.train_test_exp,
        )

    trainer = CLSplatsTrainer(cfg, scene)

    # Auto-detect number of timesteps when the user hasn't explicitly set it.
    num_times = cfg.train.num_times
    if fmt == "Colmap":
        detected = len(set(c.timestep for c in scene.train_cameras))
        if detected > 1 and num_times == 1:
            num_times = detected
            logger.info(
                "Auto-detected {n} timesteps from image names.", n=num_times
            )
    elif fmt == "Blender" and change_type and num_times == 1:
        num_times = 2
        logger.info(
            "Set num_times=2 for Blender change type '{ct}'.", ct=change_type
        )

    # Keep the config in sync so trainer.prepare_timestep's assert passes.
    omegaconf.OmegaConf.update(cfg, "train.num_times", num_times, merge=False)

    # Track the last loaded scene so we can pass its test cameras to evaluate().
    active_scene = scene

    for time in range(cfg.train.start_time, num_times):
        # For Blender temporal data: load the change-scene at t1
        if fmt == "Blender" and time > cfg.train.start_time and change_type:
            change_path = os.path.join(cfg.data_path, change_type)
            if not os.path.isdir(change_path):
                logger.error(
                    "Change directory '{path}' not found.", path=change_path
                )
                raise SystemExit(1)
            change_scene = readNerfSyntheticInfo(
                path=change_path,
                white_background=False,
                depths=cfg.depths,
                eval=cfg.eval,
            )
            trainer.update_cameras(change_scene, time)
            active_scene = change_scene

        logger.info("Optimizing observations at time {time}.", time=time)
        trainer.prepare_timestep(time)
        trainer.train()

        if cfg.eval and active_scene.test_cameras:
            trainer.evaluate(
                test_cameras=active_scene.test_cameras,
                timestep=time,
            )

        if cfg.history.log_history:
            trainer.log_history()


if __name__ == "__main__":
    app()
