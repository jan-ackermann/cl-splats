"""CL-Splats entry point.

Launches the CL-Splats training pipeline using Hydra for configuration
and Weights & Biases for experiment tracking.
"""

from typing import Optional

import hydra
import omegaconf
import typer
import wandb
from hydra.core.global_hydra import GlobalHydra
from loguru import logger

from clsplats.config import CLSplatsConfig
from clsplats.dataset.dataset_reader import readColmapSceneInfo
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


@app.command()
def main(
    data_path: str = typer.Option(
        ".", "--data-path", "-d", help="Path to the dataset directory (e.g., COLMAP workspace)."
    ),
    images: str = typer.Option("images", help="Name of the images subdirectory."),
    depths: str = typer.Option("", help="Name of the depths subdirectory (optional)."),
    eval_split: bool = typer.Option(False, "--eval", help="Whether to evaluate on a test split."),
    config_name: str = typer.Option(
        "cl-splats", help="Name of the Hydra configuration file to use."
    ),
    overrides: Optional[list[str]] = typer.Argument(
        None, help="Additional Hydra configuration overrides (e.g., train.num_times=100)."
    ),
) -> None:
    """Launch the CL-Splats training pipeline."""
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

        if overrides:
            cfg_overrides.extend(overrides)

        cfg_dict = hydra.compose(config_name=config_name, overrides=cfg_overrides)

        # Merge incoming config with structured config to maintain type safety
        base_cfg = omegaconf.OmegaConf.structured(CLSplatsConfig)
        cfg_merged = omegaconf.OmegaConf.merge(base_cfg, cfg_dict)
        
        from typing import cast
        cfg = cast(CLSplatsConfig, cfg_merged)

    setup_wandb(cfg)

    # Load scene — extend this section when adding more dataset formats.
    scene = readColmapSceneInfo(
        path=cfg.data_path,
        images=cfg.images,
        depths=cfg.depths,
        eval=cfg.eval,
        train_test_exp=cfg.train_test_exp,
    )

    trainer = CLSplatsTrainer(cfg, scene)

    for time in range(cfg.train.start_time, cfg.train.num_times):
        logger.info("Optimizing observations at time {time}.", time=time)
        trainer.prepare_timestep(time)
        trainer.train()

        if cfg.history.log_history:
            trainer.log_history()


if __name__ == "__main__":
    app()
