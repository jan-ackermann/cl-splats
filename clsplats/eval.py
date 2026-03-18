"""Standalone evaluation script for CL-Splats.

Loads a saved ``.ply`` Gaussian checkpoint and runs the evaluation loop
(PSNR / SSIM / rendered images) against a dataset's test split without
any training.

Usage::

    cl-splats-eval \\
        --ply path/to/gaussians_time_0001.ply \\
        --data-path path/to/dataset \\
        [--change-type add] \\
        [--timestep 1] \\
        [--out-dir outputs/eval] \\
        [--images images]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from loguru import logger

from clsplats.config import CLSplatsConfig
from clsplats.dataset.dataset_reader import (
    SceneInfo,
    readColmapSceneInfo,
    readNerfSyntheticInfo,
)
from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams
from clsplats.trainer import CLSplatsTrainer

app = typer.Typer(pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# PLY loader
# ---------------------------------------------------------------------------

def _load_gaussians_from_ply(ply_path: str, cfg: CLSplatsConfig) -> CLGaussians:
    """Reconstruct a :class:`CLGaussians` object from a saved ``.ply`` file.

    The file layout matches what :meth:`CLGaussians.export_ply` writes via
    ``gsplat.exporter.export_splats``.
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    # Positions
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # Scales (stored as log-scale in gsplat export)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    scales = np.exp(scales)  # undo log

    # Rotations (quaternion, stored as rot_0..rot_3)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)

    # Opacity (stored as logit in gsplat export)
    opacity_raw = v["opacity"].astype(np.float32).reshape(-1, 1)
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid

    # SH features: DC term is f_dc_0..f_dc_2, rest are f_rest_0..f_rest_N
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)  # (N, 3)

    # Count higher-order coefficients
    rest_cols = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")])
    if rest_cols:
        rest = np.stack([v[c] for c in rest_cols], axis=-1).astype(np.float32)  # (N, 3*(K-1))
        n_rest = rest.shape[1] // 3
        rest = rest.reshape(-1, n_rest, 3)  # (N, K-1, 3)
        # sh_features layout in CLGaussians: (N, K, 3)
        sh_features = np.concatenate([dc[:, None, :], rest], axis=1)
    else:
        sh_features = dc[:, None, :]  # (N, 1, 3) — DC only

    params = GaussianParams(
        positions=torch.from_numpy(xyz),
        scales=torch.from_numpy(scales),
        quats=torch.from_numpy(quats),
        sh_features=torch.from_numpy(sh_features),
        opacity=torch.from_numpy(opacity),
    )
    return CLGaussians(cfg, params)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _detect_format(data_path: str) -> str:
    """Return ``'Blender'`` or ``'Colmap'`` based on directory contents."""
    p = Path(data_path)
    if (p / "transforms_train.json").exists():
        return "Blender"
    if (p / "sparse").exists():
        return "Colmap"
    raise ValueError(
        f"Cannot detect dataset format at '{data_path}'. "
        "Expected 'transforms_train.json' (Blender) or 'sparse/' (COLMAP)."
    )


def _load_scene(
    data_path: str,
    images: str,
    change_type: Optional[str],
    timestep: int,
    eval_: bool,
) -> SceneInfo:
    """Load the correct scene for the requested *timestep*."""
    fmt = _detect_format(data_path)

    if fmt == "Blender":
        if timestep > 0 and change_type:
            change_path = str(Path(data_path) / change_type)
            if not Path(change_path).is_dir():
                logger.error("Change directory '{}' not found.", change_path)
                raise SystemExit(1)
            return readNerfSyntheticInfo(
                path=change_path,
                white_background=False,
                depths="",
                eval=eval_,
            )
        return readNerfSyntheticInfo(
            path=data_path,
            white_background=False,
            depths="",
            eval=eval_,
        )
    else:
        return readColmapSceneInfo(
            path=data_path,
            images=images,
            depths="",
            eval=eval_,
            train_test_exp=False,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    ply: str = typer.Option(..., "--ply", "-p", help="Path to the saved .ply Gaussian checkpoint."),
    data_path: str = typer.Option(..., "--data-path", "-d", help="Path to the dataset root."),
    change_type: Optional[str] = typer.Option(
        None,
        "--change-type",
        "-c",
        help="Change type (add/delete/move) for Blender CL datasets. Omit for COLMAP or t=0.",
    ),
    timestep: int = typer.Option(
        0,
        "--timestep",
        "-t",
        help="Timestep index the checkpoint corresponds to (used for naming outputs).",
    ),
    out_dir: str = typer.Option(
        "outputs/eval",
        "--out-dir",
        "-o",
        help="Directory to save rendered images and metrics.",
    ),
    images: str = typer.Option("images", "--images", help="Images sub-folder name (COLMAP only)."),
    sh_degree: int = typer.Option(0, "--sh-degree", help="SH degree used during training."),
) -> None:
    """Evaluate a saved CL-Splats checkpoint against a dataset's test split.

    Loads *ply*, renders every test-split camera, and reports PSNR / SSIM.
    Images are saved to *out_dir*.
    """
    ply_path = Path(ply)
    if not ply_path.exists():
        logger.error("PLY file not found: {}", ply)
        raise SystemExit(1)

    # Minimal config — only model.sh_degree matters for rendering
    cfg = CLSplatsConfig()
    cfg.model.sh_degree = sh_degree

    logger.info("Loading Gaussians from {}", ply_path)
    gaussians = _load_gaussians_from_ply(str(ply_path), cfg)
    n = gaussians.params.positions.shape[0]
    logger.info("Loaded {:,} Gaussians.", n)

    # Load dataset (eval split)
    logger.info("Loading dataset from {}", data_path)
    scene = _load_scene(
        data_path=data_path,
        images=images,
        change_type=change_type,
        timestep=timestep,
        eval_=True,
    )

    if not scene.test_cameras:
        logger.error(
            "No test cameras found. Make sure --eval is meaningful for this dataset "
            "(COLMAP needs ≥8 images for an LLFF-style hold-out; Blender needs "
            "transforms_test.json)."
        )
        raise SystemExit(1)

    logger.info("Evaluating on {} test views…", len(scene.test_cameras))

    # Build a minimal trainer with the loaded Gaussians (no detectors needed)
    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.timestep = timestep
    trainer.gaussians = gaussians

    metrics = trainer.evaluate(
        test_cameras=scene.test_cameras,
        timestep=timestep,
        out_dir=out_dir,
    )

    print(
        f"\n{'='*50}\n"
        f"  Timestep {timestep}  |  {len(scene.test_cameras)} views\n"
        f"  PSNR : {metrics['psnr']:.2f} dB\n"
        f"  SSIM : {metrics['ssim']:.4f}\n"
        f"  Renders saved to: {Path(out_dir) / f't{timestep:04d}'}\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    app()
