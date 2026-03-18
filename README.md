# CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization

###  ICCV 2025
[![Website](https://img.shields.io/badge/CL--Splats-%F0%9F%8C%90Website-purple?style=flat)](https://cl-splats.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.21117-b31b1b.svg)](https://arxiv.org/abs/2506.21117)


[Jan Ackermann](https://janackermann.info)
[Jonas Kulhanek](https://jkulhanek.com)
[Shengqu Cai](https://primecai.github.io)
[Haofei Xu](https://haofeixu.github.io)
[Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/)
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)
[Songyou Peng](https://pengsongyou.github.io)

![CL-Splats Teaser Graphic](assets/cl-splats-teaser.png)

*TL;DR*: CL-Splats optimizes existing 3DGS scene representations with a small set of images showing the changed region.

## Contents
<!--ts-->
   * [Install](#install)
   * [Dataset Format](#dataset-format)
   * [Usage](#usage)
   * [Configuration](#configuration)
   * [Todos](#todos)
   * [Citation](#citation)
<!--te-->

## Install

### Pre-requisites
While not strictly necessary for using our method, COLMAP is necessary to obtain camera poses for the initial reconstruction as well as to add new observations to existing models.
Please follow the instructions on the COLMAP website to install COLMAP. If possible install it with CUDA support.

### Environment

We tested our code on Ubuntu 24.04 with CUDA 12.8. Install via pip (we recommend a conda/venv environment):

```bash
# Create and activate environment
conda create -n cl-splats python=3.10
conda activate cl-splats

# Install package and all dependencies
pip install -e .
```

> **Note:**  
> Make sure that your installed PyTorch (`torch`) version is compiled with the **same CUDA version** as the one you use to compile the custom CUDA kernels in this project.  
> Additionally, you must have the **CUDA Development Kit** installed to provide access to required CUDA libraries.


## Dataset Format

CL-Splats supports two dataset formats, auto-detected at runtime.

### COLMAP (real-world scenes)

Standard COLMAP workspace layout — what `preprocessing.py` produces:

```text
path/to/dataset/
├── images/           # undistorted images
│   ├── frame_00001.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Run with:
```bash
cl-splats-train --data-path path/to/dataset
```

#### Computing Poses

For your convenience, we provide a preprocessing script that runs COLMAP automatically. It assumes raw images are organised as timestep folders:

```text
path/to/your/input/
├── t0/   # images for timestep 0 (base scene)
│   ├── *.{png,jpeg,jpg}
│   └── ...
├── t1/   # images for timestep 1 (after changes)
│   └── ...
└── ...
```

Run:
```bash
python3 clsplats/utils/preprocessing.py --input_dir <path/to/your/input>
```

> **Note:**  
> Our codebase currently only supports NeRF-Synthetic and COLMAP pose formats, and their naming scheme must be consistent with the output of our preprocessing script.

---

### Blender / NeRF-Synthetic (CL benchmark scenes)

Used for the synthetic continual-learning benchmark dataset. Each timestep lives in its own subdirectory with a `transforms_train.json` file:

```text
path/to/Level-1/
├── transforms_train.json   # base scene (t0)
├── images/
│   └── ...
└── add/                    # change subfolder named after the change type
    ├── transforms_train.json
    └── images/
        └── ...
```

Available change types: `add`, `delete`, `move`, `multi`.

Run with:
```bash
cl-splats-train --data-path path/to/Level-1 --change-type add
```


## Usage

### Basic Training

```bash
# Real-world COLMAP scene (single timestep)
cl-splats-train --data-path path/to/dataset

# Blender CL scene (base + one change timestep)
cl-splats-train --data-path path/to/Level-1 --change-type add
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--data-path` / `-d` | `.` | Path to the dataset root directory |
| `--change-type` / `-c` | `None` | Change type for Blender CL datasets (`add`, `delete`, `move`, `multi`). Omit for COLMAP/single-timestep scenes. |
| `--images` | `images` | Name of the images subdirectory |
| `--depths` | `""` | Name of the depths subdirectory (optional) |
| `--eval` | `False` | Evaluate on a held-out test split after training |
| `--offline` / `--no-offline` | `False` | Disable all network access — sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and forces W&B offline. Use when running without internet (e.g. air-gapped servers). |
| `--config-name` | `cl-splats` | Hydra config file to load from `configs/` (without `.yaml`) |

Run `cl-splats-train --help` to see all options.

### Overriding Config Values (Hydra)

Any configuration value can be overridden directly on the command line as positional arguments using Hydra dot-notation:

```bash
# More iterations, lower learning rate
cl-splats-train --data-path path/to/dataset \
    train.iters_per_timestep=500 \
    train.lr=5e-4

# Tighter change-detection threshold with dilation
cl-splats-train --data-path path/to/Level-1 --change-type add \
    change.threshold=0.7 \
    change.dilate_mask=true \
    change.dilate_kernel_size=15

# Enable live W&B syncing
cl-splats-train --data-path path/to/dataset wandb_mode=online
```

### Output

Results are saved to `outputs/<run-name>/` by default and include:

- `gaussians_t<N>.ply` — Gaussian splat checkpoint at each trained timestep,  
  viewable in [SuperSplat](https://superspl.at/editor) or Luma AI
- W&B run logs (synced live, or stored offline for later upload)


## Configuration

The default config lives in `configs/cl-splats.yaml`. A summary of the most useful keys:

### `train`
| Key | Default | Description |
|---|---|---|
| `lr` | `1e-3` | Learning rate |
| `iters_per_timestep` | `100` | Optimisation iterations per timestep |
| `num_times` | `1` | Number of timesteps (auto-set to `2` for Blender + `--change-type`) |
| `start_time` | `0` | First timestep index to optimise |

### `change` — DINOv2 change detection
| Key | Default | Description |
|---|---|---|
| `threshold` | `0.8` | Cosine-similarity threshold; lower = more sensitive |
| `dilate_mask` | `false` | Morphologically dilate the binary change mask |
| `dilate_kernel_size` | `31` | Dilation kernel size (pixels) |
| `upsample` | `true` | Upsample mask back to full image resolution |

### `lifter` — Depth-Anything V2 3D lifting
| Key | Default | Description |
|---|---|---|
| `depth_model` | `depth-anything/Depth-Anything-V2-Small-hf` | HuggingFace model ID |
| `k_nn` | `8` | Nearest Gaussian neighbours per back-projected pixel |
| `local_radius_thresh` | `2.5` | Max scale-normalised distance for a kNN match |
| `depth_tol_abs` | `0.05` | Absolute depth consistency tolerance (scene units) |
| `depth_tol_rel` | `0.05` | Relative depth consistency tolerance |
| `final_thresh` | `0.6` | Minimum score to activate a Gaussian for optimisation |

### `model` — Gaussian representation
| Key | Default | Description |
|---|---|---|
| `sh_degree` | `0` | Spherical harmonics degree (0 = colour only) |
| `init_scale` | `0.01` | Initial Gaussian scale |
| `init_opacity` | `0.5` | Initial Gaussian opacity |

### `constraints` — Geometry constraints
| Key | Default | Description |
|---|---|---|
| `prune_every` | `50` | Prune dead Gaussians every N iterations |
| `prune_dist_thresh` | `0.02` | Distance threshold for pruning |
| `lambda_bound` | `0.0` | Bounding-box constraint weight |


## Todos
I continue to release the missing modules required to replicate our method.

- [x] Release initial codebase with framework skeleton.
- [x] Release camera estimation script.
- [x] Release fast change detection module.
- [x] Release sampling module.
- [x] Release pruning module.
- [x] Release data.
- [x] ~~Release local-optimization CUDA kernels.~~
- [ ] Verify codebase.
- [ ] Release history recovery.

## Citation
```
@inproceedings{ackermann2025clsplats,
    author={Ackermann, Jan and Kulhanek, Jonas and Cai, Shengqu and Haofei, Xu and Pollefeys, Marc and Wetzstein, Gordon and Guibas, Leonidas and Peng, Songyou},
    title={CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```
