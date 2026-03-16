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
   * [Usage](#usage)
   * [Todos](#todos)
   * [Citation](#citation)
<!--te-->

## Install

### Pre-requisites
While not strictly necessary for using our method, COLMAP is necessary to obtain camera poses for the initial reconstruction as well as to add new observations to existing models.
Please follow the instructions on the COLMAP website to install COLMAP. If possible install it with CUDA support.
 
### Environment
We tested our code on Ubuntu 24.04 with CUDA 12.8. It is easiest to install the environment using conda/mamba from the environment file `environment.yaml`.

> **Note:**  
> Make sure that your installed PyTorch (`torch`) version is compiled with the **same CUDA version** as the one you use to compile the custom CUDA kernels in this project.  
> Additionally, you must have the **CUDA Development Kit** installed on your system to provide access to required CUDA libraries for compilation and runtime.


## Usage

### Computing Poses
In our work, we describe two different ways to obtain poses, which both use COLMAP. In this repository we will focus on the second incremental way. While it may lead to slightly worse reconstruction performance, it scales better with more timesteps and it is possible to build it incrementally. We also see it as a chance to promote further research improving issues with existing pose-estimation methods for incremental updates under changes.

For your convenience, we provide a python script `clsplats/utils/preprocessing.py` that given data automatically obtains the poses. The functionality of this script is explained in the supplementary material of our paper.
To run the preprocessing script and compute poses, use:
```bash
python3 clsplats/utils/preprocessing.py --input_dir <path/to/your/input>
```

The script assumes the following directory structure:
```text
path/to/your/input/
├── t0/
│   ├── *.{png,jpeg,jpg}
│   ├── *.{png,jpeg,jpg}
│   └── ...
├── t1/
│   ├── *.{png,jpeg,jpg}
│   ├── *.{png,jpeg,jpg}
│   └── ...
├── t2/
│   ├── *.{png,jpeg,jpg}
│   ├── *.{png,jpeg,jpg}
│   └── ...
└── ...
```




> **Note:**  
> While you can obtain the poses yourself, our codebase currently only supports NeRF-Synthetic and COLMAP poses. Additionally, their naming scheme must be consistent with the one produced by our script! 


### Running CL-Splats
You can run CL-Splats using our intuitive CLI `cl-splats-train`. Start by training your initial scene representation:

```bash
cl-splats-train --data-path <path/to/dataset>
```

You can view all available CLI options using the `--help` flag:
```bash
cl-splats-train --help
```

Our pipeline natively exports standard `.ply` formats after training. The resulting point clouds are saved automatically to the training output directory (by default `outputs/`), allowing them to be imported directly into standard viewers like SuperSplat.

You can also override any configuration value using Hydra overrides:
```bash
cl-splats-train --data-path <path/to/dataset> optimization.lr=0.01
```

## Todos
I continue to release the missing modules required to replicate our method.

- [x] Release initial codebase with framework skeleton.
- [x] Release camera estimation script.
- [x] Release fast change detection module.
- [x] Release sampling module.
- [x] Release pruning module.
- [x] Release data.
- [x] ~~Release local-optimization CUDA kernels.~~
- [x] Verify codebase.
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
