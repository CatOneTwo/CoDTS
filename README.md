# CoDTS (AAAI 2025)

CoDTS: Enhancing Sparsely Supervised Collaborative Perception with a Dual Teacher-Student Framework ([Paper](https://arxiv.org/abs/2412.08344))

## Features:
- Dataset Support
  - [x] [OPV2V (ICRA 2022)](https://arxiv.org/abs/2109.07644)
  - [x] [V2X-Sim 2.0 (NeurIPS 2021)](https://arxiv.org/abs/2111.0064)
  - [x] [DAIR-V2X (CVPR 2022)](https://arxiv.org/abs/2204.05575)
  - [x] [V2V4Real (CVPR 2023)](https://arxiv.org/abs/2303.07601)

- Collaborative perception methods (Used in paper)
    - [x] [Attentive Fusion [ICRA 2022]](https://arxiv.org/abs/2109.07644)
    - [x] [F-Cooper [SEC 2019]](https://arxiv.org/abs/1909.06459)
    - [x] [DiscoNet [NeurIPS 2021]](https://arxiv.org/abs/2111.00643)

- Visualization support
  - [x] BEV visualization
  - [x] 3D visualization

## Installation

This code is based on [CoAlign](https://github.com/yifanlu0227/CoAlign), so I recommend you visit [CoAlign Installation Guide](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie) to learn how to install this repo.

Or you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install CoAlign. The installation is totally the same as OpenCOOD.


## Dataset and Sparse Label

**1. Download datasets**

- DAIR-V2X [Download](https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file#dataset) 

  > Note: We use the Complemented Annotations for DAIR-V2X-C from [Website](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/)

- V2V4Real [Download](https://mobility-lab.seas.ucla.edu/v2v4real/)

- OPV2V [Download](https://mobility-lab.seas.ucla.edu/opv2v/)

- V2X-Sim 2.0 [Download](https://ai4ce.github.io/V2X-Sim/download.html)

**2. Download  sparse labels**

The sparse label in this paper is in [Google Drive](https://drive.google.com/drive/folders/1U0CE1MXR23Tg9W1H4-cPdNF7PANoWcWP?usp=sharing), you can also produce sparse labels with codes in the "data_preparation" folder.

**3. Put the folders of sparse labels for different datasets in the following:**
```bash
CoDTS 
├── data_preparation # scripts to generate sparse labels
├── dataset # root of your dataset
│   ├── my_dair_v2x
│       ├── v2x_c
│         ├── cooperative-vehicle-infrastructure
│           ├── cooperative/label_world_sparse
│           ├── infrastructure-side/label/virtuallidar_sparse
│           ├── vehicle-side/label/lidar_sparse
│   ├── v2vreal
│       ├── train
│       ├── train_sparse # sparse label
│       ├── validate
│   ├── OPV2V
│       ├── test
│       ├── train
│       ├── train_sparse # sparse label
│       ├── validate
│   ├── V2X-Sim-2.0 # original v2xsim dataset
│   ├── v2xsim2_info # the label index file for v2xsim
│       ├── v2xsim_infos_test.pkl
│       ├── v2xsim_infos_train_plus.pkl # sparse label
│       ├── v2xsim_infos_train.pkl 
│       ├── v2xsim_infos_val.pkl
├── opencood # the core codebase
```
## Training and Inference
Run the bash command in the terminal and modify the "DATASET" and "METHOD" parameters in the bash file to switch the dataset and collaborative detector.

**Inference detectors**
```bash
bash 0_test.bash
```

**Train detectors with full/sparse labels only**
```bash
bash 1_train_from_scratch.bash
```

**Train detectors with sparse and pseudo labels**

[SSC3OD: Sparsely Supervised Collaborative 3D Object Detection from LiDAR Point Clouds (SMC 2023)](https://arxiv.org/abs/2307.00717)

```bash
bash 2_ssl_pretrain.bash
bash 3_train_static_teacher.bash
bash 4_train_SSC3OD.bash
```

[HINTED: Hard Instance Enhanced Detector with Mixed-Density Feature Fusion for Sparsely-Supervised 3D Object Detection (CVPR 2024)](https://ieeexplore.ieee.org/document/10655437)
```bash
bash 5_train_HINTED.bash
```
[CoDTS: Enhancing Sparsely Supervised Collaborative Perception with a Dual Teacher-Student Framework (AAAI 2025)](https://arxiv.org/abs/2412.08344)
```bash
bash 6_train_CoDTS.bash
```

> **Note**: Both HINTED and CoDTS are based on static teachers in SSC3OD.
>
> CoDTS generates pseudo labels based on the static teacher’s predictions, and the student is initialized with pre-trained encoder. 
>
> HINTED initially loads the static teacher’s parameters to the dynamic teacher and student before training.
>
> The pre-trained encoder "**DATASET_point_pillar_lidar_single_mae_0.7**" and static teacher "**DATASET_point_pillar_lidar_METHOD_ws_single_mae_0.7**" in SSC3OD are in [Google Driver](https://drive.google.com/drive/folders/1dFl97nP808uvrixmWlyAtKJeLCnd6G1j?usp=sharing).



## Citation
```
@inproceedings{han2025codts,
  title={CoDTS: Enhancing Sparsely Supervised Collaborative Perception with a Dual Teacher-Student Framework},
  author={Han, Yushan and Zhang, Hui and Zhang, Honglei and Wang, Jing and Li, Yidong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={3366--3373},
  year={2025}
}
```

## Acknowledgment

This project is impossible without the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CoAlign](https://github.com/yifanlu0227/CoAlign)!
