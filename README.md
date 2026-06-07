# FRB-Gaze

Official code for **FRB-Gaze**, a deep learning framework for appearance-based 3D gaze estimation. This repository is prepared for the code-availability requirement of *Pattern Analysis and Applications*.

The implementation contains the model definition, HDF5 dataset loader, training script, and evaluation script used for experiments on MPIIFaceGaze, RT-GENE, and ETH-XGaze.

## Repository structure

```text
FRB-Gaze/
├── data/
│   ├── datasets.py        # HDF5 dataset reader and augmentation
│   └── preprocess.py      # HDF5 validation/preprocessing entry point
├── models/
│   ├── components.py      # CoordConv, eye/face branches, fusion heads
│   └── frb_gaze.py        # FRB-Gaze network and ablation configs
├── train.py               # Training entry point
├── test.py                # Evaluation entry point
├── utils.py               # Loss, metrics, TTA, reproducibility utilities
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Installation

```bash
git clone https://github.com/BCXCWordkai/FRB-Gaze.git
cd FRB-Gaze
conda create -n frb-gaze python=3.10 -y
conda activate frb-gaze
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default package is not suitable for your machine.

## Datasets

This code expects preprocessed HDF5 files. Raw datasets are not redistributed in this repository because they are controlled by the original dataset licenses.

Official dataset pages:

- MPIIFaceGaze: [https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/](https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/)
- RT-GENE: [https://github.com/Tobias-Fischer/rt_gene](https://github.com/Tobias-Fischer/rt_gene)
- ETH-XGaze: [https://ait.ethz.ch/xgaze](https://ait.ethz.ch/xgaze)

After preprocessing, organize files as:

```text
datasets/
├── train/
│   ├── subject_001.h5
│   └── ...
├── val/
│   ├── subject_010.h5
│   └── ...
└── test/
    ├── subject_020.h5
    └── ...
```

Each `.h5` file should contain:

| Key | Shape | Description |
| --- | --- | --- |
| `face_patch` | `N x H x W x 3` | Cropped face image in BGR/RGB-compatible array format |
| `left_eye` | `N x H x W x 3` | Cropped left-eye image |
| `right_eye` | `N x H x W x 3` | Cropped right-eye image |
| `face_head_pose` | `N x 2` | Head pose as pitch/yaw in radians |
| `face_gaze` | `N x 2` | Ground-truth gaze as pitch/yaw in radians |
| `is_valid` | `N` | Optional validity mask |

Validate a processed split with:

```bash
python data/preprocess.py --data-dir datasets/train
```

## Backbone checkpoints

The model uses `timm` backbones:

- Eye branch: `mobilenetv4_conv_small.e2400_r224_in1k`
- Face branch: `convnextv2_nano`

For fully offline training, download the corresponding pretrained weights from the `timm`/Hugging Face model pages and pass local paths with `--eye-checkpoint` and `--face-checkpoint`. If these arguments are omitted, the backbones are initialized without local pretrained weights.

## Training

```bash
python train.py \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --save-dir checkpoints/frb_gaze \
  --ablation-id F4_ADAPTIVE_HEAD \
  --batch-size 128 \
  --epochs 40
```

Optional local backbone checkpoints:

```bash
python train.py \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --save-dir checkpoints/frb_gaze \
  --eye-checkpoint weights/mobilenetv4_conv_small.safetensors \
  --face-checkpoint weights/convnextv2_nano.safetensors
```

Saved checkpoints include:

- `F4_ADAPTIVE_HEAD_best_norm.pth`
- `F4_ADAPTIVE_HEAD_best_ema.pth`
- Periodic epoch snapshots every 5 epochs

## Evaluation

```bash
python test.py \
  --test-dir datasets/test \
  --checkpoint checkpoints/frb_gaze/F4_ADAPTIVE_HEAD_best_ema.pth \
  --ablation-id F4_ADAPTIVE_HEAD
```

The reported metric is mean angular error in degrees.

## Ablation configurations

The available configurations are defined in `models/frb_gaze.py`:

- `F1_BASE_CONCAT`
- `F2_HIERARCHICAL_FUSION`
- `F3_NO_COORD_NO_GUIDE`
- `F4_ADAPTIVE_HEAD`

Use `--ablation-id` to switch between them.

## Reproducibility notes

- Random seeds are fixed by default with `--seed 42`.
- Training uses AdamW, warmup, cosine annealing warm restarts, mixed precision, and EMA.
- Horizontal flip test-time augmentation can be enabled with `--use-tta`.
- Dataset files and pretrained model weights should be cited according to their original licenses and papers.

## Citation

If you use this repository, please cite the associated Pattern Analysis and Applications paper. The citation metadata can be updated in `CITATION.cff` after the paper is accepted or assigned a DOI.
