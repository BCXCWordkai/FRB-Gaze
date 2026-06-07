# 👁️ FRB-Gaze

Official implementation of **FRB-Gaze: A Face-Guided Reliability-Aware Binocular Network for Gaze Estimation**.

This repository provides the code required to reproduce the experiments reported in the manuscript, including the complete model, data reader, training script, evaluation script, and instructions for preparing the public datasets.

## 📁 Repository Structure

```text
FRB-Gaze/
|-- data/
|   |-- datasets.py        # HDF5 dataset reader and image transformations
|   `-- preprocess.py      # HDF5 format validation script
|-- models/
|   |-- components.py      # Eye branch, face branch, and fusion modules
|   `-- frb_gaze.py        # Complete FRB-Gaze model
|-- train.py               # Training script
|-- test.py                # Evaluation script
|-- utils.py               # Loss, metric, and seed utilities
|-- requirements.txt
|-- LICENSE
`-- CITATION.cff
```

## ⚙️ Environment

The code was developed with Python and PyTorch. A CUDA-enabled GPU is recommended for training.

```bash
git clone https://github.com/BCXCWordkai/FRB-Gaze.git
cd FRB-Gaze

conda create -n frb-gaze python=3.10 -y
conda activate frb-gaze
pip install -r requirements.txt
```

If the default PyTorch package does not match your CUDA version, install the appropriate PyTorch build from the official PyTorch website first, then run `pip install -r requirements.txt`.

## 📦 Datasets

The experiments use three public gaze-estimation datasets. The raw datasets are not redistributed in this repository because they are subject to the licenses of the original dataset providers.

Please download the datasets from the official pages:

- MPIIFaceGaze: [https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/](https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/)
- RT-GENE: [https://github.com/Tobias-Fischer/rt_gene](https://github.com/Tobias-Fischer/rt_gene)
- ETH-XGaze: [https://ait.ethz.ch/xgaze](https://ait.ethz.ch/xgaze)

After downloading and preprocessing the datasets, arrange the files as follows:

```text
datasets/
|-- MPIIFaceGaze/
|   |-- train/
|   |-- val/
|   `-- test/
|-- RT-GENE/
|   |-- train/
|   |-- val/
|   `-- test/
`-- ETH-XGaze/
    |-- train/
    |-- val/
    `-- test/
```

Each split directory should contain one or more `.h5` files.

## 🗂️ HDF5 Data Format

Each `.h5` file should contain the following fields:

| Key | Shape | Description |
| --- | --- | --- |
| `face_patch` | `N x H x W x 3` | Cropped face image |
| `left_eye` | `N x H x W x 3` | Cropped left-eye image |
| `right_eye` | `N x H x W x 3` | Cropped right-eye image |
| `face_head_pose` | `N x 2` | Head pose represented by pitch and yaw in radians |
| `face_gaze` | `N x 2` | Ground-truth gaze represented by pitch and yaw in radians |
| `is_valid` | `N` | Optional validity mask |

Before training, check whether the processed files follow this format:

```bash
python data/preprocess.py --data-dir datasets/MPIIFaceGaze/train
python data/preprocess.py --data-dir datasets/RT-GENE/train
python data/preprocess.py --data-dir datasets/ETH-XGaze/train
```

## 🧠 Backbone Weights

FRB-Gaze uses two `timm` backbones:

- Eye branch: `mobilenetv4_conv_small.e2400_r224_in1k`
- Face branch: `convnextv2_nano`

For offline or fully controlled experiments, download the corresponding pretrained weights and provide their local paths during training and testing:

```bash
--eye-checkpoint weights/mobilenetv4_conv_small.safetensors
--face-checkpoint weights/convnextv2_nano.safetensors
```

If these two arguments are omitted, the backbone networks are initialized without local pretrained checkpoints.

## 🚀 Reproducing the Experiments

The following commands train the complete FRB-Gaze model and evaluate it using mean angular error in degrees.

### 📊 MPIIFaceGaze

```bash
python train.py \
  --train-dir datasets/MPIIFaceGaze/train \
  --val-dir datasets/MPIIFaceGaze/val \
  --save-dir checkpoints/MPIIFaceGaze \
  --batch-size 128 \
  --epochs 100 \
  --seed 42

python test.py \
  --test-dir datasets/MPIIFaceGaze/test \
  --checkpoint checkpoints/MPIIFaceGaze/FRB_Gaze_best.pth \
  --batch-size 128
```

### 📊 RT-GENE

```bash
python train.py \
  --train-dir datasets/RT-GENE/train \
  --val-dir datasets/RT-GENE/val \
  --save-dir checkpoints/RT-GENE \
  --batch-size 128 \
  --epochs 100 \
  --seed 42

python test.py \
  --test-dir datasets/RT-GENE/test \
  --checkpoint checkpoints/RT-GENE/FRB_Gaze_best.pth \
  --batch-size 128
```

### 📊 ETH-XGaze

```bash
python train.py \
  --train-dir datasets/ETH-XGaze/train \
  --val-dir datasets/ETH-XGaze/val \
  --save-dir checkpoints/ETH-XGaze \
  --batch-size 128 \
  --epochs 100 \
  --seed 42

python test.py \
  --test-dir datasets/ETH-XGaze/test \
  --checkpoint checkpoints/ETH-XGaze/FRB_Gaze_best.pth \
  --batch-size 128
```

The evaluation script prints:

```text
Mean angular error: XX.XX degrees
```

This value corresponds to the mean angular error reported in the manuscript.

## 🔁 Reproducibility Details

- The default random seed is `42`.
- The optimizer is AdamW.
- The initial learning rate is `3e-4`.
- The minimum learning rate is `1e-7`.
- The weight decay is `1e-3`.
- The default batch size is `128`.
- The training schedule uses warmup followed by cosine annealing.
- The best validation checkpoint is saved as `FRB_Gaze_best.pth`.
- The main evaluation metric is mean angular error in degrees.

## 🔓 Code and Data Availability

The source code is available at:

[https://github.com/BCXCWordkai/FRB-Gaze](https://github.com/BCXCWordkai/FRB-Gaze)

The public datasets used in this study are available from their official dataset websites subject to their respective licenses. The repository describes the required HDF5 data format and provides the commands used to rerun training and evaluation.

## 📚 Citation

If you use this code, please cite the associated manuscript:

```text
FRB-Gaze: A Face-Guided Reliability-Aware Binocular Network for Gaze Estimation.
```

The citation metadata in `CITATION.cff` can be updated after the paper receives a DOI.
