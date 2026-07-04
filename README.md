# FRB-Gaze

Official implementation of **FRB-Gaze: Reliability-Aware Binocular Gaze Estimation under Asymmetric Visual Degradation**.

This repository contains the model implementation, HDF5 data reader, dataset-format checks, training and evaluation scripts, reproducibility configuration examples, and representative log templates for the experiments reported in the manuscript.

## Repository Structure

```text
FRB-Gaze/
|-- configs/               # Reproducibility configuration examples
|-- data/
|   |-- datasets.py        # HDF5 dataset reader and image transformations
|   `-- preprocess.py      # HDF5 format validation script
|-- docs/
|   `-- reproduce_tables.md
|-- logs/
|   `-- README.md          # Representative log format and expected outputs
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

## Environment

The code was developed with Python and PyTorch. A CUDA-enabled GPU is recommended for training.

```bash
git clone https://github.com/BCXCWordkai/FRB-Gaze.git
cd FRB-Gaze

conda create -n frb-gaze python=3.10 -y
conda activate frb-gaze
pip install -r requirements.txt
```

The main dependencies are listed in `requirements.txt`. If the default PyTorch wheel does not match your CUDA version, install the appropriate PyTorch build from the official PyTorch website first, then run `pip install -r requirements.txt`.

## Public Datasets

The experiments use three public gaze-estimation datasets. Raw datasets, processed split files, and derived data packages are not redistributed in this repository because they are subject to the licenses and terms of use of the original dataset providers.

Download the datasets from their official sources and follow the corresponding license, use, and citation requirements:

- MPIIFaceGaze: https://www.perceptualui.org/research/datasets/MPIIFaceGaze/
- RT-GENE: https://github.com/Tobias-Fischer/rt_gene
- ETH-XGaze: https://ait.ethz.ch/xgaze

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

## HDF5 Data Format

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

## Backbone Weights

FRB-Gaze uses two `timm` backbones:

- Eye branch: `mobilenetv4_conv_small.e2400_r224_in1k`
- Face branch: `convnextv2_nano`

For offline or fully controlled experiments, download the corresponding pretrained weights and provide their local paths during training and testing:

```bash
--eye-checkpoint weights/mobilenetv4_conv_small.safetensors
--face-checkpoint weights/convnextv2_nano.safetensors
```

If these two arguments are omitted, the backbone networks are initialized without local pretrained checkpoints.

## Reproducing the Main Results

All experiments use seed `42` unless otherwise stated. The configuration examples in `configs/` record the dataset paths and main hyperparameters used for the manuscript experiments. Use `--ablation-id FULL` for the complete FRB-Gaze model.

### MPIIFaceGaze

```bash
python train.py --train-dir datasets/MPIIFaceGaze/train --val-dir datasets/MPIIFaceGaze/val --save-dir checkpoints/MPIIFaceGaze --batch-size 128 --epochs 100 --seed 42
python test.py --test-dir datasets/MPIIFaceGaze/test --checkpoint checkpoints/MPIIFaceGaze/FRB_Gaze_best.pth --batch-size 128 --ablation-id FULL
```



### RT-GENE

```bash
python train.py --train-dir datasets/RT-GENE/train --val-dir datasets/RT-GENE/val --save-dir checkpoints/RT-GENE --batch-size 128 --epochs 100 --seed 42
python test.py --test-dir datasets/RT-GENE/test --checkpoint checkpoints/RT-GENE/FRB_Gaze_best.pth --batch-size 128 --ablation-id FULL
```



### ETH-XGaze

```bash
python train.py --train-dir datasets/ETH-XGaze/train --val-dir datasets/ETH-XGaze/val --save-dir checkpoints/ETH-XGaze --batch-size 128 --epochs 100 --seed 42
python test.py --test-dir datasets/ETH-XGaze/test --checkpoint checkpoints/ETH-XGaze/FRB_Gaze_best.pth --batch-size 128 --ablation-id FULL
```



See `docs/reproduce_tables.md` for the command mapping used to reproduce the main manuscript tables.

## Reproducibility Details

- Random seed: `42`
- Optimizer: AdamW
- Initial learning rate: `3e-4`
- Minimum learning rate: `1e-7`
- Weight decay: `1e-3`
- Default batch size: `128`
- Training schedule: warmup followed by cosine annealing
- Checkpoint saved by validation performance: `FRB_Gaze_best.pth`
- Optional EMA checkpoint: `FRB_Gaze_best_ema.pth`
- Main metric: mean angular error in degrees

Representative log formats and expected final outputs are provided in `logs/README.md`. Trained weights, if publicly released, should be provided as versioned release assets rather than committed to the repository history.

## License

The code in this repository is released under the MIT License. See `LICENSE` for details.

The public datasets remain under the licenses and terms of use of their original providers.

## Citation

If you use this code, please cite the associated manuscript:

```text
FRB-Gaze: Face-Guided Reliability-Aware Binocular Gaze Estimation under Asymmetric Visual Degradation
```

The citation metadata in `CITATION.cff` can be updated after the paper receives a DOI.
