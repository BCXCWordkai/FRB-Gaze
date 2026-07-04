# FRB-Gaze

Official implementation of **FRB-Gaze: Face-Guided Reliability-Aware Binocular Gaze Estimation under Asymmetric Visual Degradation**.

This repository provides the model code, data reader, preprocessing checks, training and testing scripts, checkpoint metadata, and reproduction commands for the experiments reported in the manuscript.

## Repository Structure

```text
FRB-Gaze/
|-- config/
|   |-- mpiifacegaze_weights.yaml
|   |-- rt_gene_weights.yaml
|   `-- eth_xgaze_weights.yaml
|-- data/
|   |-- datasets.py
|   `-- preprocess.py
|-- docs/
|   `-- reproduce_tables.md
|-- logs/
|   `-- README.md
|-- models/
|   |-- components.py
|   `-- frb_gaze.py
|-- train.py
|-- test.py
|-- utils.py
|-- requirements.txt
|-- LICENSE
`-- CITATION.cff
```

## Environment

```bash
git clone https://github.com/BCXCWordkai/FRB-Gaze.git
cd FRB-Gaze

conda create -n frb-gaze python=3.10 -y
conda activate frb-gaze
pip install -r requirements.txt
```

The main dependency versions are listed in `requirements.txt`. A CUDA-enabled GPU is recommended for training. If the default PyTorch wheel does not match your CUDA version, install the corresponding PyTorch build first and then run `pip install -r requirements.txt`.

## Datasets

The experiments use three public gaze-estimation datasets:

- MPIIFaceGaze: https://www.perceptualui.org/research/datasets/MPIIFaceGaze/
- RT-GENE: https://github.com/Tobias-Fischer/rt_gene
- ETH-XGaze: https://ait.ethz.ch/xgaze

Please download the datasets from their original sources and follow their respective licenses, terms of use, and citation requirements. Raw datasets and processed split files are not redistributed in this repository.

After preprocessing, arrange the HDF5 files as follows:

```text
datasets/
|-- MPIIFaceGaze/
|   |-- p00/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   |-- p01/
|   `-- ...
|-- RT-GENE/
|   |-- fold_01/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   |-- fold_02/
|   `-- ...
`-- ETH-XGaze/
    |-- train/
    |-- val/
    `-- test/
```

Each split directory should contain one or more `.h5` files.

## HDF5 Format

Each `.h5` file should contain:

| Key | Shape | Description |
| --- | --- | --- |
| `face_patch` | `N x H x W x 3` | Cropped face image |
| `left_eye` | `N x H x W x 3` | Cropped left-eye image |
| `right_eye` | `N x H x W x 3` | Cropped right-eye image |
| `face_head_pose` | `N x 2` | Head pose represented by pitch and yaw in radians |
| `face_gaze` | `N x 2` | Ground-truth gaze represented by pitch and yaw in radians |
| `is_valid` | `N` | Optional validity mask |

Check the processed files before training:

```bash
python data/preprocess.py --data-dir datasets/MPIIFaceGaze/p00/train
python data/preprocess.py --data-dir datasets/RT-GENE/fold_01/train
python data/preprocess.py --data-dir datasets/ETH-XGaze/train
```

## Checkpoint Metadata

Checkpoint paths, SHA256 checksums, and testing commands are recorded in:

- `config/mpiifacegaze_weights.yaml`
- `config/rt_gene_weights.yaml`
- `config/eth_xgaze_weights.yaml`

## Reproducing the Main Results

All experiments use seed `42` unless otherwise specified. Use `--ablation-id FULL` for the complete FRB-Gaze model.

### MPIIFaceGaze

The provided MPIIFaceGaze checkpoint corresponds to subject fold `p00`.

```bash
python train.py --train-dir datasets/MPIIFaceGaze/p00/train --val-dir datasets/MPIIFaceGaze/p00/val --save-dir checkpoints/MPIIFaceGaze/p00 --batch-size 128 --epochs 100 --seed 42
python test.py --config config/mpiifacegaze_weights.yaml
```

### RT-GENE

The provided RT-GENE checkpoint corresponds to `fold_01`.

```bash
python train.py --train-dir datasets/RT-GENE/fold_01/train --val-dir datasets/RT-GENE/fold_01/val --save-dir checkpoints/RT-GENE/fold_01 --batch-size 128 --epochs 100 --seed 42
python test.py --config config/rt_gene_weights.yaml
```

### ETH-XGaze

```bash
python train.py --train-dir datasets/ETH-XGaze/train --val-dir datasets/ETH-XGaze/val --save-dir checkpoints/ETH-XGaze --batch-size 128 --epochs 100 --seed 42
python test.py --config config/eth_xgaze_weights.yaml
```

The manuscript result is around `3.26` degrees.

More table-level reproduction notes are provided in `docs/reproduce_tables.md`.

## Backbone Weights

FRB-Gaze uses two `timm` backbones:

- Eye branch: `mobilenetv4_conv_small.e2400_r224_in1k`
- Face branch: `convnextv2_nano`

For offline experiments, local pretrained backbone weights can be passed with:

```bash
--eye-checkpoint weights/mobilenetv4_conv_small.safetensors
--face-checkpoint weights/convnextv2_nano.safetensors
```

If these arguments are omitted, the backbones are initialized without local pretrained checkpoints.

## Reproducibility Details

- Random seed: `42`
- Optimizer: AdamW
- Initial learning rate: `3e-4`
- Minimum learning rate: `1e-7`
- Weight decay: `1e-3`
- Default batch size: `128`
- Training schedule: warmup followed by cosine annealing
- Main metric: mean angular error in degrees
- Representative log format: `logs/README.md`

## License

The code is released under the MIT License. See `LICENSE` for details. The public datasets remain under the licenses and terms of use of their original providers.

## Citation

If you use this code, please cite the associated manuscript:

```text
FRB-Gaze: Face-Guided Reliability-Aware Binocular Gaze Estimation under Asymmetric Visual Degradation
```
