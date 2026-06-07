"""Preprocessing entry point placeholder.

The public MPIIFaceGaze, RT-GENE, and ETH-XGaze datasets have different
licenses and directory layouts. For publication, raw datasets should be
downloaded from their official providers and converted into the HDF5 schema
documented in README.md.

This script currently validates a prepared HDF5 directory. Add project-specific
cropping/alignment code here if releasing the full preprocessing pipeline is
permitted by your dataset licenses.
"""

import argparse
from pathlib import Path

import h5py

from data.datasets import REQUIRED_H5_KEYS


def validate_h5_dir(data_dir: Path) -> None:
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files were found in {data_dir}")

    for path in files:
        with h5py.File(path, "r") as fid:
            missing = [key for key in REQUIRED_H5_KEYS if key not in fid]
            if missing:
                raise KeyError(f"{path} is missing H5 keys: {missing}")
        print(f"OK: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate preprocessed gaze HDF5 files.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing .h5 files.")
    return parser.parse_args()


if __name__ == "__main__":
    validate_h5_dir(parse_args().data_dir)
