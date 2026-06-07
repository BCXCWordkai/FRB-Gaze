from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


REQUIRED_H5_KEYS = ("face_patch", "left_eye", "right_eye", "face_head_pose", "face_gaze")


class GazeH5Dataset(Dataset):
    """Dataset reader for the preprocessed HDF5 format used by this repository."""

    def __init__(self, h5_file_path: str | Path, use_augmentation: bool = False):
        self.h5_file_path = Path(h5_file_path)
        self.use_augmentation = use_augmentation

        with h5py.File(self.h5_file_path, "r") as fid:
            missing = [key for key in REQUIRED_H5_KEYS if key not in fid]
            if missing:
                raise KeyError(f"{self.h5_file_path} is missing H5 keys: {missing}")
            if "is_valid" in fid:
                valid_mask = fid["is_valid"][:]
                self.valid_indices = np.where(valid_mask)[0]
            else:
                self.valid_indices = np.arange(len(fid["face_gaze"]))

        self.h5_file = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.face_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.15, contrast=0.15) if use_augmentation else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.eye_transform = transforms.Compose(
            [
                transforms.Resize((128, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, "r", swmr=True)

        real_idx = self.valid_indices[idx]
        face_img = Image.fromarray(self.h5_file["face_patch"][real_idx][..., ::-1])
        left_eye = Image.fromarray(self.h5_file["left_eye"][real_idx][..., ::-1])
        right_eye = Image.fromarray(self.h5_file["right_eye"][real_idx][..., ::-1])
        head_pose = self.h5_file["face_head_pose"][real_idx].astype(np.float32)
        gaze = self.h5_file["face_gaze"][real_idx].astype(np.float32)

        if self.use_augmentation and np.random.rand() < 0.5:
            face_img = ImageOps.mirror(face_img)
            left_eye, right_eye = ImageOps.mirror(right_eye), ImageOps.mirror(left_eye)
            if head_pose.shape[0] >= 2:
                head_pose[1] = -head_pose[1]
            if gaze.shape[0] >= 2:
                gaze[1] = -gaze[1]

        return (
            self.face_transform(face_img),
            self.eye_transform(left_eye),
            self.eye_transform(right_eye),
            torch.tensor(head_pose, dtype=torch.float32),
            torch.tensor(gaze, dtype=torch.float32),
        )


def build_gaze_dataset(data_dir: str | Path, use_augmentation: bool = False) -> ConcatDataset:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files were found in {data_dir}")
    return ConcatDataset([GazeH5Dataset(path, use_augmentation=use_augmentation) for path in files])
