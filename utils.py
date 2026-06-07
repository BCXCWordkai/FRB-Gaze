import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pitch_yaw_to_vector(pitch_yaw: torch.Tensor) -> torch.Tensor:
    pitch, yaw = pitch_yaw[:, 0], pitch_yaw[:, 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=1)


def compute_angular_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_vec = pitch_yaw_to_vector(preds)
    target_vec = pitch_yaw_to_vector(targets)
    similarity = torch.clamp(torch.sum(pred_vec * target_vec, dim=1), -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(similarity) * (180.0 / np.pi)


class CombinedGazeLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_vec = pitch_yaw_to_vector(preds)
        target_vec = pitch_yaw_to_vector(targets)
        similarity = torch.clamp(torch.sum(pred_vec * target_vec, dim=1), -1.0 + 1e-7, 1.0 - 1e-7)
        angular_loss = torch.acos(similarity).mean()
        l1_loss = self.smooth_l1(preds, targets)
        return self.alpha * angular_loss + (1.0 - self.alpha) * l1_loss


def tta_inference(model, faces, eyes_left, eyes_right, poses):
    pred = model(faces, eyes_left, eyes_right, poses)

    faces_flip = torch.flip(faces, [3])
    eyes_left_flip = torch.flip(eyes_right, [3])
    eyes_right_flip = torch.flip(eyes_left, [3])
    poses_flip = poses.clone()
    poses_flip[:, 1] = -poses_flip[:, 1]

    pred_flip = model(faces_flip, eyes_left_flip, eyes_right_flip, poses_flip)
    pred_flip[:, 1] = -pred_flip[:, 1]
    return (pred + pred_flip) / 2.0
