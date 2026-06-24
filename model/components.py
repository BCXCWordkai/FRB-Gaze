import warnings

import torch
import torch.nn as nn
import timm


class CoordEnhancer(nn.Module):
    """Append normalized coordinate channels to an image tensor."""

    def __init__(self, coord_type: str = "xyr"):
        super().__init__()
        if coord_type not in {"none", "xy", "xyr"}:
            raise ValueError("coord_type must be one of: none, xy, xyr")
        self.coord_type = coord_type

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.coord_type == "none":
            return input_tensor

        batch_size, _, height, width = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype

        y_coords = 2.0 * torch.arange(height, dtype=dtype, device=device) / (height - 1) - 1.0
        x_coords = 2.0 * torch.arange(width, dtype=dtype, device=device) / (width - 1) - 1.0
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)

        if self.coord_type == "xy":
            return torch.cat([input_tensor, x_coords, y_coords], dim=1)

        r_coords = torch.sqrt(torch.square(x_coords) + torch.square(y_coords) + 1e-6)
        return torch.cat([input_tensor, x_coords, y_coords, r_coords], dim=1)


def _load_local_checkpoint(backbone: nn.Module, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    try:
        timm.models.load_checkpoint(backbone, checkpoint_path, strict=False)
    except FileNotFoundError:
        warnings.warn(f"Backbone checkpoint not found: {checkpoint_path}", RuntimeWarning)


class EyeFeatureExtractor(nn.Module):
    def __init__(
        self,
        coord_type: str = "xyr",
        out_dim: int = 128,
        backbone_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.add_coords = CoordEnhancer(coord_type)
        in_channels = 3 + (2 if coord_type == "xy" else 3 if coord_type == "xyr" else 0)

        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        _load_local_checkpoint(self.backbone, checkpoint_path)

        orig_conv = self.backbone.conv_stem
        if in_channels > 3:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=(orig_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = orig_conv.weight.clone()
                nn.init.zeros_(new_conv.weight[:, 3:, :, :])
                if orig_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
            self.backbone.conv_stem = new_conv

        self.shared_spatial = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 7))
        self.max_pool = nn.AdaptiveMaxPool2d((4, 7))

        eye_flat_dim = 128 * 2 * 4 * 7
        self.feat_head = nn.Sequential(
            nn.Linear(eye_flat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(eye_flat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.add_coords(x)
        features = self.backbone.forward_features(x)
        features = self.shared_spatial(features)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1)
        pooled = torch.flatten(pooled, start_dim=1)
        return self.feat_head(pooled), self.confidence_head(pooled)


class FaceFeatureExtractor(nn.Module):
    def __init__(
        self,
        out_dim: int = 64,
        backbone_name: str = "convnextv2_nano",
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        _load_local_checkpoint(self.backbone, checkpoint_path)
        in_features = self.backbone.num_features

        self.attention_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.backbone(x)
        return self.attention_head(shared_features), self.proj_head(shared_features)


def pitch_yaw_to_vector(poses: torch.Tensor) -> torch.Tensor:
    pitch, yaw = poses[:, 0], poses[:, 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=1)


class FusionConcatHead(nn.Module):
    def __init__(self, visual_dim: int = 192, pose_dim: int = 2, use_pose_proj: bool = True):
        super().__init__()
        self.use_pose_proj = use_pose_proj
        final_pose_dim = 32 if use_pose_proj else pose_dim

        if self.use_pose_proj:
            self.pose_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())

        self.regressor = nn.Sequential(
            nn.Linear(visual_dim + final_pose_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, visual_features: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        pose_features = self.pose_proj(pitch_yaw_to_vector(poses)) if self.use_pose_proj else poses
        return self.regressor(torch.cat([visual_features, pose_features], dim=1))


class FusionAdaptiveHead(nn.Module):
    def __init__(self, visual_dim: int = 192, pose_dim: int = 2):
        super().__init__()
        self.visual_se = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Linear(visual_dim // 2, visual_dim),
            nn.Sigmoid(),
        )
        self.pose_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(visual_dim + 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, visual_features: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        attention = self.visual_se(visual_features)
        pose_features = self.pose_proj(pitch_yaw_to_vector(poses))
        return self.regressor(torch.cat([visual_features * attention, pose_features], dim=1))


class FusionHierarchicalHead(nn.Module):
    def __init__(self, visual_dim: int = 192, pose_dim: int = 2):
        super().__init__()
        self.visual_fusion = nn.Sequential(nn.Linear(visual_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.pose_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, visual_features: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        visual_features = self.visual_fusion(visual_features)
        pose_features = self.pose_proj(pitch_yaw_to_vector(poses))
        return self.regressor(torch.cat([visual_features, pose_features], dim=1))
