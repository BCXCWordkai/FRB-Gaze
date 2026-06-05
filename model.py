import torch
import torch.nn as nn
import timm


class CoordEnhancer(nn.Module):

    def __init__(self, coord_type='xyr'):
        super().__init__()
        self.coord_type = coord_type

    def forward(self, input_tensor):
        if self.coord_type == 'none':
            return input_tensor

        batch_size, _, dim_y, dim_x = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype


        y_coords = 2.0 * torch.arange(dim_y, dtype=dtype, device=device) / (dim_y - 1) - 1.0
        x_coords = 2.0 * torch.arange(dim_x, dtype=dtype, device=device) / (dim_x - 1) - 1.0

        y_coords = y_coords.view(1, 1, dim_y, 1).expand(batch_size, 1, dim_y, dim_x)
        x_coords = x_coords.view(1, 1, 1, dim_x).expand(batch_size, 1, dim_y, dim_x)

        if self.coord_type == 'xy':
            return torch.cat([input_tensor, x_coords, y_coords], dim=1)
        elif self.coord_type == 'xyr':
            r_coords = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2) + 1e-6)
            return torch.cat([input_tensor, x_coords, y_coords, r_coords], dim=1)
        else:
            raise ValueError(f"不支持的 coord_type: {self.coord_type}")



class EyeFeatureExtractor(nn.Module):
    def __init__(self, coord_type='xyr', out_dim=128):
        super().__init__()
        self.add_coords = CoordEnhancer(coord_type)


        in_channels = 3 + (2 if coord_type == 'xy' else 3 if coord_type == 'xyr' else 0)


        local_weight_path = r"E:\gazedata\weight\mobilenetv4_conv_small.e2400_r224_in1k.safetensors"
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=False,
            num_classes=0
        )
        timm.models.load_checkpoint(self.backbone, local_weight_path, strict=False)


        orig_conv = self.backbone.conv_stem
        if in_channels > 3:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=(orig_conv.bias is not None)
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
            nn.Linear(512, out_dim)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(eye_flat_dim, 128), # 从 64 提升到 128，因为输入变大了
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.add_coords(x)


        features = self.backbone.forward_features(x)

        res = self.shared_spatial(features)

        feat_avg = self.avg_pool(res)  # [B, 128, 4, 7]
        feat_max = self.max_pool(res)  # [B, 128, 4, 7]


        shared_896 = torch.cat([feat_avg, feat_max], dim=1)
        shared_896 = torch.flatten(shared_896, start_dim=1)  # [B, 7168]


        feat = self.feat_head(shared_896)
        conf = self.confidence_head(shared_896)
        return feat, conf



class FaceFeatureExtractor(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        local_face_weight = r"E:\gazedata\weight\convnextv2_nano.fcmae_ft_in22k_in1k_384.safetensors"
        self.backbone = timm.create_model('convnextv2_nano', pretrained=False, num_classes=0)
        timm.models.load_checkpoint(self.backbone, local_face_weight, strict=False)

        in_features = self.backbone.num_features


        self.attention_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )


        self.proj_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        shared_features = self.backbone(x)
        weights = self.attention_head(shared_features)
        face_feat = self.proj_head(shared_features)
        return weights, face_feat




def get_pose_3d(poses):

    p, y = poses[:, 0], poses[:, 1]
    x = -torch.cos(p) * torch.sin(y)
    y_v = -torch.sin(p)
    z = -torch.cos(p) * torch.cos(y)
    return torch.stack([x, y_v, z], dim=1)


class FusionConcatHead(nn.Module):
    def __init__(self, visual_dim=192, pose_dim=2, use_pose_proj=True):
        super().__init__()
        self.use_pose_proj = use_pose_proj


        final_pose_dim = 32 if use_pose_proj else pose_dim

        if self.use_pose_proj:
            self.pose_proj = nn.Sequential(
                nn.Linear(3, 32),
                nn.LayerNorm(32),
                nn.GELU()
            )


        self.regressor = nn.Sequential(
            nn.Linear(visual_dim + final_pose_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, visual_features, poses):
        if self.use_pose_proj:
            poses_3d = get_pose_3d(poses)
            pose_feat = self.pose_proj(poses_3d)
        else:

            pose_feat = poses

        return self.regressor(torch.cat([visual_features, pose_feat], dim=1))


class FusionAdaptiveHead(nn.Module):
    def __init__(self, visual_dim=192, pose_dim=2):
        super().__init__()
        self.visual_se = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Linear(visual_dim // 2, visual_dim),
            nn.Sigmoid()
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(visual_dim + 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, visual_features, poses):
        attention = self.visual_se(visual_features)
        poses_3d = get_pose_3d(poses)
        pose_feat = self.pose_proj(poses_3d)
        return self.regressor(torch.cat([visual_features * attention, pose_feat], dim=1))


class FusionHierarchicalHead(nn.Module):
    def __init__(self, visual_dim=192, pose_dim=2):
        super().__init__()
        self.visual_fusion = nn.Sequential(
            nn.Linear(visual_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, visual_features, poses):
        visual_feat = self.visual_fusion(visual_features)
        poses_3d = get_pose_3d(poses)
        pose_feat = self.pose_proj(poses_3d)
        return self.regressor(torch.cat([visual_feat, pose_feat], dim=1))