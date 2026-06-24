import torch
import torch.nn as nn

from .components import (
    EyeFeatureExtractor,
    FaceFeatureExtractor,
    FusionAdaptiveHead,
    FusionConcatHead,
    FusionHierarchicalHead,
)


ABLATION_CONFIGS = {
    "F1_BASE_CONCAT": {
        "use_eye_coordconv": "xyr",
        "eye_dim": 128,
        "use_face_guide": True,
        "face_dim": 64,
        "eye_fusion_hidden_dim": 128,
        "eye_fusion_dropout": 0.2,
        "fusion_strategy": "adaptive",
        "pose_dim": 2,
        "share_eye_backbone": True,
    },
    "F2_HIERARCHICAL_FUSION": {
        "use_eye_coordconv": "xyr",
        "eye_dim": 128,
        "use_face_guide": True,
        "face_dim": 64,
        "eye_fusion_hidden_dim": 128,
        "eye_fusion_dropout": 0.1,
        "fusion_strategy": "hierarchical",
        "pose_dim": 2,
        "share_eye_backbone": True,
    },
    "F3_NO_COORD_NO_GUIDE": {
        "use_eye_coordconv": "none",
        "eye_dim": 128,
        "use_face_guide": False,
        "face_dim": 64,
        "eye_fusion_hidden_dim": 128,
        "eye_fusion_dropout": 0.1,
        "fusion_strategy": "concat",
        "pose_dim": 2,
        "share_eye_backbone": True,
    },
    "F4_ADAPTIVE_HEAD": {
        "use_eye_coordconv": "xyr",
        "eye_dim": 128,
        "use_face_guide": False,
        "face_dim": 64,
        "eye_fusion_hidden_dim": 128,
        "eye_fusion_dropout": 0.1,
        "fusion_strategy": "concat",
        "pose_dim": 2,
        "share_eye_backbone": True,
        "use_pose_proj": False,
    },
}


class FRBGazeNet(nn.Module):
    def __init__(
        self,
        config: dict,
        eye_checkpoint: str | None = None,
        face_checkpoint: str | None = None,
    ):
        super().__init__()
        self.config = config

        if config.get("share_eye_backbone", True):
            self.eye_net = EyeFeatureExtractor(
                coord_type=config["use_eye_coordconv"],
                out_dim=config["eye_dim"],
                checkpoint_path=eye_checkpoint,
            )
            self.eyeL_net = None
            self.eyeR_net = None
        else:
            self.eyeL_net = EyeFeatureExtractor(
                coord_type=config["use_eye_coordconv"],
                out_dim=config["eye_dim"],
                checkpoint_path=eye_checkpoint,
            )
            self.eyeR_net = EyeFeatureExtractor(
                coord_type=config["use_eye_coordconv"],
                out_dim=config["eye_dim"],
                checkpoint_path=eye_checkpoint,
            )
            self.eye_net = None

        self.face_net = FaceFeatureExtractor(out_dim=config["face_dim"], checkpoint_path=face_checkpoint)
        eye_dim = config["eye_dim"]
        self.eye_fusion = nn.Sequential(
            nn.Linear(eye_dim, config.get("eye_fusion_hidden_dim", eye_dim)),
            nn.LayerNorm(config.get("eye_fusion_hidden_dim", eye_dim)),
            nn.GELU(),
            nn.Dropout(config.get("eye_fusion_dropout", 0.1)),
            nn.Linear(config.get("eye_fusion_hidden_dim", eye_dim), eye_dim),
            nn.LayerNorm(eye_dim),
            nn.GELU(),
        )

        visual_dim = eye_dim + config["face_dim"]
        pose_dim = config["pose_dim"]
        use_pose_proj = config.get("use_pose_proj", True)

        if config["fusion_strategy"] == "concat":
            self.fusion_head = FusionConcatHead(visual_dim, pose_dim, use_pose_proj=use_pose_proj)
        elif config["fusion_strategy"] == "adaptive":
            self.fusion_head = FusionAdaptiveHead(visual_dim, pose_dim)
        elif config["fusion_strategy"] == "hierarchical":
            self.fusion_head = FusionHierarchicalHead(visual_dim, pose_dim)
        else:
            raise ValueError(f"Unsupported fusion strategy: {config['fusion_strategy']}")

    def forward(
        self,
        faces: torch.Tensor,
        eyes_left: torch.Tensor,
        eyes_right: torch.Tensor,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        if self.eye_net is not None:
            feat_left, conf_left = self.eye_net(eyes_left)
            feat_right, conf_right = self.eye_net(eyes_right)
        else:
            feat_left, conf_left = self.eyeL_net(eyes_left)
            feat_right, conf_right = self.eyeR_net(eyes_right)

        weights, face_features = self.face_net(faces)
        if self.config["use_face_guide"]:
            left_weight = weights[:, 0:1] * conf_left
            right_weight = weights[:, 1:2] * conf_right
            weight_sum = left_weight + right_weight + 1e-6
            feat_left = feat_left * (left_weight / weight_sum)
            feat_right = feat_right * (right_weight / weight_sum)

        fused_eyes = self.eye_fusion(feat_left + feat_right)
        visual_features = torch.cat([fused_eyes, face_features], dim=1)
        return self.fusion_head(visual_features, poses)
