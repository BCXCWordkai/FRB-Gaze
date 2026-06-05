import torch
import torch.nn as nn

from model import (
    EyeFeatureExtractor,
    FaceFeatureExtractor,
    FusionConcatHead,
    FusionAdaptiveHead,
    FusionHierarchicalHead
)

ABLATION_CONFIGS = {
    "F1_BASE_CONCAT": {
        "use_eye_coordconv": "xyr",
        "eye_dim": 128,
        "use_face_guide": True,
        "face_dim": 64,
        "eye_fusion_hidden_dim": 128,
        "eye_fusion_dropout": 0.2,
        "fusion_strategy": "adaptive", #adaptive#hierarchical
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
        "use_eye_coordconv": "xyr",#xyr
        "eye_dim": 128,
        "use_face_guide": False,
        "face_dim": 64,
        "eye_fusion_hidden_dim": 128,
        "eye_fusion_dropout": 0.1,
        "fusion_strategy": "concat", #adaptive
        "pose_dim": 2,
        "share_eye_backbone": True,
        "use_pose_proj": False,
    }
}


class FGS_GazeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 保守版 + 共享权重
        if config.get("share_eye_backbone", True):
            self.eye_net = EyeFeatureExtractor(
                coord_type=config["use_eye_coordconv"],
                out_dim=config["eye_dim"]
            )
            self.eyeL_net = None
            self.eyeR_net = None
        else:
            self.eyeL_net = EyeFeatureExtractor(
                coord_type=config["use_eye_coordconv"],
                out_dim=config["eye_dim"]
            )
            self.eyeR_net = EyeFeatureExtractor(
                coord_type=config["use_eye_coordconv"],
                out_dim=config["eye_dim"]
            )
            self.eye_net = None

        self.face_net = FaceFeatureExtractor(out_dim=config["face_dim"])

        eye_dim = config["eye_dim"]
        eye_fusion_hidden_dim = config.get("eye_fusion_hidden_dim", eye_dim)
        eye_fusion_dropout = config.get("eye_fusion_dropout", 0.1)


        self.eye_fusion = nn.Sequential(
            nn.Linear(eye_dim, eye_fusion_hidden_dim),
            nn.LayerNorm(eye_fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(eye_fusion_dropout),
            nn.Linear(eye_fusion_hidden_dim, eye_dim),
            nn.LayerNorm(eye_dim),
            nn.GELU()
        )

        visual_dim = eye_dim + config["face_dim"]
        pose_dim = config["pose_dim"]


        use_pose_proj = config.get("use_pose_proj", True)

        if config["fusion_strategy"] == "concat":
            # 关键修改：把 use_pose_proj 传进去！
            self.fusion_head = FusionConcatHead(visual_dim, pose_dim, use_pose_proj=use_pose_proj)
        elif config["fusion_strategy"] == "adaptive":
            self.fusion_head = FusionAdaptiveHead(visual_dim, pose_dim)
        elif config["fusion_strategy"] == "hierarchical":
            self.fusion_head = FusionHierarchicalHead(visual_dim, pose_dim)
        else:
            raise ValueError(f"不支持的融合策略: {config['fusion_strategy']}")

    def forward(self, faces, eyesLeft, eyesRight, poses):
        if self.eye_net is not None:
            feat_L, conf_L = self.eye_net(eyesLeft)
            feat_R, conf_R = self.eye_net(eyesRight)
        else:
            feat_L, conf_L = self.eyeL_net(eyesLeft)
            feat_R, conf_R = self.eyeR_net(eyesRight)

        weights, feat_F = self.face_net(faces)

        if self.config["use_face_guide"]:
            w_L_raw = weights[:, 0:1] * conf_L
            w_R_raw = weights[:, 1:2] * conf_R
            w_sum = w_L_raw + w_R_raw + 1e-6

            feat_L = feat_L * (w_L_raw / w_sum)
            feat_R = feat_R * (w_R_raw / w_sum)


        fused_eyes = self.eye_fusion(feat_L + feat_R)
        visual_concat = torch.cat([fused_eyes, feat_F], dim=1)

        out_gaze = self.fusion_head(visual_concat, poses)
        return out_gaze


if __name__ == "__main__":
    
    experiment_name = "F1_BASE_CONCAT"
    current_config = ABLATION_CONFIGS[experiment_name]

    print(f"🚀 正在初始化实验: {experiment_name}")
    model = FGS_GazeNet(current_config)

    bs = 2
    dummy_faces = torch.randn(bs, 3, 224, 224)
    dummy_eyesL = torch.randn(bs, 3, 128, 224)
    dummy_eyesR = torch.randn(bs, 3, 128, 224)
    dummy_poses = torch.randn(bs, 2)

    output = model(dummy_faces, dummy_eyesL, dummy_eyesR, dummy_poses)
    print(f"✅ 前向传播成功! 输出形状: {output.shape}")