import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_gaze_dataset
from models import ABLATION_CONFIGS, FRBGazeNet
from utils import compute_angular_error, tta_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FRB-Gaze.")
    parser.add_argument("--config", type=Path, default=None, help="YAML file with checkpoint and test metadata.")
    parser.add_argument("--test-dir", type=Path, default=None, help="Directory of test .h5 files.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a trained .pth checkpoint.")
    parser.add_argument("--ablation-id", choices=ABLATION_CONFIGS.keys(), default=None)
    parser.add_argument("--eye-checkpoint", type=str, default=None, help="Optional local eye backbone checkpoint.")
    parser.add_argument("--face-checkpoint", type=str, default=None, help="Optional local face backbone checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-tta", action="store_true")
    args = parser.parse_args()
    return apply_config(args)


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        if args.test_dir is None or args.checkpoint is None:
            raise ValueError("Provide either --config or both --test-dir and --checkpoint.")
        args.ablation_id = args.ablation_id or "F1_BASE_CONCAT"
        args.batch_size = args.batch_size or 128
        return args

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    checkpoint_info = config.get("checkpoint", {})
    test_info = config.get("test", {})
    args.test_dir = args.test_dir or Path(test_info["test_dir"])
    args.checkpoint = args.checkpoint or Path(checkpoint_info["local_path"])
    args.ablation_id = args.ablation_id or config.get("ablation_id", "F1_BASE_CONCAT")
    args.batch_size = args.batch_size or int(test_info.get("batch_size", 128))
    return args


def load_state_dict(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_gaze_dataset(args.test_dir, use_augmentation=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = FRBGazeNet(
        ABLATION_CONFIGS[args.ablation_id],
        eye_checkpoint=args.eye_checkpoint,
        face_checkpoint=args.face_checkpoint,
    ).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    model.eval()

    errors = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            faces, eyes_left, eyes_right, poses, gaze = [x.to(device, non_blocking=True) for x in batch]
            pred = tta_inference(model, faces, eyes_left, eyes_right, poses) if args.use_tta else model(faces, eyes_left, eyes_right, poses)
            errors.append(compute_angular_error(pred, gaze))

    mae = torch.cat(errors).mean().item()
    print(f"Mean angular error: {mae:.2f} degrees")


if __name__ == "__main__":
    main()
