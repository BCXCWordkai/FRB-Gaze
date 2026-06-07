import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_gaze_dataset
from models import ABLATION_CONFIGS, FRBGazeNet
from utils import compute_angular_error, tta_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FRB-Gaze.")
    parser.add_argument("--test-dir", type=Path, required=True, help="Directory of test .h5 files.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained .pth checkpoint.")
    parser.add_argument("--ablation-id", choices=ABLATION_CONFIGS.keys(), default="F4_ADAPTIVE_HEAD")
    parser.add_argument("--eye-checkpoint", type=str, default=None, help="Optional local eye backbone checkpoint.")
    parser.add_argument("--face-checkpoint", type=str, default=None, help="Optional local face backbone checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-tta", action="store_true")
    return parser.parse_args()


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
