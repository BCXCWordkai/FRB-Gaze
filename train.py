import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_gaze_dataset
from models import ABLATION_CONFIGS, FRBGazeNet
from utils import CombinedGazeLoss, compute_angular_error, set_seed, tta_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FRB-Gaze.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Directory of training .h5 files.")
    parser.add_argument("--val-dir", type=Path, required=True, help="Directory of validation .h5 files.")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Checkpoint output directory.")
    parser.add_argument("--ablation-id", choices=ABLATION_CONFIGS.keys(), default="F1_BASE_CONCAT")
    parser.add_argument("--eye-checkpoint", type=str, default=None, help="Optional local eye backbone checkpoint.")
    parser.add_argument("--face-checkpoint", type=str, default=None, help="Optional local face backbone checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = build_gaze_dataset(args.train_dir, use_augmentation=True)
    val_ds = build_gaze_dataset(args.val_dir, use_augmentation=False)
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    model = FRBGazeNet(
        ABLATION_CONFIGS[args.ablation_id],
        eye_checkpoint=args.eye_checkpoint,
        face_checkpoint=args.face_checkpoint,
    ).to(device)
    ema = ModelEmaV2(model, decay=args.ema_decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs),
            CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs - args.warmup_epochs, eta_min=args.min_lr),
        ],
        milestones=[args.warmup_epochs],
    )
    criterion = CombinedGazeLoss(alpha=0.5).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_norm_mae = 100.0
    best_ema_mae = 100.0
    no_improve_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mae = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            faces, eyes_left, eyes_right, poses, gaze = [x.to(device, non_blocking=True) for x in batch]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_gaze = model(faces, eyes_left, eyes_right, poses)
                loss = criterion(pred_gaze, gaze)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            with torch.no_grad():
                train_mae.append(compute_angular_error(pred_gaze, gaze).mean().item())

        scheduler.step()
        norm_errors = []
        ema_errors = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                faces, eyes_left, eyes_right, poses, gaze = [x.to(device, non_blocking=True) for x in batch]
                pred = tta_inference(model, faces, eyes_left, eyes_right, poses) if args.use_tta else model(faces, eyes_left, eyes_right, poses)
                pred_ema = tta_inference(ema.module, faces, eyes_left, eyes_right, poses) if args.use_tta else ema.module(faces, eyes_left, eyes_right, poses)
                norm_errors.append(compute_angular_error(pred, gaze))
                ema_errors.append(compute_angular_error(pred_ema, gaze))

        norm_mae = torch.cat(norm_errors).mean().item()
        ema_mae = torch.cat(ema_errors).mean().item()
        train_epoch_mae = float(np.mean(train_mae))
        improved = False

        if norm_mae < best_norm_mae:
            best_norm_mae = norm_mae
            torch.save(model.state_dict(), args.save_dir / "FRB_Gaze_best.pth")
            improved = True
        if ema_mae < best_ema_mae:
            best_ema_mae = ema_mae
            torch.save(ema.module.state_dict(), args.save_dir / "FRB_Gaze_best_ema.pth")
            improved = True
        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "norm_mae": norm_mae,
                    "ema_mae": ema_mae,
                },
                args.save_dir / f"FRB_Gaze_epoch_{epoch}.pth",
            )

        no_improve_count = 0 if improved or epoch <= args.warmup_epochs else no_improve_count + 1
        print(
            f"Epoch {epoch}: train MAE={train_epoch_mae:.2f}, "
            f"val MAE={norm_mae:.2f}, EMA MAE={ema_mae:.2f}"
        )
        if no_improve_count >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.")
            break


if __name__ == "__main__":
    main()
