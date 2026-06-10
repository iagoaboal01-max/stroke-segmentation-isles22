"""
Clean cluster training script for the uploaded files:
- Dataset.py
- Model_acts.py
- UNet_classes.py

Main change requested:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

This script is meant to be run with sbatch on the cluster. It avoids notebook-only
code such as napari, zip/pickle visualization blocks, and local Windows paths.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from Dataset import Dataset_combined
from Model_acts import Model_actions
from UNet_classes import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net with AdamW on ISLES data")

    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path to ISLES-2022 folder. Example: $DATA/ISLES-2022/")
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV with dwi_path, adc_path and mask_path columns")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where results/checkpoints will be saved")

    parser.add_argument("--modalities", nargs="+", default=["dwi_path", "adc_path"],
                        help="CSV columns used as image modalities. Default: dwi_path adc_path")
    parser.add_argument("--image_size", nargs=3, type=int, default=[64, 128, 128],
                        help="Image size as D H W. Default: 64 128 128")
    parser.add_argument("--features", nargs="+", type=int, default=[64, 128, 256, 512],
                        help="UNet feature channels. Default: 64 128 256 512")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for final evaluation")
    parser.add_argument("--seed", type=int, default=67)

    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Fraction of cases used for test. Default: 0.15")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fraction of total cases used for validation. Default: 0.15")

    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Make base_path robust: Dataset_combined concatenates base_path + relative path
    base_path = args.base_path
    if not base_path.endswith("/"):
        base_path += "/"

    full_dataset = Dataset_combined(
        csv_file=args.csv,
        image_size=tuple(args.image_size),
        base_path=base_path,
        modalities=args.modalities,
        transform=True,
        name="all",
    )

    indices = np.arange(len(full_dataset))

    # 70/15/15 by default: first split off test, then split train/val
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    val_fraction_of_trainval = args.val_size / (1.0 - args.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_trainval,
        random_state=args.seed,
        shuffle=True,
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    pin = device == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    print(f"Total cases: {len(full_dataset)}", flush=True)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}", flush=True)
    print(f"Modalities: {args.modalities}", flush=True)
    print(f"Image size: {tuple(args.image_size)}", flush=True)

    model = UNet(
        in_channels=len(args.modalities),
        out_channels=1,
        features=args.features,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # ------------------------------------------------------------------
    # REQUESTED CHANGE: AdamW + weight_decay
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"Optimizer: AdamW | lr={args.lr} | weight_decay={args.weight_decay}", flush=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    mod_acts = Model_actions(model)

    print("\n" + "=" * 70, flush=True)
    print("Starting training", flush=True)
    print("=" * 70, flush=True)

    history_tuple = mod_acts.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=None,          # criterion is not used inside this Model_actions.train
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
    )

    history = {
        "train_losses": history_tuple[0],
        "val_losses": history_tuple[1],
        "train_dice_losses": history_tuple[2],
        "val_dice_losses": history_tuple[3],
        "train_focal_losses": history_tuple[4],
        "val_focal_losses": history_tuple[5],
    }

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save split indices for reproducibility
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    # Save final model
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "model_state": model.state_dict(),  # also save this key for compatibility with newer scripts
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
        },
        output_dir / "model_final.pt",
    )
    print(f"Saved final model to {output_dir / 'model_final.pt'}", flush=True)

    # Final evaluation
    print("\n" + "=" * 70, flush=True)
    print("Evaluating on test set", flush=True)
    print("=" * 70, flush=True)

    dice, roc_auc, pr_auc, vox_p, vox_r, les_p, les_r, mean_cd, median_cd = mod_acts.evaluate(
        test_loader,
        device=device,
        threshold=args.threshold,
        show=True,
    )

    results = {
        "dice": float(dice),
        "voxel_precision": float(vox_p),
        "voxel_recall": float(vox_r),
        "lesion_precision": float(les_p),
        "lesion_recall": float(les_r),
        "mean_center_distance": float(mean_cd),
        "median_center_distance": float(median_cd),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(args.threshold),
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nFinal test results:", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"\nAll outputs saved to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
