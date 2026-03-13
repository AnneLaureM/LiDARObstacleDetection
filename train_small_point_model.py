#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


NUM_CLASSES = 5  # Background, Antenna, Cable, Electric Pole, Wind Turbine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small point-wise segmentation model.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SegmentationDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True)

        # Prefer "features"; fall back to xyz if needed
        if "features" in data:
            x = data["features"].astype(np.float32)
        elif "xyz" in data:
            x = data["xyz"].astype(np.float32)
        else:
            raise KeyError(f"{self.files[idx]} missing 'features' and 'xyz'")

        if "labels" not in data:
            raise KeyError(f"{self.files[idx]} missing 'labels'")
        y = data["labels"].astype(np.int64)

        if x.ndim != 2:
            raise ValueError(f"{self.files[idx]}: features must be 2D, got {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"{self.files[idx]}: labels must be 1D, got {y.shape}")

        # Hard safety alignment
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        return torch.from_numpy(x), torch.from_numpy(y)


def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pads variable-length point clouds to the maximum number of points in the batch.
    Returns:
      xs:   [B, Nmax, C]
      ys:   [B, Nmax]
      mask: [B, Nmax] True for valid points
    """
    xs_list, ys_list = zip(*batch)

    lengths = [min(x.shape[0], y.shape[0]) for x, y in zip(xs_list, ys_list)]
    feat_dim = xs_list[0].shape[1]
    max_len = max(lengths)

    batch_size = len(batch)
    xs = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    ys = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs_list, ys_list)):
        n = min(x.shape[0], y.shape[0])
        xs[i, :n] = x[:n]
        ys[i, :n] = y[:n]
        mask[i, :n] = True

    return xs, ys, mask


class SmallPointMLP(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        return self.net(x)  # [B, N, num_classes]


def compute_class_weights(dataset: SegmentationDataset) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        yy = y.numpy()
        binc = np.bincount(yy, minlength=NUM_CLASSES)
        counts += binc

    counts = np.maximum(counts, 1)
    freqs = counts / counts.sum()

    # Inverse frequency with clipping for stability
    weights = 1.0 / np.sqrt(freqs)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_points = 0
    correct = 0

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(x)  # [B, N, K]

            logits_flat = logits[mask]
            y_flat = y[mask]

            loss = criterion(logits_flat, y_flat)

            preds = logits_flat.argmax(dim=-1)
            correct += (preds == y_flat).sum().item()
            total_points += y_flat.numel()
            total_loss += loss.item() * y_flat.numel()

    mean_loss = total_loss / max(total_points, 1)
    accuracy = correct / max(total_points, 1)
    return mean_loss, accuracy


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = SegmentationDataset(args.dataset_dir / "train")
    val_ds = SegmentationDataset(args.dataset_dir / "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPointMLP(in_dim=4, num_classes=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    param_count = count_parameters(model)
    print(f"Using device={device}, parameters={param_count:,}")
    print(f"Class weights={class_weights.detach().cpu().numpy().round(3).tolist()}")

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_points = 0
        running_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y, mask in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)           # [B, N, K]
            logits_flat = logits[mask]  # [M, K]
            y_flat = y[mask]            # [M]

            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()

            preds = logits_flat.argmax(dim=-1)
            running_correct += (preds == y_flat).sum().item()
            running_points += y_flat.numel()
            running_loss += loss.item() * y_flat.numel()

            pbar.set_postfix(
                loss=f"{running_loss / max(running_points, 1):.4f}",
                acc=f"{running_correct / max(running_points, 1):.4f}",
            )

        train_loss = running_loss / max(running_points, 1)
        train_acc = running_correct / max(running_points, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_info)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "in_dim": 4,
            "class_weights": class_weights.detach().cpu(),
            "history": history,
        }

        torch.save(ckpt, args.output_dir / "last_model.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, args.output_dir / "best_model.pt")

    summary = {
        "device": str(device),
        "parameters": param_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved best model to: {args.output_dir / 'best_model.pt'}")
    print(f"Saved summary to:    {args.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()