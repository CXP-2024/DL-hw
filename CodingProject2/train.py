import logging
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    ToDtype,
)

from tqdm import tqdm

from datasets import TinyImageNetDataset
from evaluate import DATASET_ROOT, DEVICE, TRANSFORM
from modules import CustomModel


def _add_time_axis(ax, epochs, elapsed_min) -> None:
    """Add a secondary x-axis showing elapsed time in minutes."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Show a few time ticks aligned to epoch positions
    n = len(elapsed_min)
    step = max(1, n // 5)
    tick_idx = list(range(0, n, step))
    if (n - 1) not in tick_idx:
        tick_idx.append(n - 1)
    ax2.set_xticks([epochs[i] for i in tick_idx])
    ax2.set_xticklabels([f"{elapsed_min[i]:.1f}m" for i in tick_idx])
    ax2.set_xlabel("Elapsed Time")


def _save_plots(history: dict[str, list[float]], best_acc: float) -> None:
    """Save training/validation curves to figures/ directory."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    elapsed_min = history.get("elapsed_min", [])
    has_time = len(elapsed_min) == len(epochs)
    total_time = f" ({elapsed_min[-1]:.1f} min)" if has_time else ""

    # --- Figure 1: Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss Curve{total_time}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if has_time:
        _add_time_axis(ax, epochs, elapsed_min)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Accuracy curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc")
    ax.axhline(y=best_acc * 100, color="r", linestyle="--", alpha=0.5, label=f"Best Val: {best_acc:.2%}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Training & Validation Accuracy{total_time}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if has_time:
        _add_time_axis(ax, epochs, elapsed_min)
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Learning Rate Schedule{total_time}")
    ax.grid(True, alpha=0.3)
    if has_time:
        _add_time_axis(ax, epochs, elapsed_min)
    fig.tight_layout()
    fig.savefig(fig_dir / "lr_schedule.png", dpi=150)
    plt.close(fig)

    logging.info(f"Training curves saved to {fig_dir}/")


def train(model: CustomModel, dataset: TinyImageNetDataset) -> None:
    # YOUR CODE BEGIN.

    # ======================== Hyperparameters ========================
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LR = 0.1
    WEIGHT_DECAY = 5e-4
    WARMUP_EPOCHS = 5
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    MAX_TRAIN_MINUTES = 100

    device_type = DEVICE.type
    use_amp = device_type == "cuda"

    # ======================== Data Augmentation ========================
    train_transform = Compose(
        [
            ToDtype(torch.float32, scale=True),
            RandomCrop(64, padding=8, padding_mode="reflect"),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(p=0.25),
        ]
    )

    train_dataset = TinyImageNetDataset(
        DATASET_ROOT, "train", transform=train_transform
    )
    val_dataset = TinyImageNetDataset(DATASET_ROOT, "val", transform=TRANSFORM)

    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=num_workers > 0,
    )

    # ======================== Loss / Optimizer / Scheduler ========================
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler (no-op when use_amp=False)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ======================== Mixup helper ========================
    def mixup_data(
        x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample().item()
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def mixup_criterion(
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

    # ======================== History (for plotting) ========================
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "elapsed_min": [],
    }

    # ======================== Training Loop ========================
    best_acc = 0.0
    best_state: dict[str, torch.Tensor] = {}
    start_time = time.time()

    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        elapsed = (time.time() - start_time) / 60.0
        if elapsed > MAX_TRAIN_MINUTES:
            logging.info(
                f"Time limit reached ({elapsed:.1f} min) at epoch {epoch}. Stopping."
            )
            break

        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False
        )
        for images, labels in batch_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(mixed_images)
                loss = mixup_criterion(outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (
                lam * preds.eq(targets_a).sum().item()
                + (1.0 - lam) * preds.eq(targets_b).sum().item()
            )

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            train=f"{train_acc:.2%}",
            val=f"{val_acc:.2%}",
            best=f"{max(best_acc, val_acc):.2%}",
            lr=f"{optimizer.param_groups[0]['lr']:.5f}",
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["elapsed_min"].append((time.time() - start_time) / 60.0)

        # Track best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model weights before returning
    if best_state:
        model.load_state_dict(best_state)

    logging.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")

    # ======================== Save training curves ========================
    _save_plots(history, best_acc)

    # YOUR CODE END.


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path

    dataset = TinyImageNetDataset(DATASET_ROOT, "train", transform=TRANSFORM)

    model = CustomModel().to(DEVICE)

    if sum(p.numel() for p in model.parameters()) > 20_000_000:
        logging.error("Model has more than 20 million parameters")
        return

    train(model, dataset)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    logging.info(f"Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
