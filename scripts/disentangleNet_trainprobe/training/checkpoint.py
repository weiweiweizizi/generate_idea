from __future__ import annotations

from pathlib import Path

import torch


def save_best_checkpoint(
    *,
    model,
    epoch: int,
    train_loss_metrics: dict,
    val_loss_metrics: dict,
    train_probe_metrics: dict,
    val_probe_metrics: dict,
    config: dict,
    output_path: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss_metrics": train_loss_metrics,
            "val_loss_metrics": val_loss_metrics,
            "train_probe_metrics": train_probe_metrics,
            "val_probe_metrics": val_probe_metrics,
            "config": config,
        },
        output_path,
    )
