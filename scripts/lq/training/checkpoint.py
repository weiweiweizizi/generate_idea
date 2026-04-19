from __future__ import annotations

from pathlib import Path

import torch


def save_best_checkpoint(
    *,
    model,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    config: dict,
    output_path: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config,
        },
        output_path,
    )
