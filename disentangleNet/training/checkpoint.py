"""
Checkpoint save/load utilities.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 8-9: save_best_checkpoint import)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 368-392: load_checkpoint_with_shape_filter)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_best_checkpoint(
    *,
    model,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    config: dict,
    output_path: Path | str,
) -> None:
    output_path = Path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def load_checkpoint_with_shape_filter(
    model: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    *,
    strict: bool,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load a checkpoint state dict with optional shape-filtered (non-strict) loading.

    When strict=False, keys whose tensor shapes don't match the model are silently skipped.
    Returns (missing_keys, unexpected_keys, skipped_shape_keys).

    Reconstructed from:
    - disentangle_modern_reconstructed/train_reflex_entry.py  (lines 368-392)
    """
    if strict:
        result = model.load_state_dict(checkpoint_state, strict=True)
        return list(result.missing_keys), list(result.unexpected_keys), []

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape_keys = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is None:
            filtered_state[key] = value
            continue
        if tuple(target.shape) != tuple(value.shape):
            skipped_shape_keys.append(key)
            continue
        filtered_state[key] = value

    result = model.load_state_dict(filtered_state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys), skipped_shape_keys


__all__ = ["save_best_checkpoint", "load_checkpoint_with_shape_filter"]
