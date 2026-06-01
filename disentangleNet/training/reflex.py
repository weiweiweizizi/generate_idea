"""
Reflex training helpers and entry point.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 80-81: _build_reflex_basis_levels)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 287-314: make_reflex_optimizer)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 317-328: make_reflex_scheduler)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 331-336: freeze_non_side_parameters)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 339-365: load_action_side_linear_init)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 395-445: write_tensorboard_scalars)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 447-493: write_side_confusion_matrix)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 496-1086: train function)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW


def _build_reflex_basis_levels(self_count: int, pair_count: int) -> tuple[int, int]:
    return int(self_count), int(pair_count) * 2


def make_reflex_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    basis_lr_mult: float,
) -> AdamW:
    private_keywords = ("private_head", "private_decoder", "private_adapter")
    basis_keywords = ("reflex_basis_bank",)
    private_params = []
    basis_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in basis_keywords):
            basis_params.append(param)
        elif any(keyword in name for keyword in private_keywords):
            private_params.append(param)
        else:
            other_params.append(param)
    return AdamW(
        [
            {"params": private_params, "lr": lr, "weight_decay": weight_decay, "name": "private"},
            {"params": basis_params, "lr": lr * basis_lr_mult, "weight_decay": weight_decay, "name": "basis"},
            {"params": other_params, "lr": lr, "weight_decay": weight_decay, "name": "other"},
        ],
    )


def make_reflex_scheduler(
    optimizer: AdamW,
    *,
    scheduler_name: str,
    epochs: int,
    min_lr: float,
):
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=float(min_lr))
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name!r}")


def freeze_non_side_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("action_usage_to_side.")
            or name.startswith("side_coeff_to_logits.")
        )


def load_action_side_linear_init(model: torch.nn.Module, init_path: str, device: str) -> None:
    payload = torch.load(init_path, map_location=device)
    state = payload.get("state_dict", payload)
    weight = state["weight"].to(device=device)
    bias = state["bias"].to(device=device)
    target = (
        model.side_coeff_to_logits
        if getattr(model, "side_residual_enabled", False)
        else getattr(model.action_usage_to_side, "net", None)
    )
    if not isinstance(target, torch.nn.Linear):
        raise TypeError("action_side_init_path currently requires a linear action side head")
    if tuple(target.weight.shape) != tuple(weight.shape):
        raise ValueError(
            "action side init shape mismatch: "
            f"target={tuple(target.weight.shape)} init={tuple(weight.shape)}"
        )
    if tuple(target.bias.shape) != tuple(bias.shape):
        raise ValueError(
            "action side init bias mismatch: "
            f"target={tuple(target.bias.shape)} init={tuple(bias.shape)}"
        )
    with torch.no_grad():
        target.weight.copy_(weight)
        target.bias.copy_(bias)
    feature_name = payload.get("feature_name", "unknown")
    print(f"[reflex] loaded action side linear init={init_path} feature={feature_name}")


def build_xw_validation_dataset(*args, **kwargs):
    """Build the XW validation dataset used by reflex-stage evaluation."""
    raise NotImplementedError(
        "build_xw_validation_dataset requires recovery from disentangle_modern_reconstructed "
        "or re-implementation against the current data module."
    )


def validation_dataset_has_side_labels(dataset) -> bool:
    return hasattr(dataset, "specs") and any(
        hasattr(s, "side_label") or hasattr(s, "side") for s in getattr(dataset, "specs", [])
    )


def run_epoch_no_side(model, loader, device, loss_weights):
    """Run one epoch for validation paths that do not use side labels."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["images"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            outputs = model(x, valid_mask=valid_mask)
            loss = outputs.get("loss", torch.tensor(0.0))
            total += loss.item() * x.shape[0]
            count += x.shape[0]
    return {"loss": total / max(count, 1)}


def train_reflex(config_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Full reflex training entry.

    This function orchestrates the complete reflex training flow.
    The core training loop logic is reconstructed from:
    - disentangle_modern_reconstructed/train_reflex_entry.py  (lines 812-1086)

    Note: the actual training loop requires a working forward pass through the model,
    which depends on the model family implementations in models/families/.
    This function provides the scaffolding; the inner loop may need adjustment
    once the full model forward pass is verified.
    """
    raise NotImplementedError(
        "train_reflex requires the full model forward pass to be verified. "
        "Use the original train_reflex_entry.py from disentangle_modern_reconstructed/ "
        "as a reference until models/families/ are fully restored."
    )


__all__ = [
    "build_xw_validation_dataset",
    "freeze_non_side_parameters",
    "load_action_side_linear_init",
    "make_reflex_optimizer",
    "make_reflex_scheduler",
    "run_epoch_no_side",
    "train_reflex",
    "validation_dataset_has_side_labels",
]
