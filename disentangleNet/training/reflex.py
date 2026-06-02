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

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from disentangleNet.io_utils import save_json
from disentangleNet.losses import attach_region_laplacian, build_reflex_loss_weights
from disentangleNet.models import build_model, build_modular_model_config
from disentangleNet.models.side_heads import build_mirror_perm
from disentangleNet.training.checkpoint import save_best_checkpoint
from disentangleNet.training.config import prepare_train_config
from disentangleNet.training.data import build_dataloaders, build_datasets, build_specs
from disentangleNet.training.engine import run_batch_memory_validation, run_epoch
from disentangleNet.training.validation import (
    build_xw_validation_dataset as _build_xw_validation_dataset,
    run_epoch_no_side as _run_epoch_no_side,
    validation_dataset_has_side_labels as _validation_dataset_has_side_labels,
)


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
    basis_keywords = ("reflex_basis_bank", "shared_basis_runtime", "lowrank_basis_bank")
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

    return _build_xw_validation_dataset(*args, **kwargs)


def validation_dataset_has_side_labels(dataset) -> bool:
    return _validation_dataset_has_side_labels(dataset)


def run_epoch_no_side(model, loader, device, loss_weights):
    """Run one epoch for validation paths that do not use side labels."""
    return _run_epoch_no_side(model, loader, device, loss_weights)


def _merge_runtime_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _write_tensorboard_scalars(
    writer: SummaryWriter,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
) -> None:
    scalar_keys = (
        "loss",
        "recon",
        "shared_recon",
        "action_side",
        "action_side_acc",
        "private_side_adv",
        "private_side_acc",
        "v9_freq",
        "lowrank_orth",
        "reflex_orth",
        "shared_coeff_l1",
        "side_coeff_l1",
        "side_private_orth",
        "lap_basis",
        "lap_side_basis",
        "lap_recon",
    )
    for key in scalar_keys:
        if key in train_metrics:
            writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
        if key in val_metrics:
            writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
    for group_index, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get("name", f"group{group_index}")
        writer.add_scalar(f"lr/{group_name}", param_group["lr"], epoch)


def _write_side_confusion_matrix(
    writer: SummaryWriter,
    *,
    epoch: int,
    model,
    val_loader,
    device: str,
) -> None:
    was_training = model.training
    model.eval()
    confusion = torch.zeros(3, 3, dtype=torch.long)
    with torch.no_grad():
        for batch in val_loader:
            x = batch["images"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            side_labels = batch["side_label"].to(device)
            group_valid_mask = valid_mask.any(dim=1)
            if not group_valid_mask.any():
                continue
            outputs = model(x, side_labels=side_labels, valid_mask=valid_mask)
            logits = outputs.get("group_action_logits")
            if logits is None:
                continue
            preds = logits[group_valid_mask].argmax(dim=1).cpu()
            labels = side_labels[group_valid_mask].cpu()
            for label, pred in zip(labels.tolist(), preds.tolist()):
                if 0 <= label < 3 and 0 <= pred < 3:
                    confusion[label, pred] += 1
    model.train(was_training)

    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(confusion.numpy(), cmap="Blues")
    class_names = ["left_aff", "bilateral", "right_aff"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(f"val side confusion ep{epoch}", fontsize=9)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(int(confusion[row, col])), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    writer.add_figure("val/side_confusion_matrix", fig, global_step=epoch)
    plt.close(fig)


def train_reflex(config_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Full reflex training entry.

    This function orchestrates the PhaseAB reflex training flow for the
    recovered mouth/x configuration path.

    TODO(recovery): broaden this beyond the current PhaseAB path after the
    remaining model family branches and laplacian regularizer are restored.
    """
    if config_path is None:
        raise ValueError("train_reflex currently requires config_path for the recovered PhaseAB path")

    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    runtime_config = _merge_runtime_overrides(raw_config, kwargs)
    config = prepare_train_config(runtime_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mirror_perm = build_mirror_perm(config["ordered_indices_path"]).to(device)

    specs = build_specs(config["data_roots"])
    val_data_roots = config.get("val_data_roots")
    if val_data_roots is not None:
        train_dataset, _ = build_datasets(
            specs,
            mode=config["mode"],
            region=config["region"],
            use_difference=config["use_difference"],
            signed_normalize=config["signed_normalize"],
            val_ratio=0.0,
            seed=config["seed"],
            group_size=config["group_size"],
            apply_deleted_filter=config["apply_deleted_filter"],
            static_side_input_enabled=bool(config.get("static_side_input_enabled", False)),
            ordered_indices_path=config.get("ordered_indices_path"),
        )
        val_dataset = build_xw_validation_dataset(
            val_data_roots=val_data_roots,
            config=config,
        )
    else:
        train_dataset, val_dataset = build_datasets(
            specs,
            mode=config["mode"],
            region=config["region"],
            use_difference=config["use_difference"],
            signed_normalize=config["signed_normalize"],
            val_ratio=config["val_ratio"],
            seed=config["seed"],
            group_size=config["group_size"],
            apply_deleted_filter=config["apply_deleted_filter"],
            static_side_input_enabled=bool(config.get("static_side_input_enabled", False)),
            ordered_indices_path=config.get("ordered_indices_path"),
        )

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    modular_model_config = build_modular_model_config(config, mirror_perm=mirror_perm)
    model = build_model(modular_model_config, num_dataset_classes=len(specs))
    model = model.to(device)
    attach_region_laplacian(
        model,
        ordered_indices_path=config["ordered_indices_path"],
        region=config["region"],
        topology_source=config.get("laplacian_topology_source"),
    )

    init_checkpoint_path = config.get("init_checkpoint_path")
    if init_checkpoint_path is not None:
        checkpoint = torch.load(init_checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        print(
            "[reflex] loaded init checkpoint="
            f"{init_checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}"
        )

    action_side_init_path = config.get("action_side_init_path")
    if action_side_init_path is not None:
        load_action_side_linear_init(model, action_side_init_path, device)
    if bool(config.get("freeze_non_side_for_action_probe", False)):
        freeze_non_side_parameters(model)

    optimizer = make_reflex_optimizer(
        model,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        basis_lr_mult=float(config["basis_lr_mult"]),
    )
    scheduler = make_reflex_scheduler(
        optimizer,
        scheduler_name=str(config.get("lr_scheduler", "none")),
        epochs=int(config["epochs"]),
        min_lr=float(config.get("min_lr", 1e-6)),
    )
    loss_weights = build_reflex_loss_weights(
        action_side_weight=float(config.get("action_side_weight", 0.0)),
        basis_l1_weight=float(config.get("basis_l1_weight", 0.0)),
        freq_weight=float(config.get("lowrank_freq_weight", 0.0)),
        lowrank_orth_weight=float(config.get("lowrank_orth_weight", 0.0)),
        residual_weight=float(config.get("residual_weight", 0.0)),
        shared_coeff_l1_weight=float(config.get("shared_coeff_l1_weight", 0.0)),
        side_coeff_l1_weight=float(config.get("side_coeff_l1_weight", 0.0)),
        side_private_orth_weight=float(config.get("side_private_orth_weight", 0.0)),
        private_side_adv_weight=float(config.get("private_side_adv_weight", 0.0)),
        laplacian_basis_weight=float(config.get("laplacian_basis_weight", 0.0)),
        laplacian_side_basis_weight=float(config.get("laplacian_side_basis_weight", 0.0)),
        laplacian_recon_weight=float(config.get("laplacian_recon_weight", 0.0)),
    )

    output_path = Path(str(config["output_dir"]))
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(output_path / "train_config.json", runtime_config)
    save_json(output_path / "train_config_structured.json", config)
    save_json(output_path / "model_config.json", modular_model_config.to_dict())
    save_json(output_path / "loss_weights.json", loss_weights)
    tensorboard_dir = output_path / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    best_metric_name = str(config.get("best_metric", "loss"))
    best_start_epoch = int(config.get("best_start_epoch", 1))
    best_val = float("inf")
    best_epoch = None
    val_has_side = val_data_roots is None or validation_dataset_has_side_labels(val_dataset)

    try:
        if bool(config.get("validate_batch_memory", True)):
            run_batch_memory_validation(model, train_loader, device, optimizer, loss_weights)

        for epoch in range(1, int(config["epochs"]) + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer, loss_weights, train=True)
            with torch.no_grad():
                if val_has_side:
                    val_metrics = run_epoch(model, val_loader, device, optimizer, loss_weights, train=False)
                else:
                    val_metrics = run_epoch_no_side(model, val_loader, device, loss_weights)

            _write_tensorboard_scalars(
                writer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                optimizer=optimizer,
            )
            if bool(config.get("log_confusion_matrix", True)) and val_has_side:
                _write_side_confusion_matrix(
                    writer,
                    epoch=epoch,
                    model=model,
                    val_loader=val_loader,
                    device=device,
                )
            writer.flush()

            print(
                f"[epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.5f} recon={train_metrics['recon']:.5f} "
                f"shared={train_metrics['shared_recon']:.5f} "
                f"action_side={train_metrics.get('action_side', 0.0):.5f} "
                f"acc={train_metrics.get('action_side_acc', 0.0):.3f} | "
                f"val_loss={val_metrics['loss']:.5f} recon={val_metrics['recon']:.5f} "
                f"shared={val_metrics['shared_recon']:.5f} "
                f"action_side={val_metrics.get('action_side', 0.0):.5f} "
                f"acc={val_metrics.get('action_side_acc', 0.0):.3f}"
            )

            if best_metric_name not in val_metrics:
                raise KeyError(f"best_metric={best_metric_name!r} not found in val_metrics")
            current_best = float(val_metrics[best_metric_name])
            if epoch >= best_start_epoch and current_best < best_val:
                best_val = current_best
                best_epoch = epoch
                save_best_checkpoint(
                    model=model,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    config=config,
                    output_path=output_path / "best.pt",
                )
                print(f"[reflex] best saved epoch={epoch} val_{best_metric_name}={best_val:.5f}")
            if scheduler is not None:
                scheduler.step()
    finally:
        writer.close()

    return {
        "output_dir": str(output_path),
        "checkpoint_path": str(output_path / "best.pt"),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
    }


def train(config_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    README-compatible reflex entry point.

    TODO(recovery): once the broader training entry layout is fully stabilized,
    reconcile package-level and module-level training entrypoints so README and
    implementation stay aligned without maintaining a thin alias here.
    """

    return train_reflex(config_path=config_path, **kwargs)


__all__ = [
    "build_xw_validation_dataset",
    "freeze_non_side_parameters",
    "load_action_side_linear_init",
    "make_reflex_optimizer",
    "make_reflex_scheduler",
    "run_epoch_no_side",
    "train",
    "train_reflex",
    "validation_dataset_has_side_labels",
]
