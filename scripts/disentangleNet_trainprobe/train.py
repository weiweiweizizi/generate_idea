#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import torch
from torch.optim import AdamW

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.disentangleNet_trainprobe.model.distnet import DistNet
from scripts.disentangleNet_trainprobe.training import (
    build_dataloaders,
    build_datasets,
    build_fold_manifest,
    build_specs,
    prepare_train_config,
    run_batch_memory_validation,
    run_epoch,
    save_best_checkpoint,
)

DEFAULT_X_ACTION_BASIS_INIT_PATH = "scripts/lq/init_basis/basis_x_full.npy"
DEFAULT_X_SIDE_BASIS_INIT_PATH = "scripts/lq/init_basis/basis_x_full.npy"
DEFAULT_Y_ACTION_BASIS_INIT_PATH = "scripts/lq/init_basis/basis_y_full.npy"
DEFAULT_Y_SIDE_BASIS_INIT_PATH = "scripts/lq/init_basis/basis_y_full.npy"
DEFAULT_X_OUTPUT_DIR = "outputs/disentangleNet_trainprobe/v32_tri_region_masked_x_win20_e50"
DEFAULT_Y_OUTPUT_DIR = "outputs/disentangleNet_trainprobe/v32_tri_region_masked_y_win20_e50"

V31_FIXED_CONFIG = {
    "mode": "x",
    "region": "full",
    "use_difference": True,
    "signed_normalize": "per_sample",
    "group_size": 4,
    "apply_deleted_filter": True,
    "basis_size": 341,
    "levels": "2,6",
    "hidden_dim": 32,
    "pool_size": 1,
    "early_branch_factorization": True,
    "free_pool_size": 2,
    "side_pool_size": 2,
    "private_pool_size": 1,
    "free_z_dim": None,
    "side_z_dim": 32,
    "private_adapter_enabled": False,
    "shared_dim": None,
    "private_dim": 32,
    "private_decoder_hidden_dim": None,
    "recon_weight": 1.0,
    "shared_recon_weight": 1.0,
    "lq_weight": 10.0,
    "orth_weight": 0.1,
    "basis_l1_weight": 1.0,
    "residual_weight": 0.02,
    # v31 uses side-group supervision, so frame-level side losses stay disabled.
    "side_weight": 0.0,
    "side_cont_weight": 0.0,
    "side_disc_weight": 0.0,
    "target_label_mode": "side",
    "num_side_classes": 3,
    "side_semantic_enabled": True,
    "side_basis_count": 3,
    "side_pooling": "tri_region_contrast",
    "side_loss_weight": 0.0,
    "mouth_group_side_loss_weight": 0.45,
    "mouth_cross_group_side_loss_weight": 0.45,
    "other_group_side_loss_weight": 0.10,
    "side_subspace_dim": None,
    "side_free_frame_qr": False,
    "subspace_orth_weight": 0.0,
    "free_side_adv_weight": 0.0,
    "free_side_grl_lambda": 1.0,
    "severity_loss_weight": 0.0,
    "dataset_private_weight": 0.0,
    "dataset_adv_weight": 0.0,
    "private_residual_weight": 0.05,
    "private_residual_max_l1": 0.5,
    "shared_basis_soft_mixing": True,
    "shared_basis_anchor_bias": 2.0,
    "shared_basis_topk": 2,
    "grl_lambda": 1.0,
    "use_dataset_aux": False,
    "lq_commitment_loss_weight": 0.1,
    "lq_quantization_loss_weight": 0.1,
    "lq_optimize_values": True,
    "quantizer_type": "residual_fsq",
    "fsq_preserve_symmetry": True,
    "basis_orthogonalization": "joint_global_qr",
    "discrete_side_loss_enabled": False,
    "require_basis_init": True,
}


def resolve_default_action_basis_init_path(mode: str) -> str:
    if mode == "x":
        return DEFAULT_X_ACTION_BASIS_INIT_PATH
    if mode == "y":
        return DEFAULT_Y_ACTION_BASIS_INIT_PATH
    raise ValueError(f"Unsupported mode: {mode!r}")


def resolve_default_side_basis_init_path(mode: str) -> str:
    if mode == "x":
        return DEFAULT_X_SIDE_BASIS_INIT_PATH
    if mode == "y":
        return DEFAULT_Y_SIDE_BASIS_INIT_PATH
    raise ValueError(f"Unsupported mode: {mode!r}")


def resolve_default_output_dir(mode: str) -> str:
    if mode == "x":
        return DEFAULT_X_OUTPUT_DIR
    if mode == "y":
        return DEFAULT_Y_OUTPUT_DIR
    raise ValueError(f"Unsupported mode: {mode!r}")


def parse_levels(levels) -> tuple[int, ...]:
    if isinstance(levels, str):
        return tuple(int(v) for v in levels.split(",") if str(v).strip())
    if isinstance(levels, (tuple, list)):
        return tuple(int(v) for v in levels)
    raise TypeError(f"Unsupported levels value: {levels!r}")


def build_v31_config(runtime_config: dict) -> dict:
    config = {
        **V31_FIXED_CONFIG,
        **runtime_config,
    }
    return prepare_train_config(config)


def build_loss_weights(config: dict) -> dict[str, float]:
    return {
        "recon": config["recon_weight"],
        "shared_recon": config["shared_recon_weight"],
        "lq": config["lq_weight"],
        "orth": config["orth_weight"],
        "basis_l1": config["basis_l1_weight"],
        "residual": config["residual_weight"],
        "mouth_side_group": config["mouth_group_side_loss_weight"],
        "mouth_cross_side_group": config["mouth_cross_group_side_loss_weight"],
        "other_side_group": config["other_group_side_loss_weight"],
        "severity_group": 0.0,
        "subspace_orth": 0.0,
        "free_side_adv": 0.0,
        "dataset_private": 0.0,
        "dataset_adv": 0.0,
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_metrics_history(
    path: Path,
    *,
    epoch: int,
    train_loss_metrics: dict,
    train_probe_metrics: dict,
    val_loss_metrics: dict,
    val_probe_metrics: dict,
) -> None:
    row = {
        "epoch": int(epoch),
        "train_loss_metrics": train_loss_metrics,
        "train_probe_metrics": train_probe_metrics,
        "val_loss_metrics": val_loss_metrics,
        "val_probe_metrics": val_probe_metrics,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _train_single_fold(
    *,
    config: dict,
    output_dir: Path,
    num_folds: int,
    fold_index: int | None,
):
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    specs = build_specs(config["data_roots"])
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
        num_folds=num_folds,
        fold_index=fold_index,
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = DistNet(
        levels=parse_levels(config["levels"]),
        basis_size=config["basis_size"],
        hidden_dim=config["hidden_dim"],
        pool_size=config["pool_size"],
        early_branch_factorization=config["early_branch_factorization"],
        free_pool_size=config["free_pool_size"],
        side_pool_size=config["side_pool_size"],
        private_pool_size=config["private_pool_size"],
        free_z_dim=config["free_z_dim"],
        side_z_dim=config["side_z_dim"],
        private_adapter_enabled=config["private_adapter_enabled"],
        side_basis_init_path=config["side_basis_init_path"],
        shared_dim=config["shared_dim"],
        private_dim=config["private_dim"],
        private_decoder_hidden_dim=config["private_decoder_hidden_dim"],
        num_side_classes=config["num_side_classes"],
        num_severity_classes=3,
        num_dataset_classes=len(specs),
        target_label_mode=config["target_label_mode"],
        private_residual_weight=config["private_residual_weight"],
        private_residual_max_l1=config["private_residual_max_l1"],
        shared_basis_soft_mixing=config["shared_basis_soft_mixing"],
        shared_basis_anchor_bias=config["shared_basis_anchor_bias"],
        shared_basis_topk=config["shared_basis_topk"],
        side_semantic_enabled=config["side_semantic_enabled"],
        side_basis_count=config["side_basis_count"],
        side_pooling=config["side_pooling"],
        side_subspace_dim=config["side_subspace_dim"],
        side_free_frame_qr=config["side_free_frame_qr"],
        free_side_grl_lambda=config["free_side_grl_lambda"],
        grl_lambda=config["grl_lambda"],
        use_dataset_aux=config["use_dataset_aux"],
        action_basis_init_path=config["action_basis_init_path"],
        lq_commitment_loss_weight=config["lq_commitment_loss_weight"],
        lq_quantization_loss_weight=config["lq_quantization_loss_weight"],
        lq_optimize_values=config["lq_optimize_values"],
        quantizer_type=config["quantizer_type"],
        fsq_preserve_symmetry=config["fsq_preserve_symmetry"],
        basis_orthogonalization=config["basis_orthogonalization"],
        discrete_side_loss_enabled=config["discrete_side_loss_enabled"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_weights = build_loss_weights(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_history_path = output_dir / "metrics_history.jsonl"
    if metrics_history_path.exists():
        metrics_history_path.unlink()

    if config["validate_batch_memory"]:
        run_batch_memory_validation(model, train_loader, device, optimizer, loss_weights)

    best_val = float("inf")
    best_epoch = None
    for epoch in range(1, config["epochs"] + 1):
        train_loss_metrics, train_probe_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            loss_weights,
            train=True,
        )
        with torch.no_grad():
            val_loss_metrics, val_probe_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer,
                loss_weights,
                train=False,
            )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss_metrics} "
            f"train_probe={train_probe_metrics} "
            f"val_loss={val_loss_metrics} "
            f"val_probe={val_probe_metrics}"
        )
        append_metrics_history(
            metrics_history_path,
            epoch=epoch,
            train_loss_metrics=train_loss_metrics,
            train_probe_metrics=train_probe_metrics,
            val_loss_metrics=val_loss_metrics,
            val_probe_metrics=val_probe_metrics,
        )

        if val_loss_metrics["loss"] < best_val:
            best_val = val_loss_metrics["loss"]
            best_epoch = epoch
            save_best_checkpoint(
                model=model,
                epoch=epoch,
                train_loss_metrics=train_loss_metrics,
                val_loss_metrics=val_loss_metrics,
                train_probe_metrics=train_probe_metrics,
                val_probe_metrics=val_probe_metrics,
                config=config,
                output_path=output_dir / "best.pt",
            )

    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(output_dir / "best.pt"),
        "metrics_history_path": str(metrics_history_path),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
    }


def train(
    data_roots="data/win20-step20/IMR,data/win20-step20/TT",
    mode="x",
    epochs=50,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    seed=42,
    val_ratio=0.2,
    action_basis_init_path=None,
    side_basis_init_path=None,
    validate_batch_memory=True,
    num_workers=0,
    output_dir=None,
    private_residual_weight=0.05,
    private_residual_max_l1=0.5,
    num_folds=1,
    fold_index=None,
    run_all_folds=False,
):
    resolved_mode = str(mode)
    resolved_action_basis_init_path = (
        action_basis_init_path
        if action_basis_init_path is not None
        else resolve_default_action_basis_init_path(resolved_mode)
    )
    resolved_side_basis_init_path = (
        side_basis_init_path
        if side_basis_init_path is not None
        else resolve_default_side_basis_init_path(resolved_mode)
    )
    resolved_output_dir = (
        output_dir if output_dir is not None else resolve_default_output_dir(resolved_mode)
    )
    config = build_v31_config(
        {
            "data_roots": data_roots,
            "mode": resolved_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "val_ratio": val_ratio,
            "action_basis_init_path": resolved_action_basis_init_path,
            "side_basis_init_path": resolved_side_basis_init_path,
            "validate_batch_memory": validate_batch_memory,
            "num_workers": num_workers,
            "output_dir": resolved_output_dir,
            "private_residual_weight": private_residual_weight,
            "private_residual_max_l1": private_residual_max_l1,
            "num_folds": int(num_folds),
            "fold_index": None if fold_index is None else int(fold_index),
        }
    )
    resolved_num_folds = int(num_folds)
    if resolved_num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {resolved_num_folds}")

    specs = build_specs(config["data_roots"])
    if run_all_folds and resolved_num_folds == 1:
        raise ValueError("run_all_folds requires num_folds > 1")

    if run_all_folds:
        output_root = Path(config["output_dir"])
        output_root.mkdir(parents=True, exist_ok=True)
        fold_manifest = build_fold_manifest(
            specs,
            num_folds=resolved_num_folds,
            seed=config["seed"],
        )
        fold_summaries = []
        for current_fold_index in range(resolved_num_folds):
            fold_output_dir = output_root / f"fold_{current_fold_index}"
            fold_config = {
                **config,
                "num_folds": resolved_num_folds,
                "fold_index": int(current_fold_index),
                "output_dir": str(fold_output_dir),
            }
            current_fold_entry = fold_manifest["folds"][current_fold_index]
            subject_split_payload = {
                "fold_index": int(current_fold_index),
                "num_folds": resolved_num_folds,
                "datasets": current_fold_entry["datasets"],
            }
            write_json(fold_output_dir / "subject_split.json", subject_split_payload)
            write_json(fold_output_dir / "train_config.json", fold_config)
            fold_summary = _train_single_fold(
                config=fold_config,
                output_dir=fold_output_dir,
                num_folds=resolved_num_folds,
                fold_index=current_fold_index,
            )
            fold_summary["fold_index"] = int(current_fold_index)
            fold_summaries.append(fold_summary)
            current_fold_entry["output_dir"] = str(fold_output_dir)
            current_fold_entry["checkpoint_path"] = str(fold_output_dir / "best.pt")

        write_json(output_root / "fold_manifest.json", fold_manifest)
        summary = {
            "output_root": str(output_root),
            "num_folds": resolved_num_folds,
            "folds": fold_summaries,
            "fold_manifest_path": str(output_root / "fold_manifest.json"),
        }
        write_json(output_root / "kfold_summary.json", summary)
        return summary

    if resolved_num_folds > 1:
        resolved_fold_index = int(0 if fold_index is None else fold_index)
        output_dir = Path(config["output_dir"]) / f"fold_{resolved_fold_index}"
        config = {
            **config,
            "num_folds": resolved_num_folds,
            "fold_index": resolved_fold_index,
            "output_dir": str(output_dir),
        }
        fold_manifest = build_fold_manifest(
            specs,
            num_folds=resolved_num_folds,
            seed=config["seed"],
        )
        current_fold_entry = fold_manifest["folds"][resolved_fold_index]
        write_json(
            output_dir / "subject_split.json",
            {
                "fold_index": resolved_fold_index,
                "num_folds": resolved_num_folds,
                "datasets": current_fold_entry["datasets"],
            },
        )
        write_json(output_dir / "train_config.json", config)
        return _train_single_fold(
            config=config,
            output_dir=output_dir,
            num_folds=resolved_num_folds,
            fold_index=resolved_fold_index,
        )

    output_dir = Path(config["output_dir"])
    write_json(output_dir / "train_config.json", config)
    return _train_single_fold(
        config=config,
        output_dir=output_dir,
        num_folds=1,
        fold_index=None,
    )


if __name__ == "__main__":
    fire.Fire(train)
