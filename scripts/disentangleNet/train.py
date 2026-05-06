#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire
import torch
from torch.optim import AdamW

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.disentangleNet.model.distnet import DistNet
from scripts.disentangleNet.training import (
    build_dataloaders,
    build_datasets,
    build_specs,
    prepare_train_config,
    run_batch_memory_validation,
    run_epoch,
    save_best_checkpoint,
)

DEFAULT_ACTION_BASIS_INIT_PATH = "scripts/disentangleNet/init_basis/basis_x_shared_2_6.npy"
DEFAULT_SIDE_BASIS_INIT_PATH = "scripts/disentangleNet/init_basis/basis_x_side_from_level2.npy"
DEFAULT_OUTPUT_DIR = "outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50"

V31_FIXED_CONFIG = {
    "action_basis_init_path": DEFAULT_ACTION_BASIS_INIT_PATH,
    "side_basis_init_path": DEFAULT_SIDE_BASIS_INIT_PATH,
    "mode": "x",
    "region": "mouth",
    "use_difference": True,
    "signed_normalize": "per_sample",
    "group_size": 4,
    "apply_deleted_filter": True,
    "basis_size": 119,
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
    "side_pooling": "fixed_region2_contrast",
    "side_loss_weight": 0.3,
    "group_side_loss_weight": 0.3,
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
        "side_cont": config["side_cont_weight"],
        "side_disc": config["side_disc_weight"],
        "side_group": config["group_side_loss_weight"],
        "severity_group": 0.0,
        "subspace_orth": 0.0,
        "free_side_adv": 0.0,
        "dataset_private": 0.0,
        "dataset_adv": 0.0,
    }


def train(
    data_roots="data/win20-step20/IMR,data/win20-step20/TT",
    epochs=50,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    seed=42,
    val_ratio=0.2,
    action_basis_init_path=DEFAULT_ACTION_BASIS_INIT_PATH,
    side_basis_init_path=DEFAULT_SIDE_BASIS_INIT_PATH,
    validate_batch_memory=True,
    num_workers=0,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    config = build_v31_config(
        {
            "data_roots": data_roots,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "val_ratio": val_ratio,
            "action_basis_init_path": action_basis_init_path,
            "side_basis_init_path": side_basis_init_path,
            "validate_batch_memory": validate_batch_memory,
            "num_workers": num_workers,
            "output_dir": output_dir,
        }
    )
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

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["validate_batch_memory"]:
        run_batch_memory_validation(model, train_loader, device, optimizer, loss_weights)

    best_val = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, loss_weights, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer, loss_weights, train=False)

        print(f"[epoch {epoch}] train={train_metrics} val={val_metrics}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_best_checkpoint(
                model=model,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                config=config,
                output_path=output_dir / "best.pt",
            )


if __name__ == "__main__":
    fire.Fire(train)
