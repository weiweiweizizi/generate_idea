#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire
import torch
from torch.optim import AdamW

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.lq.model.network import DistNet
    from scripts.lq.training import (
        build_dataloaders,
        build_datasets,
        build_specs,
        prepare_train_config,
        run_batch_memory_validation,
        run_epoch,
        save_best_checkpoint,
    )
else:
    from .model.network import DistNet
    from .training import (
        build_dataloaders,
        build_datasets,
        build_specs,
        prepare_train_config,
        run_batch_memory_validation,
        run_epoch,
        save_best_checkpoint,
    )


def train(
    data_roots="data/win20-step20/IMR,data/win20-step20/TT",
    epochs=20,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    seed=42,
    mode="x",
    region="mouth",
    use_difference=True,
    signed_normalize="per_sample",
    val_ratio=0.2,
    group_size=4,
    apply_deleted_filter=True,
    basis_size=119,
    levels="2,3,6",
    hidden_dim=32,
    pool_size=1,
    shared_dim=None,
    private_dim=32,
    private_decoder_hidden_dim=None,
    recon_weight=1.0,
    shared_recon_weight=0.0,
    lq_weight=1.0,
    orth_weight=0.05,
    basis_l1_weight=0.0,
    residual_weight=0.05,
    side_weight=0.5,
    side_cont_weight=None,
    side_disc_weight=None,
    side_semantic_enabled=False,
    side_basis_count=0,
    side_pooling="masked_mean",
    side_loss_weight=0.0,
    dataset_private_weight=0.3,
    dataset_adv_weight=0.3,
    private_residual_weight=0.25,
    private_residual_max_l1=None,
    shared_basis_soft_mixing=False,
    shared_basis_anchor_bias=1.0,
    shared_basis_topk=None,
    grl_lambda=1.0,
    use_dataset_aux=False,
    action_basis_init_path=None,
    lq_commitment_loss_weight=0.1,
    lq_quantization_loss_weight=0.1,
    lq_optimize_values=True,
    quantizer_type="latent_quantize",
    fsq_preserve_symmetry=True,
    basis_orthogonalization="normalize",
    require_basis_init=True,
    validate_batch_memory=True,
    num_workers=0,
    output_dir="outputs/lq",
):
    config = prepare_train_config(locals())
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
        levels=tuple(int(v) for v in config["levels"].split(",")),
        basis_size=config["basis_size"],
        hidden_dim=config["hidden_dim"],
        pool_size=config["pool_size"],
        shared_dim=config["shared_dim"],
        private_dim=config["private_dim"],
        private_decoder_hidden_dim=config["private_decoder_hidden_dim"],
        num_side_classes=3,
        num_dataset_classes=len(specs),
        private_residual_weight=config["private_residual_weight"],
        private_residual_max_l1=config["private_residual_max_l1"],
        shared_basis_soft_mixing=config["shared_basis_soft_mixing"],
        shared_basis_anchor_bias=config["shared_basis_anchor_bias"],
        shared_basis_topk=config["shared_basis_topk"],
        grl_lambda=config["grl_lambda"],
        use_dataset_aux=config["use_dataset_aux"],
        action_basis_init_path=config["action_basis_init_path"],
        lq_commitment_loss_weight=config["lq_commitment_loss_weight"],
        lq_quantization_loss_weight=config["lq_quantization_loss_weight"],
        lq_optimize_values=config["lq_optimize_values"],
        quantizer_type=config["quantizer_type"],
        fsq_preserve_symmetry=config["fsq_preserve_symmetry"],
        basis_orthogonalization=config["basis_orthogonalization"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_weights = {
        "recon": config["recon_weight"],
        "shared_recon": config["shared_recon_weight"],
        "lq": config["lq_weight"],
        "orth": config["orth_weight"],
        "basis_l1": config["basis_l1_weight"],
        "residual": config["residual_weight"],
        "side_cont": config["side_cont_weight"],
        "side_disc": config["side_disc_weight"],
        "side_group": config["side_loss_weight"],
        "dataset_private": config["dataset_private_weight"],
        "dataset_adv": config["dataset_adv_weight"],
    }

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
