#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from datasets import DatasetSpec, FacialMotionDataset, subject_split
from model.network import DistNet


def build_specs(data_roots: str):
    roots = [Path(root).expanduser() for root in data_roots.split(",") if root.strip()]
    specs = []
    for idx, root in enumerate(roots):
        specs.append(DatasetSpec(root=root, dataset_label=idx, dataset_name=root.name))
    return specs


def build_datasets(
    specs,
    *,
    mode: str,
    region: str,
    use_difference: bool,
    signed_normalize: str,
    val_ratio: float,
    seed: int,
):
    train_sets = []
    val_sets = []

    for spec in specs:
        train_subjects, val_subjects = subject_split(spec, val_ratio=val_ratio, seed=seed)
        global_scale = None
        if signed_normalize == "global":
            global_scale = FacialMotionDataset.compute_global_scale(
                spec,
                train_subjects,
                mode=mode,
                region=region,
                use_difference=use_difference,
                seed=seed,
            )

        common = dict(
            mode=mode,
            region=region,
            use_difference=use_difference,
            signed_normalize=signed_normalize,
            global_scale=global_scale,
        )
        train_sets.append(FacialMotionDataset(spec, train_subjects, **common))
        val_sets.append(FacialMotionDataset(spec, val_subjects, **common))

    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def step_model(model, batch, device, loss_weights):
    x = batch["image"].to(device)
    side_labels = batch["side_label"].to(device)
    dataset_labels = batch["dataset_label"].to(device) if model.use_dataset_aux else None

    outputs = model(x, side_labels=side_labels, dataset_labels=dataset_labels)

    recon_loss = F.l1_loss(outputs["reconstructed"], x)
    total_loss = loss_weights["recon"] * recon_loss
    total_loss = total_loss + loss_weights["lq"] * outputs["lq_loss"]
    total_loss = total_loss + loss_weights["orth"] * outputs["orth_loss"]
    total_loss = total_loss + loss_weights["residual"] * outputs["residual_l1"]

    side_loss = outputs["side_loss"]["side_loss"]
    if side_loss is not None:
        total_loss = total_loss + loss_weights["side"] * side_loss

    dataset_private_loss = outputs["dataset_loss"]["private_dataset_loss"]
    if dataset_private_loss is not None:
        total_loss = total_loss + loss_weights["dataset_private"] * dataset_private_loss

    dataset_adv_loss = outputs["dataset_loss"]["shared_dataset_adv_loss"]
    if dataset_adv_loss is not None:
        total_loss = total_loss + loss_weights["dataset_adv"] * dataset_adv_loss

    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "recon": float(recon_loss.detach().cpu()),
        "lq": float(outputs["lq_loss"].detach().cpu()),
        "orth": float(outputs["orth_loss"].detach().cpu()),
        "residual": float(outputs["residual_l1"].detach().cpu()),
    }
    if side_loss is not None:
        metrics["side"] = float(side_loss.detach().cpu())
    if dataset_private_loss is not None:
        metrics["dataset_private"] = float(dataset_private_loss.detach().cpu())
    if dataset_adv_loss is not None:
        metrics["dataset_adv"] = float(dataset_adv_loss.detach().cpu())

    return total_loss, metrics


def run_epoch(model, loader, device, optimizer, loss_weights, train: bool):
    model.train(train)
    total = {}

    for batch in tqdm(loader, leave=False):
        if train:
            optimizer.zero_grad()

        loss, metrics = step_model(model, batch, device, loss_weights)

        if train:
            loss.backward()
            optimizer.step()

        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + value

    denom = max(len(loader), 1)
    return {key: value / denom for key, value in total.items()}


def train(
    data_roots="data/win10-step10/IMR,data/win10-step10/TT",
    epochs=20,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    seed=42,
    mode="x",
    region="mouth",
    use_difference=True,
    signed_normalize="per_sample",
    val_ratio=0.2,
    basis_size=119,
    levels="2,3,6",
    hidden_dim=32,
    private_dim=32,
    recon_weight=1.0,
    lq_weight=1.0,
    orth_weight=0.05,
    residual_weight=0.05,
    side_weight=0.5,
    dataset_private_weight=0.3,
    dataset_adv_weight=0.3,
    private_residual_weight=0.25,
    grl_lambda=1.0,
    use_dataset_aux=False,
    action_basis_init_path=None,
    num_workers=0,
    output_dir="outputs/lq",
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    specs = build_specs(data_roots)
    train_dataset, val_dataset = build_datasets(
        specs,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if region == "mouth" and basis_size != 119:
        raise ValueError("Mouth region expects basis_size=119")
    if region == "full" and basis_size != 341:
        raise ValueError("Full region expects basis_size=341")

    model = DistNet(
        levels=tuple(int(v) for v in levels.split(",")),
        basis_size=basis_size,
        hidden_dim=hidden_dim,
        private_dim=private_dim,
        num_side_classes=3,
        num_dataset_classes=len(specs),
        private_residual_weight=private_residual_weight,
        grl_lambda=grl_lambda,
        use_dataset_aux=use_dataset_aux,
        action_basis_init_path=action_basis_init_path,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_weights = {
        "recon": recon_weight,
        "lq": lq_weight,
        "orth": orth_weight,
        "residual": residual_weight,
        "side": side_weight,
        "dataset_private": dataset_private_weight,
        "dataset_adv": dataset_adv_weight,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, loss_weights, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer, loss_weights, train=False)

        print(f"[epoch {epoch}] train={train_metrics} val={val_metrics}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    fire.Fire(train)
