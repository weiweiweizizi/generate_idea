#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from datasets import DatasetSpec, FacialMotionSequenceDataset, subject_split
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
    group_size: int,
    apply_deleted_filter: bool,
):
    train_sets = []
    val_sets = []

    for spec in specs:
        train_subjects, val_subjects = subject_split(spec, val_ratio=val_ratio, seed=seed)
        global_scale = None
        if signed_normalize == "global":
            global_scale = FacialMotionSequenceDataset.compute_global_scale(
                spec,
                train_subjects,
                mode=mode,
                region=region,
                use_difference=use_difference,
                seed=seed,
                apply_deleted_filter=apply_deleted_filter,
            )

        common = dict(
            mode=mode,
            region=region,
            use_difference=use_difference,
            signed_normalize=signed_normalize,
            global_scale=global_scale,
            group_size=group_size,
            apply_deleted_filter=apply_deleted_filter,
        )
        train_sets.append(FacialMotionSequenceDataset(spec, train_subjects, **common))
        val_sets.append(FacialMotionSequenceDataset(spec, val_subjects, **common))

    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def tensor_memory_mib(tensor: torch.Tensor) -> float:
    """Estimate tensor memory footprint in MiB."""

    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked positions while keeping gradients well-defined."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def step_model(model, batch, device, loss_weights):
    x = batch["images"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    recon_mask = ~padding_mask
    supervision_mask = valid_mask
    side_labels = batch["side_label"].to(device)
    dataset_labels = batch["dataset_label"].to(device) if model.use_dataset_aux else None

    outputs = model(x, side_labels=side_labels, dataset_labels=dataset_labels)

    recon_loss_per_frame = (outputs["reconstructed"] - x).abs().mean(dim=(2, 3, 4))
    recon_loss = masked_mean(recon_loss_per_frame, recon_mask)
    lq_loss = masked_mean(outputs["lq_loss_per_sample"], recon_mask)
    residual_l1 = masked_mean(outputs["residual_l1_per_sample"], recon_mask)

    total_loss = loss_weights["recon"] * recon_loss
    total_loss = total_loss + loss_weights["lq"] * lq_loss
    total_loss = total_loss + loss_weights["orth"] * outputs["orth_loss"]
    total_loss = total_loss + loss_weights["residual"] * residual_l1

    side_loss = outputs["side_loss"]["side_loss_per_sample"]
    side_loss_value = None
    if side_loss is not None:
        side_loss_value = masked_mean(side_loss, supervision_mask)
        total_loss = total_loss + loss_weights["side"] * side_loss_value

    dataset_private_loss = outputs["dataset_loss"]["private_dataset_loss_per_sample"]
    dataset_private_loss_value = None
    if dataset_private_loss is not None:
        dataset_private_loss_value = masked_mean(dataset_private_loss, supervision_mask)
        total_loss = total_loss + loss_weights["dataset_private"] * dataset_private_loss_value

    dataset_adv_loss = outputs["dataset_loss"]["shared_dataset_adv_loss_per_sample"]
    dataset_adv_loss_value = None
    if dataset_adv_loss is not None:
        dataset_adv_loss_value = masked_mean(dataset_adv_loss, supervision_mask)
        total_loss = total_loss + loss_weights["dataset_adv"] * dataset_adv_loss_value

    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "recon": float(recon_loss.detach().cpu()),
        "lq": float(lq_loss.detach().cpu()),
        "orth": float(outputs["orth_loss"].detach().cpu()),
        "residual": float(residual_l1.detach().cpu()),
        "recon_frames": float(recon_mask.sum().detach().cpu()),
        "supervision_frames": float(supervision_mask.sum().detach().cpu()),
    }
    if side_loss_value is not None:
        metrics["side"] = float(side_loss_value.detach().cpu())
    if dataset_private_loss_value is not None:
        metrics["dataset_private"] = float(dataset_private_loss_value.detach().cpu())
    if dataset_adv_loss_value is not None:
        metrics["dataset_adv"] = float(dataset_adv_loss_value.detach().cpu())

    return total_loss, metrics


def run_batch_memory_validation(model, loader, device, optimizer, loss_weights):
    """Run one forward/backward smoke pass and print batch memory info."""

    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Training loader is empty; cannot validate batch memory") from exc

    x = batch["images"]
    valid_mask = batch["valid_mask"]
    padding_mask = batch["padding_mask"]
    print(
        "[memory-check] "
        f"images.shape={tuple(x.shape)} "
        f"valid_mask.shape={tuple(valid_mask.shape)} "
        f"padding_mask.shape={tuple(padding_mask.shape)} "
        f"dtype={x.dtype} "
        f"input_mib={tensor_memory_mib(x):.2f}"
    )

    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        loss, metrics = step_model(model, batch, device, loss_weights)
        loss.backward()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "Batch memory validation failed with OOM. "
                "Try region=mouth, smaller group_size, or smaller batch_size."
            ) from exc
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[memory-check] smoke_metrics={metrics}")
    if device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[memory-check] peak_cuda_mib={peak_mib:.2f}")


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
    require_basis_init=True,
    validate_batch_memory=True,
    num_workers=0,
    output_dir="outputs/lq",
):
    config = locals().copy()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if require_basis_init and not action_basis_init_path:
        raise ValueError(
            "action_basis_init_path is required for this training stage. "
            "Set require_basis_init=False to override."
        )
    if action_basis_init_path is None:
        print("[warning] Training without basis initialization.")
    elif not Path(action_basis_init_path).exists():
        raise FileNotFoundError(action_basis_init_path)

    specs = build_specs(data_roots)
    train_dataset, val_dataset = build_datasets(
        specs,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
        group_size=group_size,
        apply_deleted_filter=apply_deleted_filter,
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

    if validate_batch_memory:
        run_batch_memory_validation(model, train_loader, device, optimizer, loss_weights)

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
                    "config": config,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    fire.Fire(train)
