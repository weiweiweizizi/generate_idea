from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from disentangleNet.data import DatasetSpec, FacialMotionSequenceDataset
from disentangleNet.losses import masked_mean


def build_xw_validation_dataset(
    *,
    val_data_roots: str,
    config: dict,
) -> FacialMotionSequenceDataset:
    val_root = Path(val_data_roots)
    val_meta = pd.read_csv(val_root / "metadata.csv")
    val_subjects_raw = sorted(val_meta["subj"].astype(str).unique().tolist())
    val_spec = DatasetSpec(root=val_root, dataset_label=0, dataset_name=val_root.name)
    val_dataset = FacialMotionSequenceDataset(
        spec=val_spec,
        subjects=val_subjects_raw,
        mode=config["mode"],
        region=config["region"],
        use_difference=config["use_difference"],
        signed_normalize=config["signed_normalize"],
        global_scale=None,
        group_size=config["group_size"],
        apply_deleted_filter=config["apply_deleted_filter"],
        static_side_input_enabled=bool(config.get("static_side_input_enabled", False)),
        ordered_indices_path=config.get("ordered_indices_path"),
    )
    val_dataset.subject_width = 0
    val_dataset.meta["subj"] = val_dataset.meta["subj"].str.replace(r"^0+", "", regex=True)
    val_dataset.groups = val_dataset._build_groups()
    return val_dataset


def validation_dataset_has_side_labels(dataset: FacialMotionSequenceDataset) -> bool:
    meta = getattr(dataset, "meta", None)
    if meta is None or "side" not in meta.columns:
        return False
    side_values = pd.to_numeric(meta["side"], errors="coerce")
    valid_values = side_values.dropna()
    if valid_values.empty:
        return False
    return bool((valid_values >= 0).any())


def run_epoch_no_side(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    loss_weights: dict,
) -> dict[str, float]:
    model.eval()
    total: dict[str, float] = {}

    for batch in tqdm(loader, leave=False):
        x = batch["images"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        recon_mask = ~padding_mask
        side_labels = batch["side_label"].to(device)
        static_side_input = batch.get("static_side_input")
        if isinstance(static_side_input, torch.Tensor):
            static_side_input = static_side_input.to(device)

        outputs = model(
            x,
            side_labels=side_labels,
            dataset_labels=None,
            valid_mask=valid_mask,
            static_side_input=static_side_input,
        )

        recon_per = (outputs["reconstructed"] - x).abs().mean(dim=(2, 3, 4))
        shared_per = (outputs["action_reconstruction"] - x).abs().mean(dim=(2, 3, 4))
        recon_loss = masked_mean(recon_per, recon_mask)
        shared_loss = masked_mean(shared_per, recon_mask)

        total_loss = loss_weights.get("recon", 1.0) * recon_loss
        total_loss = total_loss + loss_weights.get("shared_recon", 1.0) * shared_loss

        _orth = outputs.get("lowrank_orth_loss")
        if _orth is not None:
            total_loss = total_loss + loss_weights.get("lowrank_orth", 0.0) * _orth

        _freq = outputs.get("v9_freq_loss")
        if _freq is not None:
            total_loss = total_loss + loss_weights.get("v9_freq", 0.0) * _freq

        _shared_coeff_l1 = outputs.get("shared_coeff_l1")
        if isinstance(_shared_coeff_l1, torch.Tensor):
            total_loss = total_loss + loss_weights.get("shared_coeff_l1", 0.0) * _shared_coeff_l1

        _sc_l1 = outputs.get("side_coeff_l1")
        if isinstance(_sc_l1, torch.Tensor):
            total_loss = total_loss + loss_weights.get("side_coeff_l1", 0.0) * _sc_l1

        metrics: dict[str, float] = {
            "loss": float(total_loss.detach().cpu()),
            "recon": float(recon_loss.detach().cpu()),
            "shared_recon": float(shared_loss.detach().cpu()),
        }
        if _orth is not None:
            metrics["lowrank_orth"] = float(_orth.detach().cpu())
        if _freq is not None:
            metrics["v9_freq"] = float(_freq.detach().cpu())
        if isinstance(_shared_coeff_l1, torch.Tensor):
            metrics["shared_coeff_l1"] = float(_shared_coeff_l1.detach().cpu())
        if isinstance(_sc_l1, torch.Tensor):
            metrics["side_coeff_l1"] = float(_sc_l1.detach().cpu())

        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + value

    denom = max(len(loader), 1)
    return {key: value / denom for key, value in total.items()}


__all__ = [
    "DataLoader",
    "build_xw_validation_dataset",
    "run_epoch_no_side",
    "validation_dataset_has_side_labels",
]
