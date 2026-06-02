from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from disentangleNet.data import DatasetSpec, FacialMotionSequenceDataset
from disentangleNet.losses import build_no_side_eval_metrics, forward_reflex_batch


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
        outputs, x, _, recon_mask, _ = forward_reflex_batch(
            model,
            batch,
            device,
        )
        metrics = build_no_side_eval_metrics(
            outputs=outputs,
            x=x,
            recon_mask=recon_mask,
            loss_weights=loss_weights,
            model=model,
        )

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
