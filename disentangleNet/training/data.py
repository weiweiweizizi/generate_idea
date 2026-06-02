from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from disentangleNet.data import DatasetSpec, FacialMotionSequenceDataset, subject_split
from disentangleNet.data.io import estimate_global_signed_scale, infer_subject_width, load_metadata


def build_specs(data_roots: str) -> list[DatasetSpec]:
    """Build dataset specifications from a comma-separated roots string."""
    roots = [Path(part.strip()) for part in str(data_roots).split(",") if part.strip()]
    specs = []
    for dataset_label, root in enumerate(roots):
        specs.append(
            DatasetSpec(
                root=root,
                dataset_label=dataset_label,
                dataset_name=root.name,
            )
        )
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
    static_side_input_enabled: bool = False,
    ordered_indices_path: str | None = None,
):
    """
    Build concatenated train/val datasets from dataset specs.

    Recovered fragments suggest that this function:
    - splits subjects with `subject_split`
    - optionally computes a global normalization scale
    - instantiates `FacialMotionSequenceDataset` for train and val
    - returns `(ConcatDataset(train_sets), ConcatDataset(val_sets))`
    """
    train_sets = []
    val_sets = []

    for spec in specs:
        if val_ratio <= 0:
            subject_width = infer_subject_width(spec)
            meta = pd.read_csv(spec.root / "metadata.csv")
            subjects = (
                meta["subj"].astype(str).str.zfill(subject_width).sort_values().unique().tolist()
            )
            train_subjects = subjects
            val_subjects = []
        else:
            train_subjects, val_subjects = subject_split(
                spec,
                val_ratio=val_ratio,
                seed=seed,
            )

        train_global_scale = None
        val_global_scale = None
        if signed_normalize == "global":
            if train_subjects:
                train_global_scale = estimate_global_signed_scale(
                    spec,
                    train_subjects,
                    mode=mode,
                    region=region,
                    use_difference=use_difference,
                    seed=seed,
                    apply_deleted_filter=apply_deleted_filter,
                )
            if val_subjects:
                val_global_scale = estimate_global_signed_scale(
                    spec,
                    val_subjects,
                    mode=mode,
                    region=region,
                    use_difference=use_difference,
                    seed=seed,
                    apply_deleted_filter=apply_deleted_filter,
                )

        train_sets.append(
            FacialMotionSequenceDataset(
                spec=spec,
                subjects=train_subjects,
                mode=mode,
                region=region,
                use_difference=use_difference,
                signed_normalize=signed_normalize,
                global_scale=train_global_scale,
                group_size=group_size,
                apply_deleted_filter=apply_deleted_filter,
                static_side_input_enabled=static_side_input_enabled,
                ordered_indices_path=ordered_indices_path,
            )
        )
        val_sets.append(
            FacialMotionSequenceDataset(
                spec=spec,
                subjects=val_subjects,
                mode=mode,
                region=region,
                use_difference=use_difference,
                signed_normalize=signed_normalize,
                global_scale=val_global_scale,
                group_size=group_size,
                apply_deleted_filter=apply_deleted_filter,
                static_side_input_enabled=static_side_input_enabled,
                ordered_indices_path=ordered_indices_path,
            )
        )

    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def build_dataloaders(
    train_dataset,
    val_dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap train/val datasets into standard PyTorch dataloaders."""
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


__all__ = ["build_specs", "build_datasets", "build_dataloaders"]
