from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from ..data import DatasetSpec, FacialMotionSequenceDataset, subject_split


def build_specs(data_roots: str) -> list[DatasetSpec]:
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


def build_dataloaders(
    train_dataset,
    val_dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
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
    return train_loader, val_loader
