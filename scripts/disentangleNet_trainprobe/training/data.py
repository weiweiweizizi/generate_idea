from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from ..data import (
    DatasetSpec,
    FacialMotionSequenceDataset,
    build_subject_folds,
    subject_kfold_split,
    subject_split,
)


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
    num_folds: int = 1,
    fold_index: int | None = None,
):
    train_sets = []
    val_sets = []

    for spec in specs:
        if num_folds > 1:
            if fold_index is None:
                raise ValueError("fold_index must be provided when num_folds > 1")
            train_subjects, val_subjects = subject_kfold_split(
                spec,
                num_folds=num_folds,
                fold_index=fold_index,
                seed=seed,
            )
        else:
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


def build_fold_manifest(
    specs,
    *,
    num_folds: int,
    seed: int,
) -> dict:
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    fold_rows = []
    all_subject_entries = []
    for fold_index in range(num_folds):
        fold_entry = {
            "fold_index": int(fold_index),
            "datasets": [],
        }
        for spec in specs:
            folds = build_subject_folds(spec, num_folds=num_folds, seed=seed)
            val_subjects = folds[fold_index]
            train_subjects = sorted(
                subject
                for current_fold_idx, fold_subjects in enumerate(folds)
                if current_fold_idx != fold_index
                for subject in fold_subjects
            )
            fold_entry["datasets"].append(
                {
                    "dataset_name": spec.dataset_name,
                    "dataset_label": int(spec.dataset_label),
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                }
            )
            for subject in val_subjects:
                all_subject_entries.append(
                    {
                        "dataset_name": spec.dataset_name,
                        "dataset_label": int(spec.dataset_label),
                        "subject": subject,
                        "fold_index": int(fold_index),
                    }
                )
        fold_rows.append(fold_entry)

    return {
        "num_folds": int(num_folds),
        "seed": int(seed),
        "folds": fold_rows,
        "subject_assignments": all_subject_entries,
    }


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
