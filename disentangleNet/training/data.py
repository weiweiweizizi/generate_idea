from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from disentangleNet.data import DatasetSpec, FacialMotionSequenceDataset, subject_split


def build_specs(data_roots: str) -> list[DatasetSpec]:
    """Build dataset specifications from a comma-separated roots string."""


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


def build_dataloaders(
    train_dataset,
    val_dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap train/val datasets into standard PyTorch dataloaders."""


__all__ = ["build_specs", "build_datasets", "build_dataloaders"]

