from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import infer_subject_width


def create_side_label(label_5class: int) -> int:
    """Map the 5-class clinical label to the 3-way side label used by LQ."""

    if label_5class < 2:
        return 0
    if label_5class == 2:
        return 1
    return 2


def create_severity_label(score: int) -> int:
    """Coarse severity binning used as auxiliary supervision."""

    if score <= 0:
        return 0
    if score <= 2:
        return 1
    return 2


@dataclass(frozen=True)
class DatasetSpec:
    """Minimal descriptor for one dataset root (e.g. TT or IMR)."""

    root: Path
    dataset_label: int
    dataset_name: str


def list_subjects(spec: DatasetSpec) -> list[str]:
    meta = pd.read_csv(spec.root / "metadata.csv")
    subject_width = infer_subject_width(spec)
    return sorted(meta["subj"].astype(str).str.zfill(subject_width).unique().tolist())


def subject_split(
    spec: DatasetSpec,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    subjects = list_subjects(spec)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = sorted(perm[:n_val].tolist())
    train_subjects = sorted(perm[n_val:].tolist())
    return train_subjects, val_subjects


def build_subject_folds(
    spec: DatasetSpec,
    *,
    num_folds: int,
    seed: int = 42,
) -> list[list[str]]:
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    subjects = list_subjects(spec)
    if num_folds > len(subjects):
        raise ValueError(
            f"num_folds={num_folds} exceeds number of subjects={len(subjects)} for {spec.dataset_name}"
        )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects).tolist()
    fold_arrays = np.array_split(np.asarray(perm, dtype=object), num_folds)
    return [sorted(arr.tolist()) for arr in fold_arrays]


def subject_kfold_split(
    spec: DatasetSpec,
    *,
    num_folds: int,
    fold_index: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    folds = build_subject_folds(spec, num_folds=num_folds, seed=seed)
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds) - 1}], got {fold_index}")

    val_subjects = folds[fold_index]
    train_subjects = sorted(
        subject
        for current_fold_idx, fold_subjects in enumerate(folds)
        if current_fold_idx != fold_index
        for subject in fold_subjects
    )
    return train_subjects, val_subjects
