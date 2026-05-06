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


def subject_split(
    spec: DatasetSpec,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    meta = pd.read_csv(spec.root / "metadata.csv")
    subject_width = infer_subject_width(spec)
    subjects = sorted(meta["subj"].astype(str).str.zfill(subject_width).unique().tolist())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = sorted(perm[:n_val].tolist())
    train_subjects = sorted(perm[n_val:].tolist())
    return train_subjects, val_subjects
