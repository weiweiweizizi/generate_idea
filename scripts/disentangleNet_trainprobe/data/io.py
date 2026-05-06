from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from ..regions import crop_region

if TYPE_CHECKING:
    from .specs import DatasetSpec


def zero_pad_array(region: str) -> np.ndarray:
    """Create one all-zero matrix with the same spatial size as the target region."""

    size = crop_region(np.zeros((341, 341), dtype=np.float32), region).shape[0]
    return np.zeros((size, size), dtype=np.float32)


def get_deleted_column(mode: str) -> str:
    """Resolve the direction-specific metadata column for weak-motion deletion."""

    return f"deleted_{mode}"


def infer_subject_width(spec: DatasetSpec) -> int:
    """
    Infer subject ID width from directory names.

    TT subjects are 6-digit IDs while IMR subjects are 5-digit IDs. The dataset
    code should not hardcode one width, otherwise file lookup breaks.
    """

    subject_dirs = [p.name for p in spec.root.iterdir() if p.is_dir()]
    if not subject_dirs:
        return 6
    return max(len(name) for name in subject_dirs)


def load_metadata(spec: DatasetSpec, subjects: Iterable[str], subject_width: int) -> pd.DataFrame:
    """Load and subject-filter metadata while normalizing subject ID width."""

    meta = pd.read_csv(spec.root / "metadata.csv")
    meta["subj"] = meta["subj"].astype(str).str.zfill(subject_width)
    subjects = {str(subj).zfill(subject_width) for subj in subjects}
    return meta[meta["subj"].isin(subjects)].copy()


def estimate_global_signed_scale(
    spec: DatasetSpec,
    subjects: Iterable[str],
    *,
    mode: str = "x",
    region: str = "mouth",
    use_difference: bool = True,
    sample_limit: int = 256,
    seed: int = 42,
    apply_deleted_filter: bool = True,
) -> float:
    """Estimate a robust global signed scale from a filtered subset of samples."""

    subject_width = infer_subject_width(spec)
    meta = load_metadata(spec, subjects, subject_width)
    if use_difference:
        meta = meta[meta["window_idx"] > 0].copy()
    if apply_deleted_filter:
        deleted_col = get_deleted_column(mode)
        if deleted_col in meta.columns:
            meta = meta[meta[deleted_col] == 0].copy()
    if len(meta) == 0:
        return 1.0

    rng = np.random.RandomState(seed)
    if len(meta) > sample_limit:
        meta = meta.iloc[rng.choice(len(meta), size=sample_limit, replace=False)]

    values = []
    for _, row in meta.iterrows():
        subj = str(row["subj"]).zfill(subject_width)
        window_idx = int(row["window_idx"])
        current = np.load(spec.root / subj / f"win_{window_idx:03d}_{mode}.npy").astype(np.float32)
        if use_difference:
            prev = np.load(spec.root / subj / f"win_{window_idx - 1:03d}_{mode}.npy").astype(
                np.float32
            )
            current = current - prev
        current = crop_region(current, region)
        values.append(current.reshape(-1))

    all_vals = np.concatenate(values)
    return max(float(np.percentile(np.abs(all_vals), 98)), 1e-6)
