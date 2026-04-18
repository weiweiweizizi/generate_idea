"""
Datasets for the LQ facial-motion prototype.

Two dataset styles live in this file:
1. `FacialMotionDataset`
   - flat per-window samples
   - suitable for the current simple training loop
2. `FacialMotionSequenceDataset`
   - grouped per-patient sequences of fixed length
   - built to support the later sequence-aware version of the model

Key data semantics encoded here:
- input comes from `data/winXX-stepXX/{IMR,TT}`
- motion is represented as signed difference matrices `ΔD = D_t - D_{t-1}`
- `deleted_x` / `deleted_y` mark weak-motion windows that should not count as
  valid diff samples for the corresponding direction
- grouped sequences must have exactly `group_size` windows:
  valid samples first, then deleted-window fill, then zero padding
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .regions import crop_region
except ImportError:
    from regions import crop_region


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


def _zero_pad_array(region: str) -> np.ndarray:
    """Create one all-zero matrix with the same spatial size as the target region."""

    size = crop_region(np.zeros((341, 341), dtype=np.float32), region).shape[0]
    return np.zeros((size, size), dtype=np.float32)


def _get_deleted_column(mode: str) -> str:
    """Resolve the direction-specific metadata column for weak-motion deletion."""

    return f"deleted_{mode}"


def _infer_subject_width(spec: DatasetSpec) -> int:
    """
    Infer subject ID width from directory names.

    TT subjects are 6-digit IDs while IMR subjects are 5-digit IDs. The dataset
    code should not hardcode one width, otherwise file lookup breaks.
    """

    subject_dirs = [p.name for p in spec.root.iterdir() if p.is_dir()]
    if not subject_dirs:
        return 6
    return max(len(name) for name in subject_dirs)


def _load_metadata(spec: DatasetSpec, subjects: Iterable[str], subject_width: int) -> pd.DataFrame:
    """Load and subject-filter metadata while normalizing subject ID width."""

    meta = pd.read_csv(spec.root / "metadata.csv")
    meta["subj"] = meta["subj"].astype(str).str.zfill(subject_width)
    subjects = {str(subj).zfill(subject_width) for subj in subjects}
    return meta[meta["subj"].isin(subjects)].copy()


class _BaseFacialMotionDataset(Dataset):
    """
    Shared utilities for both flat and grouped datasets.

    This base class deliberately does not define `__len__` / `__getitem__`;
    subclasses decide whether one sample is one diff window or one fixed-length
    patient sequence.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        subjects: Iterable[str],
        *,
        mode: str = "x",
        region: str = "mouth",
        use_difference: bool = True,
        signed_normalize: str = "per_sample",
        global_scale: float | None = None,
    ) -> None:
        self.spec = spec
        self.mode = mode
        self.region = region
        self.use_difference = use_difference
        self.signed_normalize = signed_normalize
        self.global_scale = global_scale
        self.subject_width = _infer_subject_width(spec)
        self.meta = _load_metadata(spec, subjects, self.subject_width)

    def _load_matrix(self, subj: str, window_idx: int) -> np.ndarray:
        """Read one direction-specific matrix from disk."""

        path = self.spec.root / subj / f"win_{window_idx:03d}_{self.mode}.npy"
        return np.load(path).astype(np.float32)

    def _normalize_signed(self, mat: np.ndarray) -> np.ndarray:
        """
        Signed normalization that preserves the sign of ΔD entries.

        This is a key difference from the earlier classification dataset. Here
        we intentionally keep negative values because the research conclusion is
        that motion semantics live in signed frame-to-frame changes.
        """

        if self.signed_normalize == "none":
            return mat

        if self.signed_normalize == "global":
            scale = self.global_scale
            if scale is None or scale <= 0:
                raise ValueError("global_scale must be provided for global normalization")
        elif self.signed_normalize == "per_sample":
            scale = float(np.percentile(np.abs(mat), 98))
        else:
            raise ValueError(f"Unsupported signed_normalize mode: {self.signed_normalize}")

        scale = max(scale, 1e-6)
        return np.clip(mat / scale, -1.0, 1.0)

    def _make_motion(self, subj: str, window_idx: int) -> np.ndarray:
        """Construct one model input matrix, usually a signed diff window."""

        current = self._load_matrix(subj, window_idx)
        if self.use_difference:
            prev = self._load_matrix(subj, window_idx - 1)
            motion = current - prev
        else:
            motion = current
        motion = crop_region(motion, self.region)
        return self._normalize_signed(motion)

    def _build_sample_dict(
        self,
        *,
        row: pd.Series,
        motion: np.ndarray,
        sample_source: str = "valid",
        sample_suffix: str = "",
    ) -> dict:
        """
        Build the standard metadata-rich sample dictionary used throughout LQ code.

        `sample_source` records whether the sample is:
        - `valid`:        true kept diff window
        - `deleted_fill`: window marked deleted but reused to complete a group
        - `zero_pad`:     synthetic all-zero filler
        """

        subj = str(row["subj"]).zfill(self.subject_width)
        label_5class = int(row["label_5class"])
        score = int(row["score"])
        window_idx = int(row["window_idx"])
        prev_window_idx = window_idx - 1 if self.use_difference else None
        matrix_path = self.spec.root / subj / f"win_{window_idx:03d}_{self.mode}.npy"
        sample_id = f"{self.spec.dataset_name}:{subj}:{self.mode}:win{window_idx:03d}{sample_suffix}"

        return {
            "image": torch.from_numpy(motion).unsqueeze(0).float(),
            "side_label": torch.tensor(create_side_label(label_5class), dtype=torch.long),
            "severity_label": torch.tensor(create_severity_label(score), dtype=torch.long),
            "dataset_label": torch.tensor(self.spec.dataset_label, dtype=torch.long),
            "label_5class": torch.tensor(label_5class, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.long),
            "window_idx": torch.tensor(window_idx, dtype=torch.long),
            "prev_window_idx": (
                torch.tensor(prev_window_idx, dtype=torch.long)
                if prev_window_idx is not None
                else None
            ),
            "subject": subj,
            "dataset_name": self.spec.dataset_name,
            "dataset_root": str(self.spec.root),
            "mode": self.mode,
            "region": self.region,
            "sample_id": sample_id,
            "matrix_path": str(matrix_path),
            "sample_source": sample_source,
        }


class FacialMotionDataset(_BaseFacialMotionDataset):
    """Flat dataset: one training sample corresponds to one diff window."""

    def __init__(
        self,
        spec: DatasetSpec,
        subjects: Iterable[str],
        *,
        mode: str = "x",
        region: str = "mouth",
        use_difference: bool = True,
        signed_normalize: str = "per_sample",
        global_scale: float | None = None,
        apply_deleted_filter: bool = True,
    ) -> None:
        super().__init__(
            spec,
            subjects,
            mode=mode,
            region=region,
            use_difference=use_difference,
            signed_normalize=signed_normalize,
            global_scale=global_scale,
        )
        samples = self.meta.copy()
        if use_difference:
            # Window 0 cannot form a diff sample because there is no previous
            # window to subtract from.
            samples = samples[samples["window_idx"] > 0].copy()
        if apply_deleted_filter:
            deleted_col = _get_deleted_column(mode)
            if deleted_col in samples.columns:
                # `deleted_x / deleted_y == 1` means the motion between the
                # current window and the previous one is too small to be useful.
                samples = samples[samples[deleted_col] == 0].copy()
        self.samples = samples.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        motion = self._make_motion(str(row["subj"]).zfill(self.subject_width), int(row["window_idx"]))
        return self._build_sample_dict(row=row, motion=motion)


class FacialMotionSequenceDataset(_BaseFacialMotionDataset):
    """
    Grouped dataset: one sample is a fixed-length per-patient diff sequence.

    Current grouping rule:
    - prefer true valid diff windows
    - if not enough, reuse deleted diff windows as `deleted_fill`
    - if still not enough, append all-zero `zero_pad` windows
    """

    def __init__(
        self,
        spec: DatasetSpec,
        subjects: Iterable[str],
        *,
        mode: str = "x",
        region: str = "mouth",
        use_difference: bool = True,
        signed_normalize: str = "per_sample",
        global_scale: float | None = None,
        group_size: int = 4,
        apply_deleted_filter: bool = True,
    ) -> None:
        super().__init__(
            spec,
            subjects,
            mode=mode,
            region=region,
            use_difference=use_difference,
            signed_normalize=signed_normalize,
            global_scale=global_scale,
        )
        self.group_size = group_size
        self.apply_deleted_filter = apply_deleted_filter
        self.zero_motion = _zero_pad_array(region)
        self.groups = self._build_groups()

    def _build_groups(self) -> list[list[dict]]:
        """
        Build per-patient groups of exactly `group_size` windows.

        Example for `group_size=4`:
        - 9 valid windows -> [4] + [4] + [1 + fill]
        - 2 valid + 1 deleted -> [2 valid + 1 deleted_fill + 1 zero_pad]
        """

        deleted_col = _get_deleted_column(self.mode)
        groups: list[list[dict]] = []

        for subj, subj_df in self.meta.groupby("subj", sort=True):
            subj_df = subj_df.sort_values("window_idx").copy()
            diff_df = subj_df[subj_df["window_idx"] > 0].copy() if self.use_difference else subj_df.copy()

            if deleted_col in diff_df.columns and self.apply_deleted_filter:
                valid_df = diff_df[diff_df[deleted_col] == 0].copy()
                deleted_df = diff_df[diff_df[deleted_col] != 0].copy()
            else:
                valid_df = diff_df.copy()
                deleted_df = diff_df.iloc[0:0].copy()

            valid_rows = [row for _, row in valid_df.iterrows()]
            deleted_rows = [row for _, row in deleted_df.iterrows()]

            # First emit all fully valid groups. This keeps the easy case clean
            # and prevents deleted / padded windows from leaking into groups
            # that could already be completed by valid data alone.
            n_full_groups = len(valid_rows) // self.group_size
            for group_idx in range(n_full_groups):
                start = group_idx * self.group_size
                end = start + self.group_size
                groups.append(
                    [{"row": row, "source": "valid"} for row in valid_rows[start:end]]
                )

            remainder = valid_rows[n_full_groups * self.group_size :]
            if remainder or not groups or subj not in {g[0]["row"]["subj"] for g in groups if g}:
                # The final partial group is completed in two stages:
                # 1. deleted-window fill
                # 2. zero padding if even deleted windows are insufficient
                current_group = [{"row": row, "source": "valid"} for row in remainder]
                deleted_needed = max(0, self.group_size - len(current_group))
                deleted_fill = deleted_rows[:deleted_needed]
                current_group.extend({"row": row, "source": "deleted_fill"} for row in deleted_fill)

                zero_needed = self.group_size - len(current_group)
                for zero_idx in range(zero_needed):
                    current_group.append(
                        {
                            "row": subj_df.iloc[0],
                            "source": "zero_pad",
                            "zero_pad_index": zero_idx,
                        }
                    )
                groups.append(current_group)

        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        group = self.groups[idx]
        frames = []
        sample_ids = []
        sample_sources = []
        window_indices = []
        prev_window_indices = []
        valid_mask = []
        padding_mask = []

        for item_idx, item in enumerate(group):
            row = item["row"]
            source = item["source"]
            if source == "zero_pad":
                # Zero padding uses the patient's metadata only as a carrier for
                # labels / identifiers; the actual matrix is synthetic zeros.
                motion = self.zero_motion.copy()
                sample = self._build_sample_dict(
                    row=row,
                    motion=motion,
                    sample_source=source,
                    sample_suffix=f":pad{item_idx}",
                )
            else:
                motion = self._make_motion(str(row["subj"]).zfill(self.subject_width), int(row["window_idx"]))
                sample = self._build_sample_dict(
                    row=row,
                    motion=motion,
                    sample_source=source,
                    sample_suffix=f":grp{idx}_{item_idx}",
                )
            frames.append(sample["image"])
            sample_ids.append(sample["sample_id"])
            sample_sources.append(sample["sample_source"])
            window_indices.append(sample["window_idx"])
            prev_window_indices.append(sample["prev_window_idx"])
            # `valid_mask` marks true usable diff windows.
            # `padding_mask` marks only synthetic zero pads.
            # deleted-window fills are intentionally neither valid nor padding:
            # they are real matrices but should not be treated as clean samples.
            valid_mask.append(1 if source == "valid" else 0)
            padding_mask.append(1 if source == "zero_pad" else 0)

        ref = self._build_sample_dict(
            row=group[0]["row"],
            motion=self.zero_motion.copy(),
            sample_source="group_ref",
            sample_suffix=f":group{idx}",
        )

        return {
            "images": torch.stack(frames, dim=0),
            "side_label": ref["side_label"],
            "severity_label": ref["severity_label"],
            "dataset_label": ref["dataset_label"],
            "label_5class": ref["label_5class"],
            "score": ref["score"],
            "subject": ref["subject"],
            "dataset_name": ref["dataset_name"],
            "dataset_root": ref["dataset_root"],
            "mode": ref["mode"],
            "region": ref["region"],
            "group_id": f"{ref['dataset_name']}:{ref['subject']}:{self.mode}:group{idx:03d}",
            "sample_ids": sample_ids,
            "sample_sources": sample_sources,
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "window_indices": torch.stack(window_indices),
            "prev_window_indices": [
                value.item() if isinstance(value, torch.Tensor) else None for value in prev_window_indices
            ],
        }

    @staticmethod
    def compute_global_scale(
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

        subject_width = _infer_subject_width(spec)
        meta = _load_metadata(spec, subjects, subject_width)
        if use_difference:
            meta = meta[meta["window_idx"] > 0].copy()
        if apply_deleted_filter:
            deleted_col = _get_deleted_column(mode)
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
                prev = np.load(spec.root / subj / f"win_{window_idx - 1:03d}_{mode}.npy").astype(np.float32)
                current = current - prev
            current = crop_region(current, region)
            values.append(current.reshape(-1))

        all_vals = np.concatenate(values)
        return max(float(np.percentile(np.abs(all_vals), 98)), 1e-6)


def subject_split(
    spec: DatasetSpec,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    meta = pd.read_csv(spec.root / "metadata.csv")
    subject_width = _infer_subject_width(spec)
    subjects = sorted(meta["subj"].astype(str).str.zfill(subject_width).unique().tolist())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = sorted(perm[:n_val].tolist())
    train_subjects = sorted(perm[n_val:].tolist())
    return train_subjects, val_subjects
