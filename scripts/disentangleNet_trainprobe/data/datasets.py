"""
Datasets for the LQ facial-motion prototype.

Two dataset styles live in this module:
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

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..regions import crop_region

from .io import (
    estimate_global_signed_scale,
    get_deleted_column,
    infer_subject_width,
    load_metadata,
    zero_pad_array,
)
from .samples import build_sample_dict
from .specs import DatasetSpec


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
        self.subject_width = infer_subject_width(spec)
        self.meta = load_metadata(spec, subjects, self.subject_width)

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
        row,
        motion: np.ndarray,
        sample_source: str = "valid",
        sample_suffix: str = "",
    ) -> dict:
        return build_sample_dict(
            row=row,
            motion=motion,
            spec=self.spec,
            subject_width=self.subject_width,
            mode=self.mode,
            region=self.region,
            use_difference=self.use_difference,
            sample_source=sample_source,
            sample_suffix=sample_suffix,
        )


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
            samples = samples[samples["window_idx"] > 0].copy()
        if apply_deleted_filter:
            deleted_col = get_deleted_column(mode)
            if deleted_col in samples.columns:
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
        self.zero_motion = zero_pad_array(region)
        self.groups = self._build_groups()

    def _build_groups(self) -> list[list[dict]]:
        """
        Build per-patient groups of exactly `group_size` windows.

        Example for `group_size=4`:
        - 9 valid windows -> [4] + [4] + [1 + fill]
        - 2 valid + 1 deleted -> [2 valid + 1 deleted_fill + 1 zero_pad]
        """

        deleted_col = get_deleted_column(self.mode)
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

            n_full_groups = len(valid_rows) // self.group_size
            for group_idx in range(n_full_groups):
                start = group_idx * self.group_size
                end = start + self.group_size
                groups.append([{"row": row, "source": "valid"} for row in valid_rows[start:end]])

            remainder = valid_rows[n_full_groups * self.group_size :]
            if remainder or not groups or subj not in {g[0]["row"]["subj"] for g in groups if g}:
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
        return estimate_global_signed_scale(
            spec,
            subjects,
            mode=mode,
            region=region,
            use_difference=use_difference,
            sample_limit=sample_limit,
            seed=seed,
            apply_deleted_filter=apply_deleted_filter,
        )
