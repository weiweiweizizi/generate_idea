from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .specs import DatasetSpec, create_severity_label, create_side_label


def build_sample_dict(
    *,
    row: pd.Series,
    motion: Any,
    spec: DatasetSpec,
    subject_width: int,
    mode: str,
    region: str,
    use_difference: bool,
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

    subj = str(row["subj"]).zfill(subject_width)
    label_5class = int(row["label_5class"])
    score = int(row["score"])
    window_idx = int(row["window_idx"])
    prev_window_idx = window_idx - 1 if use_difference else None
    matrix_path = spec.root / subj / f"win_{window_idx:03d}_{mode}.npy"
    sample_id = f"{spec.dataset_name}:{subj}:{mode}:win{window_idx:03d}{sample_suffix}"

    return {
        "image": torch.from_numpy(motion).unsqueeze(0).float(),
        "side_label": torch.tensor(create_side_label(label_5class), dtype=torch.long),
        "severity_label": torch.tensor(create_severity_label(score), dtype=torch.long),
        "dataset_label": torch.tensor(spec.dataset_label, dtype=torch.long),
        "label_5class": torch.tensor(label_5class, dtype=torch.long),
        "score": torch.tensor(score, dtype=torch.long),
        "window_idx": torch.tensor(window_idx, dtype=torch.long),
        "prev_window_idx": (
            torch.tensor(prev_window_idx, dtype=torch.long)
            if prev_window_idx is not None
            else None
        ),
        "subject": subj,
        "dataset_name": spec.dataset_name,
        "dataset_root": str(spec.root),
        "mode": mode,
        "region": region,
        "sample_id": sample_id,
        "matrix_path": str(matrix_path),
        "sample_source": sample_source,
    }
