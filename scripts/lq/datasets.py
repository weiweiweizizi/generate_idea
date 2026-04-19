"""Compatibility shim for the extracted LQ dataset modules."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.lq.data import (
        DatasetSpec,
        FacialMotionDataset,
        FacialMotionSequenceDataset,
        create_severity_label,
        create_side_label,
        subject_split,
    )
else:
    from .data import (
        DatasetSpec,
        FacialMotionDataset,
        FacialMotionSequenceDataset,
        create_severity_label,
        create_side_label,
        subject_split,
    )

__all__ = [
    "DatasetSpec",
    "FacialMotionDataset",
    "FacialMotionSequenceDataset",
    "create_severity_label",
    "create_side_label",
    "subject_split",
]
