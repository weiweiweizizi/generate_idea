"""Compatibility layer for the refactored DistNet implementation."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.lq.model.distnet import DistNet, GradientReversalFn, grad_reverse
else:
    from .distnet import DistNet, GradientReversalFn, grad_reverse

__all__ = ["DistNet", "GradientReversalFn", "grad_reverse"]
