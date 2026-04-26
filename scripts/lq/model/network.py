"""Compatibility layer for the refactored DistNet implementation."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.lq.model.distnet import DistNet, GradientReversalFn, grad_reverse
    from scripts.lq.model.encoder import build_branch_adapter, build_branch_pool
    from scripts.lq.model.heads import build_free_head, build_side_head
else:
    from .distnet import DistNet, GradientReversalFn, grad_reverse
    from .encoder import build_branch_adapter, build_branch_pool
    from .heads import build_free_head, build_side_head

__all__ = [
    "DistNet",
    "GradientReversalFn",
    "grad_reverse",
    "build_branch_adapter",
    "build_branch_pool",
    "build_free_head",
    "build_side_head",
]
