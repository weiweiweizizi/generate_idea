"""
Side branch heads.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 45-116: SYMMETRIC_PAIRS + build_mirror_perm)
"""
from .factories import build_side_branch_config_overrides
from .features import build_mirror_perm, fold_mouth_chunk_features, SYMMETRIC_PAIRS
from .runtime import (
    ActionSideOutputs,
    SideResidualOutputs,
    build_action_side_outputs,
    build_side_residual_outputs,
)

__all__ = [
    "ActionSideOutputs",
    "build_mirror_perm",
    "fold_mouth_chunk_features",
    "build_action_side_outputs",
    "build_side_branch_config_overrides",
    "build_side_residual_outputs",
    "SideResidualOutputs",
    "SYMMETRIC_PAIRS",
]
