"""
Side branch feature helpers: mirror permutation and symmetric pairs.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 45-80: SYMMETRIC_PAIRS)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 85-116: build_mirror_perm)
"""
from __future__ import annotations

import torch

from disentangleNet.data.regions import SYMMETRIC_PAIRS, build_region_mirror_perm
from disentangleNet.models.features import (
    fold_mouth_chunk_features as _fold_mouth_chunk_features_impl,
)


def fold_mouth_chunk_features(
    x: torch.Tensor,
    *,
    side_feature_mode: str,
    basis_size: int,
) -> torch.Tensor:
    """Side-branch entrypoint for folded mouth chunk features."""

    return _fold_mouth_chunk_features_impl(
        x,
        side_feature_mode=side_feature_mode,
        basis_size=basis_size,
    )


def build_mirror_perm(
    ordered_indices_path: str,
    mouth_start: int = 188,
    mouth_end: int = 307,
) -> torch.LongTensor:
    if mouth_start != 188 or mouth_end != 307:
        raise ValueError(
            "build_mirror_perm currently routes through the recovered mouth region "
            f"only; got mouth_start={mouth_start}, mouth_end={mouth_end}"
        )
    return torch.from_numpy(
        build_region_mirror_perm(ordered_indices_path, region="mouth")
    ).long()


__all__ = ["SYMMETRIC_PAIRS", "build_mirror_perm", "fold_mouth_chunk_features"]
