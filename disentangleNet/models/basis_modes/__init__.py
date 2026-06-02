from __future__ import annotations

import torch

from disentangleNet.config.schema import BasisModeType, ModelConfig

from disentangleNet.models.basis_pipeline.direct import DirectBasisRuntime
from disentangleNet.models.basis_pipeline.runtime import LowRankBasisRuntime


def build_basis_provider(config: ModelConfig):
    """Build the concrete basis runtime selected by the modular config."""

    mode_type = config.basis.mode_type
    if mode_type in {BasisModeType.LOWRANK, BasisModeType.LOWRANK_BASIS}:
        mirror_perm = config.extra.get("mirror_perm")
        if mirror_perm is not None:
            mirror_perm = torch.as_tensor(mirror_perm, dtype=torch.long)
        return LowRankBasisRuntime(
            levels=tuple(int(v) for v in config.basis.levels),
            basis_size=int(config.basis.basis_size),
            lowrank_level_ranks=tuple(int(v) for v in config.basis.lowrank_level_ranks),
            init_path=config.basis.init_path,
            mirror_perm=mirror_perm,
            basis_abs_max=float(config.extra.get("basis_abs_max", 0.05)),
            reflex_self_count=int(config.reflex.self_count if config.reflex.enabled else 0),
            reflex_pair_count=int(config.reflex.pair_count if config.reflex.enabled else 0),
        )
    if mode_type == BasisModeType.DIRECT:
        # TODO(recovery): restore the historical dense/direct basis runtime.
        _ = DirectBasisRuntime
        return None
    raise ValueError(f"Unsupported basis mode: {mode_type!r}")


__all__ = ["build_basis_provider"]
