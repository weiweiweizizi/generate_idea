"""
Public export surface for disentangleNet.models.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (line 39: build_model import)
- disentangle_modern_reconstructed/train_v31_entry.py    (line 18: build_model, build_v31_model_config imports)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 408-278: build_modular_model_config)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config_builders import build_modular_model_config, _build_reflex_basis_levels

if TYPE_CHECKING:
    from .builder import build_model, build_modular_model


def build_model(*args, **kwargs):
    from .builder import build_model as _build_model
    return _build_model(*args, **kwargs)


def build_modular_model(*args, **kwargs):
    from .builder import build_modular_model as _build
    return _build(*args, **kwargs)


def build_v31_model_config(config: dict) -> object:
    """
    Build a modern ModelConfig from a flat v31-style training config dict.

    Reconstructed from:
    - disentangle_modern_reconstructed/train_v31_entry.py  (lines 103-158)
    """
    from disentangleNet.config.schema import (
        BasisModeConfig, BasisModeType, ModelConfig, ModelFamily,
        ReflexConfig, SideBranchConfig, SideBranchType, SideVariant,
    )
    basis_size = int(config.get("basis_size", 119))
    levels_str = str(config.get("levels", "2,6"))
    levels = tuple(int(v.strip()) for v in levels_str.split(",") if v.strip())
    return ModelConfig(
        family=ModelFamily.LEGACY_V31,
        mode=str(config.get("mode", "x")),
        region=str(config.get("region", "mouth")),
        hidden_dim=int(config.get("hidden_dim", 32)),
        pool_size=int(config.get("pool_size", 1)),
        shared_dim=config.get("shared_dim"),
        private_dim=int(config.get("private_dim", 32)),
        private_decoder_hidden_dim=config.get("private_decoder_hidden_dim"),
        free_pool_size=int(config.get("free_pool_size", 2)),
        private_pool_size=int(config.get("private_pool_size", 1)),
        free_z_dim=config.get("free_z_dim"),
        private_branch_enabled=bool(config.get("private_branch_enabled", False)),
        private_adapter_enabled=bool(config.get("private_adapter_enabled", False)),
        basis=BasisModeConfig(
            mode_type=BasisModeType.DIRECT,
            basis_size=basis_size,
            levels=levels,
            init_path=config.get("action_basis_init_path"),
            orthogonalization=str(config.get("basis_orthogonalization", "joint_global_qr")),
        ),
        reflex=ReflexConfig(enabled=False),
        side=SideBranchConfig(
            enabled=bool(config.get("side_semantic_enabled", False)),
            variant=SideVariant.THREE_WAY,
            branch_type=SideBranchType.RESIDUAL,
        ),
        extra={
            "num_side_classes": int(config.get("num_side_classes", 3)),
            "side_basis_count": int(config.get("side_basis_count", 3)),
            "side_pooling": str(config.get("side_pooling", "masked_mean")),
            "side_semantic_enabled": bool(config.get("side_semantic_enabled", False)),
        },
    )


def normalize_init_basis_path(path: str | None) -> str | None:
    if path is None:
        return None
    import pathlib
    p = pathlib.Path(path)
    if p.exists():
        return str(p.resolve())
    return str(p)


__all__ = [
    "build_model",
    "build_modular_model",
    "build_modular_model_config",
    "build_v31_model_config",
    "normalize_init_basis_path",
]
