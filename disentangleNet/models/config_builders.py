"""
Config-to-ModelConfig builders.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 80-81: _build_reflex_basis_levels)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 198-278: build_modular_model_config)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 281-284: build_reflex_model)
"""
from __future__ import annotations

from typing import Any

import torch

from disentangleNet.config.schema import (
    BasisModeConfig,
    BasisModeType,
    ModelConfig,
    ModelFamily,
    ReflexConfig,
    SideBranchConfig,
    SideBranchType,
    SidePredictionInput,
    SideVariant,
)


def _build_reflex_basis_levels(self_count: int, pair_count: int) -> tuple[int, int]:
    return int(self_count), int(pair_count) * 2


def _coerce_level_tuple(value: object, *, default: tuple[int, int]) -> tuple[int, int]:
    """
    Parse a levels value from string, list, or tuple into a 2-tuple of ints.

    Reconstructed from:
    - disentangle_modern_reconstructed/train_reflex_entry_alt2.py  (lines 87-99)
    """
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Expected two comma-separated levels, got {value!r}")
        return int(parts[0]), int(parts[1])
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected two levels, got {value!r}")
        return int(value[0]), int(value[1])
    raise TypeError(f"Unsupported levels value: {value!r}")


def build_modular_model_config(
    config: dict[str, Any],
    *,
    mirror_perm: torch.LongTensor,
) -> ModelConfig:
    side_basis_count = int(config.get("side_basis_count", 3))
    side_prediction_input = SidePredictionInput(str(config.get("action_side_input", "free_path_coeff")))
    reflex_self_count = int(config.get("reflex_self_count", 2))
    reflex_pair_count = int(config.get("reflex_pair_count", 3))
    basis_levels = tuple(
        int(v)
        for v in config.get(
            "reflex_basis_levels",
            _build_reflex_basis_levels(reflex_self_count, reflex_pair_count),
        )
    )
    side_branch_type_raw = str(config.get("side_branch_type", "structured_reflex_side"))
    if side_branch_type_raw == "paired_competitive_side":
        side_branch_type = SideBranchType.PAIRED_COMPETITIVE
    elif side_branch_type_raw == "structured_reflex_side":
        side_branch_type = SideBranchType.STRUCTURED_REFLEX
    else:
        side_branch_type = SideBranchType.RESIDUAL

    side_variant_raw = str(config.get("side_variant", config.get("variant", "three_way")))
    side_variant = SideVariant.THREE_SIDE if side_variant_raw == "three_side" else SideVariant.THREE_WAY

    mode_type_raw = str(config.get("mode_type", config.get("basis_mode_type", "lowrank")))
    if mode_type_raw == "lowrank_basis":
        basis_mode_type = BasisModeType.LOWRANK_BASIS
    elif mode_type_raw == "lowrank":
        basis_mode_type = BasisModeType.LOWRANK
    else:
        basis_mode_type = BasisModeType.DIRECT

    return ModelConfig(
        family=ModelFamily.MODULAR,
        mode=str(config["mode"]),
        region=str(config["region"]),
        hidden_dim=int(config["hidden_dim"]),
        pool_size=int(config["pool_size"]),
        shared_dim=config["shared_dim"],
        private_dim=int(config["private_dim"]),
        private_decoder_hidden_dim=config["private_decoder_hidden_dim"],
        free_pool_size=int(config["free_pool_size"]),
        private_pool_size=int(config["private_pool_size"]),
        free_z_dim=config["free_z_dim"],
        private_branch_enabled=bool(config["private_branch_enabled"]),
        private_adapter_enabled=bool(config["private_adapter_enabled"]),
        shared_trunk_attention_enabled=bool(config.get("shared_trunk_attention_enabled", False)),
        shared_trunk_attention_layers=int(config.get("shared_trunk_attention_layers", 2)),
        shared_trunk_attention_heads=int(config.get("shared_trunk_attention_heads", 4)),
        shared_trunk_attention_ffn_dim=int(config.get("shared_trunk_attention_ffn_dim", 64)),
        basis=BasisModeConfig(
            mode_type=basis_mode_type,
            basis_size=int(config["basis_size"]),
            levels=basis_levels,
            init_path=config["action_basis_init_path"],
            orthogonalization=str(config["basis_orthogonalization"]),
            lowrank_level_ranks=tuple(int(v) for v in config.get("lowrank_level_ranks", (3, 5))),
            self_reflex_all=bool(config.get("self_reflex_all", config.get("shared_self_reflex_all", False))),
            postprocess=str(config.get("basis_postprocess", "none")),
            svd_truncation_rank=int(config["svd_truncation_rank"]) if config.get("svd_truncation_rank") is not None else None,
        ),
        reflex=ReflexConfig(
            enabled=True,
            self_count=reflex_self_count,
            pair_count=reflex_pair_count,
            mirror_perm_source=str(config.get("ordered_indices_path", "")) or None,
        ),
        side=SideBranchConfig(
            enabled=bool(config["side_residual_enabled"]),
            variant=side_variant,
            branch_type=side_branch_type,
            prediction_input=side_prediction_input,
            feature_mode=str(config["side_feature_mode"]),
            residual_weight=float(config["side_residual_weight"]),
            coeff_l1_weight=float(config["side_coeff_l1_weight"]),
            private_orth_weight=float(config["side_private_orth_weight"]),
            private_adv_weight=float(config["private_side_adv_weight"]),
            private_grl_lambda=float(config["private_side_grl_lambda"]),
            pair_count=int(config.get("side_pair_count", config.get("pair_count", 0))),
            pair_rank=int(config.get("side_pair_rank", config.get("pair_rank", 0))),
        ),
        extra={
            "num_side_classes": int(config["num_side_classes"]),
            "num_severity_classes": 3,
            "private_residual_weight": float(config["private_residual_weight"]),
            "private_residual_max_l1": config["private_residual_max_l1"],
            "shared_basis_soft_mixing": bool(config["shared_basis_soft_mixing"]),
            "shared_basis_anchor_bias": float(config["shared_basis_anchor_bias"]),
            "shared_basis_topk": config.get("shared_basis_topk"),
            "shared_selection_mode": str(config.get("shared_selection_mode", "mlp_coeff")),
            "basis_abs_max": config.get("basis_abs_max"),
            "lq_commitment_loss_weight": float(config["lq_commitment_loss_weight"]),
            "lq_quantization_loss_weight": float(config["lq_quantization_loss_weight"]),
            "lq_optimize_values": bool(config["lq_optimize_values"]),
            "quantizer_type": str(config["quantizer_type"]),
            "fsq_preserve_symmetry": bool(config["fsq_preserve_symmetry"]),
            "action_side_detach": bool(config.get("action_side_detach", False)),
            "mirror_perm": mirror_perm,
            "side_basis_count_override": side_basis_count,
            "side_branch_type": side_branch_type_raw,
            "early_branch_factorization": bool(config["early_branch_factorization"]),
            "static_side_input_enabled": bool(config.get("static_side_input_enabled", False)),
            "static_side_fusion_mode": str(config.get("static_side_fusion_mode", "add")),
            "shared_self_reflex_all": bool(config.get("shared_self_reflex_all", False)),
            "reflex_self_count": reflex_self_count,
            "reflex_pair_count": reflex_pair_count,
            "reflex_basis_levels": list(basis_levels),
        },
    )


def build_reflex_model(config: dict, *, mirror_perm: torch.LongTensor, num_dataset_classes: int):
    from disentangleNet.models import build_model
    modular_model_config = build_modular_model_config(config, mirror_perm=mirror_perm)
    model = build_model(modular_model_config, num_dataset_classes=num_dataset_classes)
    return model, modular_model_config


__all__ = [
    "_build_reflex_basis_levels",
    "build_modular_model_config",
    "build_reflex_model",
]
