from __future__ import annotations

from typing import Any

from disentangleNet.config import ModelFamily
from disentangleNet.config.schema import BasisModeType, ModelConfig
from disentangleNet.models.families import (
    DistNet,
    LowRankDistNet,
    LowRankReflexDistNet,
    V6DistNet,
)

from .basis_modes import build_basis_provider
from .side_heads import build_side_branch_config_overrides


def build_modular_model(config: ModelConfig, *, num_dataset_classes: int) -> object:
    """
    Build a model instance from the modular modern config tree.

    This draft is now grounded by a recovered `models/assemblers.py` fragment.
    The exact implementation still needs reconciliation with live dependencies,
    but the branching logic and the key keyword arguments are now high-confidence.
    """

    side_kwargs = build_side_branch_config_overrides(config)
    basis_provider = build_basis_provider(config)
    if config.reflex.enabled:
        basis_levels = tuple(
            int(v)
            for v in config.extra.get("reflex_basis_levels", config.basis.levels)
        )
    else:
        basis_levels = config.basis.levels
    num_side_classes = int(config.extra.get("num_side_classes", 3))
    num_severity_classes = int(config.extra.get("num_severity_classes", 3))
    private_residual_weight = float(config.extra.get("private_residual_weight", 0.0))
    private_residual_max_l1 = config.extra.get("private_residual_max_l1")
    early_branch_factorization = bool(config.extra.get("early_branch_factorization", True))
    shared_basis_soft_mixing = bool(config.extra.get("shared_basis_soft_mixing", False))
    shared_selection_mode = config.extra.get("shared_selection_mode", "mlp_coeff")
    shared_basis_anchor_bias = float(config.extra.get("shared_basis_anchor_bias", 0.0))
    shared_basis_topk = config.extra.get("shared_basis_topk")
    quantizer_type = str(config.extra.get("quantizer_type", "residual_fsq"))
    fsq_preserve_symmetry = bool(config.extra.get("fsq_preserve_symmetry", True))
    lq_commitment_loss_weight = float(config.extra.get("lq_commitment_loss_weight", 0.0))
    lq_quantization_loss_weight = float(config.extra.get("lq_quantization_loss_weight", 0.0))
    lq_optimize_values = bool(config.extra.get("lq_optimize_values", False))
    action_side_detach = bool(config.extra.get("action_side_detach", False))

    common_kwargs: dict[str, Any] = {
        "levels": basis_levels,
        "basis_size": config.basis.basis_size,
        "basis_provider": basis_provider,
        "hidden_dim": config.hidden_dim,
        "pool_size": config.pool_size,
        "early_branch_factorization": early_branch_factorization,
        "shared_dim": config.shared_dim,
        "private_dim": config.private_dim,
        "private_decoder_hidden_dim": config.private_decoder_hidden_dim,
        "free_pool_size": config.free_pool_size,
        "private_pool_size": config.private_pool_size,
        "free_z_dim": config.free_z_dim,
        "shared_trunk_attention_enabled": bool(config.shared_trunk_attention_enabled),
        "shared_trunk_attention_layers": int(config.shared_trunk_attention_layers),
        "shared_trunk_attention_heads": int(config.shared_trunk_attention_heads),
        "shared_trunk_attention_ffn_dim": int(config.shared_trunk_attention_ffn_dim),
        "private_adapter_enabled": config.private_adapter_enabled,
        "private_branch_enabled": config.private_branch_enabled,
        "num_side_classes": num_side_classes,
        "num_severity_classes": num_severity_classes,
        "private_residual_weight": private_residual_weight,
        "private_residual_max_l1": private_residual_max_l1,
        "shared_basis_soft_mixing": shared_basis_soft_mixing,
        "shared_selection_mode": (
            shared_selection_mode.value
            if hasattr(shared_selection_mode, "value")
            else str(shared_selection_mode)
        ),
        "shared_basis_anchor_bias": shared_basis_anchor_bias,
        "shared_basis_topk": shared_basis_topk,
        "quantizer_type": quantizer_type,
        "fsq_preserve_symmetry": fsq_preserve_symmetry,
        "lq_commitment_loss_weight": lq_commitment_loss_weight,
        "lq_quantization_loss_weight": lq_quantization_loss_weight,
        "lq_optimize_values": lq_optimize_values,
        "num_dataset_classes": num_dataset_classes,
        **side_kwargs,
    }

    if config.family == ModelFamily.LEGACY_V31:
        return DistNet(
            levels=basis_levels,
            basis_size=config.basis.basis_size,
            hidden_dim=config.hidden_dim,
            pool_size=config.pool_size,
            early_branch_factorization=bool(config.early_branch_factorization),
            free_pool_size=config.free_pool_size,
            side_pool_size=int(config.extra.get("side_pool_size", 2)),
            private_pool_size=config.private_pool_size,
            free_z_dim=config.free_z_dim,
            side_z_dim=config.extra.get("side_z_dim"),
            private_adapter_enabled=config.private_adapter_enabled,
            side_basis_init_path=config.extra.get("side_basis_init_path"),
            shared_dim=config.shared_dim,
            private_dim=config.private_dim,
            private_decoder_hidden_dim=config.private_decoder_hidden_dim,
            num_side_classes=int(config.num_side_classes),
            num_severity_classes=int(config.num_severity_classes),
            num_dataset_classes=num_dataset_classes,
            target_label_mode=str(config.extra.get("target_label_mode", "side")),
            private_residual_weight=float(config.extra.get("private_residual_weight", 0.05)),
            private_residual_max_l1=config.private_residual_max_l1,
            shared_basis_soft_mixing=bool(config.shared_basis_soft_mixing),
            shared_basis_anchor_bias=float(config.shared_basis_anchor_bias),
            shared_basis_topk=config.shared_basis_topk,
            side_semantic_enabled=bool(config.extra.get("side_semantic_enabled", True)),
            side_basis_count=int(config.extra.get("side_basis_count", 3)),
            side_pooling=str(config.extra.get("side_pooling", "fixed_region2_contrast")),
            static_side_input_enabled=bool(config.extra.get("static_side_input_enabled", False)),
            static_side_fusion_mode=str(config.extra.get("static_side_fusion_mode", "add")),
            side_subspace_dim=config.extra.get("side_subspace_dim"),
            side_free_frame_qr=bool(config.extra.get("side_free_frame_qr", False)),
            free_side_grl_lambda=float(config.extra.get("free_side_grl_lambda", 1.0)),
            grl_lambda=float(config.grl_lambda),
            use_dataset_aux=bool(config.use_dataset_aux),
            action_basis_init_path=config.basis.init_path,
            lq_commitment_loss_weight=float(config.lq_commitment_loss_weight),
            lq_quantization_loss_weight=float(config.lq_quantization_loss_weight),
            lq_optimize_values=bool(config.lq_optimize_values),
            quantizer_type=str(config.quantizer_type),
            fsq_preserve_symmetry=bool(config.fsq_preserve_symmetry),
            basis_orthogonalization=config.basis.orthogonalization,
            basis_abs_max=config.extra.get("basis_abs_max"),
            discrete_side_loss_enabled=bool(config.extra.get("discrete_side_loss_enabled", False)),
        )

    if config.basis.mode_type in {BasisModeType.LOWRANK, BasisModeType.LOWRANK_BASIS}:
        lowrank_common_kwargs = {
            **common_kwargs,
            "action_basis_init_path": config.basis.init_path,
            "basis_orthogonalization": config.basis.orthogonalization,
            "lowrank_level_ranks": tuple(int(v) for v in config.basis.lowrank_level_ranks),
            "reflex_basis_enabled": config.reflex.enabled,
            "mirror_perm": config.extra.get("mirror_perm"),
            "action_side_detach": action_side_detach,
        }
        if config.reflex.enabled:
            return LowRankReflexDistNet(**lowrank_common_kwargs)
        return LowRankDistNet(**lowrank_common_kwargs)

    return V6DistNet(
        **common_kwargs,
        action_basis_init_path=config.basis.init_path,
        basis_orthogonalization=config.basis.orthogonalization,
        reflex_basis_enabled=config.reflex.enabled,
        mirror_perm=config.extra.get("mirror_perm"),
    )


def build_model(config: ModelConfig, *, num_dataset_classes: int) -> object:
    """
    Public builder entry used by training scripts.

    Downstream recovered call sites consistently use:
    - `build_model(model_config, num_dataset_classes=...)`
    """

    return build_modular_model(config, num_dataset_classes=num_dataset_classes)
