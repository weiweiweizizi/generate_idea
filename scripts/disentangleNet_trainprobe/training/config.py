from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

MOUTH_BASIS_SIZE = 119
FULL_BASIS_SIZE = 341


def resolve_side_weights(
    side_weight: float,
    side_cont_weight: float | None,
    side_disc_weight: float | None,
) -> tuple[float, float]:
    if side_cont_weight is None:
        side_cont_weight = side_weight
    if side_disc_weight is None:
        side_disc_weight = side_weight
    return side_cont_weight, side_disc_weight


def validate_action_basis_init(
    action_basis_init_path: str | None,
    *,
    require_basis_init: bool,
) -> None:
    if require_basis_init and not action_basis_init_path:
        raise ValueError(
            "action_basis_init_path is required for this training stage. "
            "Set require_basis_init=False to override."
        )
    if action_basis_init_path is None:
        print("[warning] Training without basis initialization.")
        return
    if not Path(action_basis_init_path).exists():
        raise FileNotFoundError(action_basis_init_path)


def validate_optional_init_path(init_path: str | None) -> None:
    if init_path is None:
        return
    if not Path(init_path).exists():
        raise FileNotFoundError(init_path)


def validate_region_basis_size(region: str, basis_size: int) -> None:
    if region == "mouth" and basis_size != MOUTH_BASIS_SIZE:
        raise ValueError("Mouth region expects basis_size=119")
    if region == "full" and basis_size != FULL_BASIS_SIZE:
        raise ValueError("Full region expects basis_size=341")


def validate_side_semantic_config(
    *,
    side_semantic_enabled: bool,
    side_basis_count: int,
    side_pooling: str,
    side_loss_weight: float,
    side_subspace_dim: int | None,
    side_free_frame_qr: bool,
    effective_shared_dim: int,
) -> None:
    if side_basis_count < 0:
        raise ValueError("side_basis_count must be >= 0")
    if not side_pooling:
        raise ValueError("side_pooling must be a non-empty string")
    if side_loss_weight < 0:
        raise ValueError("side_loss_weight must be >= 0")
    if side_semantic_enabled and side_basis_count <= 0:
        raise ValueError("side_basis_count must be > 0 when side_semantic_enabled=True")
    if side_subspace_dim is not None and (
        side_subspace_dim <= 0 or side_subspace_dim >= effective_shared_dim
    ):
        raise ValueError(
            "side_subspace_dim must satisfy 0 < side_subspace_dim < effective shared_dim"
        )
    if side_free_frame_qr:
        if side_subspace_dim is None:
            raise ValueError("side_subspace_dim must be set when side_free_frame_qr=True")
        free_subspace_dim = effective_shared_dim - side_subspace_dim
        if free_subspace_dim != side_subspace_dim:
            raise ValueError(
                "side_free_frame_qr=True requires side_subspace_dim == free_subspace_dim"
            )


def validate_aux_supervision_config(
    *,
    severity_loss_weight: float,
) -> None:
    if severity_loss_weight < 0:
        raise ValueError("severity_loss_weight must be >= 0")


def validate_label_supervision_config(config: Mapping[str, Any]) -> None:
    target_label_mode = str(config["target_label_mode"])
    if target_label_mode != "side":
        raise ValueError("disentangleNet v31 training only supports target_label_mode='side'")
    if int(config["num_side_classes"]) != 3:
        raise ValueError("disentangleNet_trainprobe requires num_side_classes=3")
    for key in ("group_side_loss_weight",):
        if float(config[key]) < 0:
            raise ValueError(f"{key} must be >= 0")

    if float(config["group_side_loss_weight"]) <= 0:
        raise ValueError("side mode requires group_side_loss_weight > 0")


def prepare_train_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(raw_config)
    config.setdefault("early_branch_factorization", False)
    config.setdefault("free_pool_size", 2)
    config.setdefault("side_pool_size", 2)
    config.setdefault("private_pool_size", 1)
    config.setdefault("free_z_dim", None)
    config.setdefault("side_z_dim", None)
    config.setdefault("private_adapter_enabled", False)
    config.setdefault("side_basis_init_path", None)
    config.setdefault("discrete_side_loss_enabled", True)
    config.setdefault("side_semantic_enabled", False)
    config.setdefault("side_basis_count", 0)
    config.setdefault("side_pooling", "masked_mean")
    config.setdefault("side_loss_weight", 0.0)
    config.setdefault("side_subspace_dim", None)
    config.setdefault("side_free_frame_qr", False)
    config.setdefault("subspace_orth_weight", 0.0)
    config.setdefault("free_side_adv_weight", 0.0)
    config.setdefault("free_side_grl_lambda", 1.0)
    config.setdefault("severity_loss_weight", 0.0)
    config.setdefault("target_label_mode", "side")
    config.setdefault("num_side_classes", 3)
    config.setdefault("group_side_loss_weight", None)

    side_cont_weight, side_disc_weight = resolve_side_weights(
        config["side_weight"],
        config["side_cont_weight"],
        config["side_disc_weight"],
    )
    config["side_cont_weight"] = side_cont_weight
    config["side_disc_weight"] = side_disc_weight
    if config["group_side_loss_weight"] is None:
        config["group_side_loss_weight"] = config["side_loss_weight"]

    validate_action_basis_init(
        config["action_basis_init_path"],
        require_basis_init=config["require_basis_init"],
    )
    validate_optional_init_path(config["side_basis_init_path"])
    validate_region_basis_size(config["region"], config["basis_size"])
    effective_shared_dim = (
        int(config["shared_dim"]) if config["shared_dim"] is not None else int(config["hidden_dim"])
    )
    if config["early_branch_factorization"]:
        if config["free_z_dim"] is None:
            config["free_z_dim"] = config["hidden_dim"]
        if config["side_z_dim"] is None:
            config["side_z_dim"] = config["hidden_dim"]

        if int(config["free_pool_size"]) <= 0 or int(config["side_pool_size"]) <= 0:
            raise ValueError("branch pool sizes must be positive")
        if int(config["private_pool_size"]) <= 0:
            raise ValueError("private_pool_size must be positive")
        if int(config["free_z_dim"]) <= 0 or int(config["side_z_dim"]) <= 0:
            raise ValueError("branch latent dims must be positive")

        config["side_free_frame_qr"] = False
        config["free_side_adv_weight"] = 0.0
        config["subspace_orth_weight"] = 0.0

    validate_side_semantic_config(
        side_semantic_enabled=config["side_semantic_enabled"],
        side_basis_count=config["side_basis_count"],
        side_pooling=config["side_pooling"],
        side_loss_weight=config["side_loss_weight"],
        side_subspace_dim=config["side_subspace_dim"],
        side_free_frame_qr=config["side_free_frame_qr"],
        effective_shared_dim=effective_shared_dim,
    )
    if config["subspace_orth_weight"] < 0:
        raise ValueError("subspace_orth_weight must be >= 0")
    if config["free_side_adv_weight"] < 0:
        raise ValueError("free_side_adv_weight must be >= 0")
    if config["free_side_grl_lambda"] < 0:
        raise ValueError("free_side_grl_lambda must be >= 0")
    validate_aux_supervision_config(severity_loss_weight=config["severity_loss_weight"])
    validate_label_supervision_config(config)
    return config
