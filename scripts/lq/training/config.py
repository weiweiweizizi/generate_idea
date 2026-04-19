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


def validate_region_basis_size(region: str, basis_size: int) -> None:
    if region == "mouth" and basis_size != MOUTH_BASIS_SIZE:
        raise ValueError("Mouth region expects basis_size=119")
    if region == "full" and basis_size != FULL_BASIS_SIZE:
        raise ValueError("Full region expects basis_size=341")


def prepare_train_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(raw_config)
    side_cont_weight, side_disc_weight = resolve_side_weights(
        config["side_weight"],
        config["side_cont_weight"],
        config["side_disc_weight"],
    )
    config["side_cont_weight"] = side_cont_weight
    config["side_disc_weight"] = side_disc_weight

    validate_action_basis_init(
        config["action_basis_init_path"],
        require_basis_init=config["require_basis_init"],
    )
    validate_region_basis_size(config["region"], config["basis_size"])
    return config
