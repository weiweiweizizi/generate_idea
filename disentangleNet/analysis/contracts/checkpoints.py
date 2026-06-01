"""
Checkpoint contract for disentangleNet analysis.

Reconstructed from:
- scripts/matrix_vis/docs/disentanglenet_matrix_vis_contract.md  (Section 3.1: CheckpointContract)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckpointContract:
    """Describes which disentangleNet family/framework a checkpoint belongs to."""

    framework: str = "disentangleNet"
    model_family: str = "legacy_v31"
    mode: str = "x"
    region: str = "mouth"
    levels: tuple[int, ...] = (2, 6)
    basis_size: int = 119
    side_branch_type: str = "legacy_residual_side"
    side_basis_count: int = 0
    config: dict[str, Any] = field(default_factory=dict)


def _infer_model_family(config: dict[str, Any]) -> tuple[str, str]:
    """
    Infer (framework, model_family) from a flat config dict.

    Returns (framework, model_family).
    """
    family = str(config.get("family", config.get("model_family", "")))
    has_reflex = bool(config.get("reflex_self_count")) or bool(config.get("reflex_pair_count"))
    has_lowrank = bool(config.get("lowrank_level_ranks"))
    has_side_residual = bool(config.get("side_residual_enabled"))

    if family == "legacy_v31" or (not has_reflex and not has_lowrank):
        return "disentangleNet", "legacy_v31"

    if has_reflex and has_lowrank:
        return "disentangleNet_lowrank", "lowrank_reflex"
    if has_lowrank:
        return "disentangleNet_lowrank", "lowrank"
    if has_reflex:
        return "disentangleNet", "legacy_v6_reflex"

    return "disentangleNet", "legacy_v31"


def infer_checkpoint_contract(checkpoint_path: str | Path) -> CheckpointContract:
    """
    Build a CheckpointContract from a checkpoint file and its sibling ``model_config.json``.

    The loader reads:
    1. The checkpoint payload (``torch.load``).
    2. ``model_config.json`` in the same directory as the checkpoint.

    It does NOT rely on the filename alone.
    """
    import torch

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "config" in payload:
        config = dict(payload["config"])
    elif isinstance(payload, dict) and "model_config" in payload:
        config = dict(payload["model_config"])
    else:
        config = {}

    config_json_path = ckpt_path.parent / "model_config.json"
    if config_json_path.exists():
        with open(config_json_path, encoding="utf-8") as f:
            file_config = json.load(f)
        config = {**file_config, **config}

    framework, model_family = _infer_model_family(config)
    levels_raw = config.get("levels", config.get("reflex_basis_levels", (2, 6)))
    if isinstance(levels_raw, str):
        levels = tuple(int(v.strip()) for v in levels_raw.split(",") if v.strip())
    elif isinstance(levels_raw, (list, tuple)):
        levels = tuple(int(v) for v in levels_raw)
    else:
        levels = (2, 6)

    side_branch_type = str(config.get("side_branch_type", "legacy_residual_side"))
    if side_branch_type == "structured_reflex_side":
        side_branch_type = "structured_reflex_side"
    else:
        side_branch_type = "legacy_residual_side"

    return CheckpointContract(
        framework=framework,
        model_family=model_family,
        mode=str(config.get("mode", "x")),
        region=str(config.get("region", "mouth")),
        levels=levels,
        basis_size=int(config.get("basis_size", 119)),
        side_branch_type=side_branch_type,
        side_basis_count=int(config.get("side_basis_count", 0)),
        config=config,
    )


__all__ = [
    "CheckpointContract",
    "infer_checkpoint_contract",
]
