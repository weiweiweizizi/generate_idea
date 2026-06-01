"""
Checkpoint and model loaders for analysis / export workflows.

Reconstructed from:
- disentangleNet/analysis/exporters/basis.py  (imports infer_checkpoint_contract, load_model_for_analysis)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from disentangleNet.analysis.contracts.checkpoints import (
    CheckpointContract,
    infer_checkpoint_contract,
)


def load_model_for_analysis(
    checkpoint_path: str | Path,
    *,
    num_dataset_classes: int = 1,
) -> tuple[Any, dict[str, Any], CheckpointContract]:
    """
    Load a model and its config from a checkpoint for analysis / export.

    Returns ``(model, config, contract)``.

    Note: full model loading requires the model families in
    ``disentangleNet.models.families`` to be fully restored.  Until then,
    this function loads the checkpoint payload and config but returns
    a ``None`` model.  Callers that only need the config and contract
    (e.g. basis manifest builders) can still proceed.
    """
    import torch

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    contract = infer_checkpoint_contract(ckpt_path)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "config" in payload:
        config = dict(payload["config"])
    elif isinstance(payload, dict) and "model_config" in payload:
        config = dict(payload["model_config"])
    else:
        config = dict(contract.config)

    # Attempt full model build — may fail if families are not yet restored.
    try:
        from disentangleNet.models import build_model
        from disentangleNet.models.config_builders import build_modular_model_config
        from disentangleNet.models.side_heads.features import build_mirror_perm

        mirror_perm = build_mirror_perm(config.get("ordered_indices_path"))
        model_config = build_modular_model_config(config, mirror_perm=mirror_perm)
        model = build_model(model_config, num_dataset_classes=num_dataset_classes)

        state = payload.get("state_dict", payload)
        if isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values()):
            missing, unexpected, skipped = [], [], []
            for k, v in state.items():
                if k in model.state_dict():
                    try:
                        model.state_dict()[k].copy_(v)
                    except Exception:
                        skipped.append(k)
                else:
                    unexpected.append(k)
        model.eval()
        return model, config, contract
    except Exception:
        return None, config, contract


__all__ = [
    "CheckpointContract",
    "infer_checkpoint_contract",
    "load_model_for_analysis",
]
