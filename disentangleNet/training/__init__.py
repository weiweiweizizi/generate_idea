"""
Training package.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 14-18: package-level imports)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 496-1086: train function)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def train(config_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Unified training entry point.

    Dispatches to reflex or v31 training based on config contents.
    Mirrors the train() function in train_reflex_entry.py and train_v31_entry.py.
    """
    import json

    if config_path is not None:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        family = payload.get("model_family", payload.get("family", "reflex"))
    else:
        family = kwargs.get("family", "reflex")

    if family == "v31":
        from disentangleNet.training.v31 import train_v31
        return train_v31(config_path=config_path, **kwargs)
    else:
        from disentangleNet.training.reflex import train_reflex
        return train_reflex(config_path=config_path, **kwargs)


__all__ = ["train"]
