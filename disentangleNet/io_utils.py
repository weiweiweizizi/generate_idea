"""
Shared IO utilities.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (line 19: import save_json)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 999-1002: save_json usage)
- disentangle_modern_reconstructed/train_reflex_entry_alt.py (line 84-86: write_json)
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _jsonify(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def save_json(path: str | pathlib.Path, payload: Any) -> None:
    """Write payload as JSON, creating parent directories if needed."""
    p = pathlib.Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["save_json"]
