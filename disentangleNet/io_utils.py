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


def save_json(path: str | pathlib.Path, payload: Any) -> None:
    """Write payload as JSON, creating parent directories if needed."""
    p = pathlib.Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["save_json"]
