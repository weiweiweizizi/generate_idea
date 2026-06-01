"""
Side branch heads.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 45-116: SYMMETRIC_PAIRS + build_mirror_perm)
"""
from .features import build_mirror_perm, SYMMETRIC_PAIRS

__all__ = ["build_mirror_perm", "SYMMETRIC_PAIRS"]
