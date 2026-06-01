"""
Recovered package placeholder for disentangleNet/losses/__init__.py.

Training entry fragments import from `disentangleNet.losses`:
- attach_region_laplacian
- build_reflex_loss_weights
- build_v31_loss_weights

PYC-confirmed related modules:
- laplacian.py
- runtime.py
- weights.py

Important correction from parsed pyc metadata:
- `reflex.py` is a training module: `disentangleNet/training/reflex.py`
- `v31.py` is a training module: `disentangleNet/training/v31.py`
- `static_side.py` is a data module: `disentangleNet/data/static_side.py`
"""

from .laplacian import attach_region_laplacian, matrix_laplacian_loss
from .runtime import step_model
from .weights import (
    build_lowrank_loss_weights,
    build_reflex_loss_weights,
    build_v31_loss_weights,
)

__all__ = [
    "attach_region_laplacian",
    "build_lowrank_loss_weights",
    "build_reflex_loss_weights",
    "build_v31_loss_weights",
    "matrix_laplacian_loss",
    "step_model",
]
