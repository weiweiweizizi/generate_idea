"""
Utility functions for analysis and export workflows.

Reconstructed from:
- disentangleNet/analysis/exporters/basis.py  (imports: compute_level_boundaries, get_shared_basis_bank, etc.)
- disentangleNet/analysis/exporters/patient.py  (imports: compute_shared_frame_weights, compose_window_matrix, etc.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(path: str | Path, payload: Any) -> None:
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Level / basis helpers
# ---------------------------------------------------------------------------

def parse_levels(levels: Any, *, default: tuple[int, ...] = (2, 6)) -> tuple[int, ...]:
    """Parse ``levels`` from a string ``"2,6"``, list, tuple, or None."""
    if levels is None:
        return default
    if isinstance(levels, str):
        parts = [v.strip() for v in levels.split(",") if v.strip()]
        return tuple(int(v) for v in parts)
    if isinstance(levels, (list, tuple)):
        return tuple(int(v) for v in levels)
    return default


def compute_level_boundaries(levels: tuple[int, ...]) -> list[int]:
    """Return cumulative boundaries from level counts, e.g. ``(2, 6) -> [2, 8]``."""
    boundaries: list[int] = []
    running = 0
    for count in levels:
        running += int(count)
        boundaries.append(running)
    return boundaries


def get_shared_basis_bank(model: Any) -> np.ndarray:
    """
    Extract the shared (action) basis bank as a numpy array from a loaded model.

    Returns shape ``[K, N, N]`` where ``K`` is the number of bases and ``N`` is
    the matrix size.
    """
    if model is None:
        raise ValueError("Model is None — full model loading is required for basis extraction")
    for attr in ("shared_basis_runtime", "reflex_basis_bank", "lowrank_basis_bank"):
        bank = getattr(model, attr, None)
        if bank is not None:
            if hasattr(bank, "get_structured_basis"):
                return bank.get_structured_basis().detach().cpu().numpy()
            if hasattr(bank, "basis"):
                return bank.basis.detach().cpu().numpy()
    # Fallback: look for a parameter named basis_bank or similar
    for name, param in model.named_parameters():
        if "basis_bank" in name and "side" not in name:
            return param.detach().cpu().numpy()
    raise AttributeError("Could not locate shared basis bank in model")


def get_side_basis_bank(model: Any) -> np.ndarray:
    """
    Extract the side basis bank as a numpy array from a loaded model.
    """
    if model is None:
        raise ValueError("Model is None — full model loading is required for side basis extraction")
    for name, param in model.named_parameters():
        if "side_basis" in name and "bank" in name:
            return param.detach().cpu().numpy()
        if "side_basis_bank" in name:
            return param.detach().cpu().numpy()
    # Some models store side basis under a module
    side_bank = getattr(model, "side_basis_bank", None)
    if side_bank is not None:
        if hasattr(side_bank, "basis"):
            return side_bank.basis.detach().cpu().numpy()
        return side_bank.detach().cpu().numpy()
    # Return empty array if no side basis found
    return np.zeros((0, 119, 119), dtype=np.float32)


def plot_basis_grid(
    basis: np.ndarray,
    levels: tuple[int, ...],
    output_path: str | Path,
) -> None:
    """Save a grid heatmap of basis matrices as a PNG image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    num_basis = basis.shape[0]
    ncols = min(num_basis, 8)
    nrows = (num_basis + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(num_basis):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mat = basis[idx]
        vmax = max(abs(mat.min()), abs(mat.max()), 1e-8)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"B{idx}", fontsize=8)
        ax.tick_params(labelsize=5)
    # Hide unused axes
    for idx in range(num_basis, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"Basis grid ({num_basis} bases, levels={list(levels)})", fontsize=10)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_BRIDGE_LAYOUT_MAP: dict[str, tuple[str, list[str]]] = {
    "mouth": ("face_regions_grouped", ["around_mouth", "mouth"]),
}


def resolve_bridge_point_layout(*, region: str) -> str:
    if region in _BRIDGE_LAYOUT_MAP:
        return _BRIDGE_LAYOUT_MAP[region][0]
    return "face_regions_grouped"


def resolve_bridge_point_layout_region_names(*, region: str) -> list[str] | None:
    if region in _BRIDGE_LAYOUT_MAP:
        return _BRIDGE_LAYOUT_MAP[region][1]
    return None


# ---------------------------------------------------------------------------
# Patient-level helpers (referenced by analysis/exporters/patient.py)
# ---------------------------------------------------------------------------

def compute_shared_frame_weights(
    coeffs: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute per-frame soft weights from basis coefficients via softmax."""
    abs_coeffs = np.abs(coeffs)
    max_vals = abs_coeffs.max(axis=-1, keepdims=True)
    logits = (abs_coeffs - max_vals) / max(max(temperature, 1e-8), 1e-8)
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum(axis=-1, keepdims=True).clip(min=1e-8)
    return weights.astype(np.float32)


def compose_window_matrix(
    coeffs: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Compose a single-window observation matrix from basis coefficients and basis
    matrices.

    ``coeffs``: shape ``[K]``
    ``basis``:  shape ``[K, N, N]``
    Returns:    shape ``[N, N]``
    """
    return np.einsum("k,kij->ij", coeffs.astype(np.float32), basis.astype(np.float32))


def expand_level_coefficients_to_basis_weights(
    coeffs: np.ndarray,
    level_boundaries: list[int],
    level_ranks: list[int] | None = None,
) -> np.ndarray:
    """
    Expand per-level coefficients into per-basis weights.

    For lowrank models, each level has a rank-sized coefficient vector that maps
    to a subset of the full basis bank.
    """
    num_bases = level_boundaries[-1] if level_boundaries else coeffs.shape[-1]
    weights = np.zeros(num_bases, dtype=np.float32)
    prev = 0
    for level_idx, boundary in enumerate(level_boundaries):
        count = boundary - prev
        rank = (level_ranks[level_idx] if level_ranks and level_idx < len(level_ranks)
                else count)
        level_coeffs = coeffs[..., prev:prev + rank] if coeffs.ndim > 1 else coeffs[prev:prev + rank]
        weights[prev:prev + count] = level_coeffs[:count]
        prev = boundary
    return weights


def resolve_motion_normalization_scale(
    raw_distance_matrix: np.ndarray,
    *,
    reference_matrix: np.ndarray | None = None,
) -> float:
    """Compute a per-window scale factor for normalizing observation matrices."""
    if reference_matrix is not None:
        ref_scale = np.abs(reference_matrix).mean()
        if ref_scale > 1e-8:
            return float(ref_scale)
    mean_abs = np.abs(raw_distance_matrix).mean()
    return float(max(mean_abs, 1e-8))


def resolve_target_spec(
    metadata: dict[str, Any],
    *,
    subject: str | None = None,
) -> dict[str, Any]:
    """Resolve the target patient specification from metadata."""
    return {
        "dataset_name": str(metadata.get("dataset_name", "")),
        "subject": str(subject or metadata.get("subject", "")),
        "mode": str(metadata.get("mode", "x")),
        "region": str(metadata.get("region", "mouth")),
    }


def reshape_sequence_feature(
    feature: np.ndarray,
    *,
    target_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Reshape a sequence feature array to the target shape if specified."""
    if target_shape is not None:
        return feature.reshape(target_shape)
    return feature


__all__ = [
    "compute_level_boundaries",
    "compute_shared_frame_weights",
    "compose_window_matrix",
    "expand_level_coefficients_to_basis_weights",
    "get_shared_basis_bank",
    "get_side_basis_bank",
    "parse_levels",
    "plot_basis_grid",
    "resolve_bridge_point_layout",
    "resolve_bridge_point_layout_region_names",
    "resolve_motion_normalization_scale",
    "resolve_target_spec",
    "reshape_sequence_feature",
    "save_json",
]
