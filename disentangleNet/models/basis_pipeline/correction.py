from __future__ import annotations

import torch


def project_symmetric_zero_diagonal(mats: torch.Tensor) -> torch.Tensor:
    """Project matrices to symmetric zero-diagonal form."""

    mats = 0.5 * (mats + mats.transpose(-1, -2))
    diag = torch.diagonal(mats, dim1=-2, dim2=-1)
    return mats - torch.diag_embed(diag)


def scale_basis_abs_max(
    basis: torch.Tensor,
    *,
    target_abs_max: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    if target_abs_max <= 0:
        raise ValueError(f"target_abs_max must be positive, got {target_abs_max}")
    max_abs = basis.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(
        max_abs > eps,
        torch.full_like(max_abs, float(target_abs_max)) / max_abs,
        torch.ones_like(max_abs),
    )
    return basis * scale


@torch.no_grad()
def project_basis_abs_max_(
    basis: torch.Tensor,
    *,
    target_abs_max: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    basis.copy_(scale_basis_abs_max(basis, target_abs_max=target_abs_max, eps=eps))
    return basis


__all__ = [
    "project_basis_abs_max_",
    "project_symmetric_zero_diagonal",
    "scale_basis_abs_max",
]
