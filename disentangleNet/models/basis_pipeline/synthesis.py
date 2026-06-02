from __future__ import annotations

import torch


def synthesize_lowrank_basis(
    latent_vectors: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Compose symmetric low-rank basis matrices as sum_k coeff_k * u_k u_k^T.

    latent_vectors: [N, R, D]
    coefficients:   [N, R]
    returns:        [N, D, D]
    """

    if latent_vectors.ndim != 3:
        raise ValueError(
            f"latent_vectors must have shape [N, R, D], got ndim={latent_vectors.ndim}"
        )
    if coefficients.ndim != 2:
        raise ValueError(
            f"coefficients must have shape [N, R], got ndim={coefficients.ndim}"
        )
    if latent_vectors.shape[:2] != coefficients.shape:
        raise ValueError(
            "latent/coeff shape mismatch: "
            f"latent={tuple(latent_vectors.shape)} coeff={tuple(coefficients.shape)}"
        )
    weighted = latent_vectors * coefficients.unsqueeze(-1)
    return torch.einsum("nrd,nre->nde", weighted, latent_vectors)


def synthesize_direct_basis(*args, **kwargs) -> torch.Tensor:
    # TODO(recovery): restore the historical dense/direct basis runtime.
    raise NotImplementedError("direct basis synthesis is not recovered yet")


__all__ = [
    "synthesize_direct_basis",
    "synthesize_lowrank_basis",
]
