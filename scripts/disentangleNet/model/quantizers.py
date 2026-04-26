from __future__ import annotations

import torch
import torch.nn as nn

try:
    from vector_quantize_pytorch import FSQ
except ImportError:
    FSQ = None


def build_shared_quantizer(
    *,
    quantizer_type: str,
    levels: tuple[int, ...],
    shared_dim: int,
    lq_commitment_loss_weight: float,
    lq_quantization_loss_weight: float,
    lq_optimize_values: bool,
    fsq_preserve_symmetry: bool,
):
    """Build the fixed residual-FSQ quantizer used by disentangleNet v31."""

    if quantizer_type != "residual_fsq":
        raise ValueError(
            "disentangleNet only supports quantizer_type='residual_fsq', got "
            f"{quantizer_type!r}"
        )
    if FSQ is None:
        raise ImportError("FSQ is unavailable; install vector-quantize-pytorch")

    residual_fsq_layers = nn.ModuleList(
        [
            FSQ(
                levels=[level],
                dim=shared_dim,
                preserve_symmetry=fsq_preserve_symmetry,
            )
            for level in levels
        ]
    )
    return None, residual_fsq_layers


def quantize_shared_latent(
    shared_raw: torch.Tensor,
    *,
    quantizer_type: str,
    lq,
    residual_fsq_layers,
):
    """Quantize the shared latent with the fixed residual-FSQ stack."""

    if quantizer_type != "residual_fsq":
        raise ValueError(
            "disentangleNet only supports quantizer_type='residual_fsq', got "
            f"{quantizer_type!r}"
        )

    residual = shared_raw
    shared_quantized = torch.zeros_like(shared_raw)
    all_indices = []
    all_stage_quantized = []

    for layer in residual_fsq_layers:
        stage_quantized, stage_indices = layer(residual.unsqueeze(1))
        stage_quantized = stage_quantized.squeeze(1)
        stage_indices = stage_indices.squeeze(1)
        residual = residual - stage_quantized.detach()
        shared_quantized = shared_quantized + stage_quantized
        all_indices.append(stage_indices)
        all_stage_quantized.append(stage_quantized)

    return (
        shared_quantized,
        torch.stack(all_indices, dim=-1),
        torch.stack(all_stage_quantized, dim=1),
    )


def decode_latent_indices(
    indices: torch.Tensor,
    *,
    quantizer_type: str,
    levels: tuple[int, ...],
    lq,
):
    """Decode stacked residual-FSQ indices into one tensor per level."""

    if quantizer_type != "residual_fsq":
        raise ValueError(
            "disentangleNet only supports quantizer_type='residual_fsq', got "
            f"{quantizer_type!r}"
        )
    if indices.ndim < 2 or indices.shape[-1] != len(levels):
        raise ValueError(
            "residual_fsq expects stacked per-stage indices with "
            f"last dim {len(levels)}, got shape {tuple(indices.shape)}"
        )
    return [indices[..., i].long() for i in range(len(levels))]
