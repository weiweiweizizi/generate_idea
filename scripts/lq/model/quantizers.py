from __future__ import annotations

import torch
import torch.nn as nn

try:
    from vector_quantize_pytorch import FSQ, LatentQuantize
except ImportError:
    try:
        from .latent_quantization import LatentQuantize
        FSQ = None
    except ImportError:
        from latent_quantization import LatentQuantize
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
    """Build quantizer modules while preserving DistNet attribute layout."""

    if quantizer_type == "latent_quantize":
        lq = LatentQuantize(
            levels=levels,
            dim=shared_dim,
            commitment_loss_weight=lq_commitment_loss_weight,
            quantization_loss_weight=lq_quantization_loss_weight,
            optimize_values=lq_optimize_values,
        )
        return lq, None

    if quantizer_type == "fsq":
        if FSQ is None:
            raise ImportError("FSQ is unavailable; install vector-quantize-pytorch")
        lq = FSQ(
            levels=list(levels),
            dim=shared_dim,
            preserve_symmetry=fsq_preserve_symmetry,
        )
        return lq, None

    if quantizer_type == "residual_fsq":
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

    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")


def quantize_shared_latent(
    shared_raw: torch.Tensor,
    *,
    quantizer_type: str,
    lq,
    residual_fsq_layers,
):
    """Quantize shared latent and normalize outputs across quantizer variants."""

    if quantizer_type == "latent_quantize":
        shared_quantized, indices, _ = lq(shared_raw)
        return shared_quantized, indices, None

    if quantizer_type == "fsq":
        shared_quantized, indices = lq(shared_raw.unsqueeze(1))
        return shared_quantized.squeeze(1), indices.squeeze(1), None

    if quantizer_type == "residual_fsq":
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

    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")


def decode_latent_indices(
    indices: torch.Tensor,
    *,
    quantizer_type: str,
    levels: tuple[int, ...],
    lq,
):
    """Decode flattened quantizer indices into one index tensor per level."""

    if quantizer_type == "residual_fsq":
        if indices.ndim < 2 or indices.shape[-1] != len(levels):
            raise ValueError(
                "residual_fsq expects stacked per-stage indices with "
                f"last dim {len(levels)}, got shape {tuple(indices.shape)}"
            )
        return [indices[..., i].long() for i in range(len(levels))]

    indices = indices.long()
    basis = lq._basis.to(indices.device).long()
    quant_levels = lq._levels.to(indices.device).long()
    return [(indices // basis[i]) % quant_levels[i] for i in range(len(levels))]
