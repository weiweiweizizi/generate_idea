from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from disentangleNet.models.basis import split_basis


@dataclass(frozen=True)
class SharedReconstructionOutputs:
    shared_reconstruction: torch.Tensor
    free_path_coefficients: torch.Tensor
    free_path_usage: torch.Tensor
    free_path_rep: torch.Tensor
    free_level2_usage: torch.Tensor | None
    free_level2_rep: torch.Tensor | None
    free_level2_coefficients: torch.Tensor | None


def build_phaseab_outputs(
    *,
    sequence_shape,
    valid_mask: torch.Tensor | None,
    return_group_pooled: bool,
    reshape_sequence_tensor,
    mean_pool_sequence_tensor,
    reconstructed: torch.Tensor,
    shared_reconstruction: torch.Tensor,
    side_residual: torch.Tensor,
    private_residual: torch.Tensor,
    free_path_coefficients: torch.Tensor,
    free_path_usage: torch.Tensor,
    free_level2_coefficients: torch.Tensor | None,
    side_coefficients: torch.Tensor,
    fold_features: torch.Tensor,
    private_side_logits: torch.Tensor | None,
    free_latent: torch.Tensor,
    side_z: torch.Tensor | None,
    private_z: torch.Tensor,
    action_side_representation: torch.Tensor | None,
    group_action_logits: torch.Tensor,
    stage_quantized: torch.Tensor | None,
    side_coeff_l1: torch.Tensor,
    side_private_orth: torch.Tensor,
) -> dict[str, torch.Tensor | dict | None]:
    side_residual_seq = reshape_sequence_tensor(
        side_residual.unsqueeze(1), sequence_shape
    )
    private_residual_seq = reshape_sequence_tensor(
        private_residual.unsqueeze(1), sequence_shape
    )
    shared_recon_seq = reshape_sequence_tensor(
        shared_reconstruction.unsqueeze(1), sequence_shape
    )
    reconstructed_seq = reshape_sequence_tensor(
        reconstructed.unsqueeze(1), sequence_shape
    )
    free_path_coefficients_seq = reshape_sequence_tensor(
        free_path_coefficients, sequence_shape,
    )
    free_path_usage_seq = reshape_sequence_tensor(
        free_path_usage, sequence_shape,
    )
    free_level2_coefficients_seq = reshape_sequence_tensor(
        free_level2_coefficients, sequence_shape,
    )
    side_coefficients_seq = reshape_sequence_tensor(
        side_coefficients, sequence_shape,
    )
    private_z_seq = reshape_sequence_tensor(private_z, sequence_shape)
    private_side_logits_seq = (
        reshape_sequence_tensor(private_side_logits, sequence_shape)
        if private_side_logits is not None
        else None
    )
    free_latent_seq = reshape_sequence_tensor(free_latent, sequence_shape)
    group_pooled_free_rep = (
        mean_pool_sequence_tensor(free_latent_seq, sequence_shape, mask=valid_mask)
        if return_group_pooled else None
    )

    lq_loss = reconstructed.new_zeros(())
    orth_loss = reconstructed.new_zeros(())
    shared_basis_l1 = reconstructed.new_zeros(())
    side_basis_l1 = reconstructed.new_zeros(())
    basis_l1 = reconstructed.new_zeros(())
    residual_l1 = reconstructed.new_zeros(())
    lq_loss_per_sample = reshape_sequence_tensor(
        reconstructed.new_ones(reconstructed.shape[0]) * 1e-6,
        sequence_shape,
    )
    residual_l1_per_sample = reshape_sequence_tensor(
        private_residual.abs().mean(dim=(1, 2)),
        sequence_shape,
    )

    return {
        "reconstructed": reconstructed_seq,
        "action_reconstruction": shared_recon_seq,
        "shared_reconstruction": shared_recon_seq,
        "side_reconstruction": side_residual_seq,
        "private_residual": private_residual_seq,
        "shared_quantized": stage_quantized[0] if stage_quantized is not None else free_latent,
        "free_latent": free_latent,
        "side_latent": side_z,
        "private_latent": private_z,
        "action_path_representation": action_side_representation,
        "lq_loss": lq_loss,
        "lq_loss_per_sample": lq_loss_per_sample,
        "orth_loss": orth_loss,
        "shared_basis_l1": shared_basis_l1,
        "side_basis_l1": side_basis_l1,
        "basis_l1": basis_l1,
        "residual_l1": residual_l1,
        "residual_l1_per_sample": residual_l1_per_sample,
        "side_coeff_l1": side_coeff_l1,
        "side_private_orth_loss": side_private_orth,
        "free_path_coefficients": free_path_coefficients,
        "free_path_usage": free_path_usage,
        "free_level2_coefficients": free_level2_coefficients,
        "side_coefficients": side_coefficients_seq,
        "side_coefficients_flat": side_coefficients,
        "side_fold_features": fold_features,
        "private_side_logits": private_side_logits_seq,
        "private_latent_seq": private_z_seq,
        "group_action_logits": group_action_logits,
        "group_pooled_free_rep": group_pooled_free_rep,
        "side_loss": {
            "side_loss": None,
            "side_loss_cont": None,
            "side_loss_disc": None,
            "side_loss_per_sample": None,
            "side_loss_cont_per_sample": None,
            "side_loss_disc_per_sample": None,
        },
        "_free_path_coefficients_seq": free_path_coefficients_seq,
        "_free_path_usage_seq": free_path_usage_seq,
        "_free_level2_coefficients_seq": free_level2_coefficients_seq,
        "_side_coefficients_seq": side_coefficients_seq,
    }


def build_shared_reconstruction(
    *,
    basis: torch.Tensor,
    levels: tuple[int, ...],
    level_quantized_list: list[torch.Tensor],
    coeffs: torch.Tensor | None,
    shared_basis_heads,
    shared_coeff_heads,
    shared_basis_soft_mixing: bool,
    shared_basis_anchor_bias: float,
    apply_sparse_basis_topk,
) -> SharedReconstructionOutputs:
    basis_list = split_basis(basis, levels)
    batch_size = basis.shape[0] if basis.ndim == 4 else level_quantized_list[0].shape[0]
    basis_size = basis.shape[-1]

    shared_reconstruction = torch.zeros(
        batch_size,
        basis_size,
        basis_size,
        device=basis.device,
        dtype=basis.dtype,
    )
    free_path_coeff_levels = []
    free_path_usage_levels = []
    free_path_rep_levels = []

    for level_idx, (basis_i, level_quantized_i) in enumerate(
        zip(basis_list, level_quantized_list)
    ):
        if shared_basis_soft_mixing:
            level_logits = shared_basis_heads[level_idx](level_quantized_i)
            if shared_basis_anchor_bias != 0.0:
                # TODO(recovery): restore anchor-biasing toward discrete indices.
                pass
            level_logits = apply_sparse_basis_topk(level_logits)
            level_weights = F.softmax(level_logits, dim=-1)
            selected_basis = torch.einsum("bl,lxy->bxy", level_weights, basis_i)
        else:
            raise NotImplementedError("V6 only supports shared_basis_soft_mixing=True")

        if coeffs is None:
            coeff = shared_coeff_heads[level_idx](level_quantized_i)
            coeff = coeff.view(level_quantized_i.shape[0], 1, 1)
        else:
            coeff = coeffs[:, level_idx].view(level_quantized_i.shape[0], 1, 1)

        shared_reconstruction = shared_reconstruction + coeff * selected_basis
        free_path_coeff_levels.append(coeff.view(level_quantized_i.shape[0], 1))
        free_path_usage_levels.append(level_weights)
        free_path_rep_levels.append(level_weights * coeff.view(level_quantized_i.shape[0], 1))

    free_path_coefficients = torch.cat(free_path_coeff_levels, dim=1)
    free_path_usage = torch.cat(free_path_usage_levels, dim=1)
    free_path_rep = torch.cat(free_path_rep_levels, dim=1)

    return SharedReconstructionOutputs(
        shared_reconstruction=shared_reconstruction,
        free_path_coefficients=free_path_coefficients,
        free_path_usage=free_path_usage,
        free_path_rep=free_path_rep,
        free_level2_usage=free_path_usage_levels[1] if len(free_path_usage_levels) >= 2 else None,
        free_level2_rep=free_path_rep_levels[1] if len(free_path_rep_levels) >= 2 else None,
        free_level2_coefficients=(
            free_path_coeff_levels[1] if len(free_path_coeff_levels) >= 2 else None
        ),
    )


__all__ = [
    "build_phaseab_outputs",
    "SharedReconstructionOutputs",
    "build_shared_reconstruction",
]
