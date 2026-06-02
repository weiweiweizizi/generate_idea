from __future__ import annotations

from dataclasses import dataclass

import torch

from disentangleNet.losses import side_private_orthogonality_loss


@dataclass(frozen=True)
class SideResidualOutputs:
    fold_features: torch.Tensor
    side_coefficients: torch.Tensor
    side_residual: torch.Tensor
    private_side_logits: torch.Tensor | None
    side_coeff_l1: torch.Tensor
    side_private_orth: torch.Tensor


@dataclass(frozen=True)
class ActionSideOutputs:
    group_action_logits: torch.Tensor
    action_side_representation: torch.Tensor | None


def build_side_residual_outputs(
    *,
    x: torch.Tensor,
    side_z: torch.Tensor | None,
    private_z: torch.Tensor,
    private_residual: torch.Tensor,
    side_residual_enabled: bool,
    side_fold_feature_dim: int,
    side_head_input_builder,
    side_coeff_head,
    side_basis_bank: torch.Tensor | None,
    private_side_adversary,
    private_side_grl_lambda: float,
    grad_reverse,
    enforce_matrix_constraints,
) -> SideResidualOutputs:
    if side_residual_enabled:
        fold_features = side_head_input_builder(x)
        side_coeff_input = (
            side_z
            if side_fold_feature_dim == 0
            else torch.cat([side_z, fold_features], dim=1)
        )
        side_coefficients = side_coeff_head(side_coeff_input)
        side_basis = enforce_matrix_constraints(side_basis_bank)
        side_residual = torch.einsum("bc,cxy->bxy", side_coefficients, side_basis)
        private_side_logits = (
            private_side_adversary(
                grad_reverse(private_z, private_side_grl_lambda)
            )
            if private_side_adversary is not None
            else None
        )
        side_coeff_l1 = side_coefficients.abs().mean()
        side_private_orth = side_private_orthogonality_loss(
            side_residual,
            private_residual,
        )
    else:
        fold_features = x.new_zeros((x.shape[0], 0))
        side_coefficients = x.new_zeros((x.shape[0], 0))
        side_residual = torch.zeros_like(private_residual)
        private_side_logits = None
        side_coeff_l1 = x.new_zeros(())
        side_private_orth = x.new_zeros(())

    return SideResidualOutputs(
        fold_features=fold_features,
        side_coefficients=side_coefficients,
        side_residual=side_residual,
        private_side_logits=private_side_logits,
        side_coeff_l1=side_coeff_l1,
        side_private_orth=side_private_orth,
    )


def build_action_side_outputs(
    *,
    side_residual_enabled: bool,
    action_side_input: str,
    action_side_detach: bool,
    side_coefficients: torch.Tensor,
    side_coefficients_seq: torch.Tensor | None,
    free_path_coefficients: torch.Tensor,
    free_path_coefficients_seq: torch.Tensor | None,
    free_path_usage: torch.Tensor,
    free_path_usage_seq: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    sequence_shape,
    mean_pool_sequence_tensor,
    side_coeff_to_logits,
    action_usage_to_side,
) -> ActionSideOutputs:
    if free_path_usage_seq is None:
        action_side_representation = None
    elif free_path_usage_seq.ndim == 2:
        action_side_representation = free_path_usage_seq.unsqueeze(1)
    else:
        action_side_representation = free_path_usage_seq

    if side_residual_enabled:
        pooled_side = mean_pool_sequence_tensor(
            side_coefficients_seq, sequence_shape, mask=valid_mask,
        )
        if pooled_side is None:
            pooled_side = side_coefficients
        if action_side_input == "shared_side_coeff":
            pooled_shared = mean_pool_sequence_tensor(
                free_path_coefficients_seq, sequence_shape, mask=valid_mask,
            )
            if pooled_shared is None:
                pooled_shared = free_path_coefficients
            pooled = torch.cat([pooled_shared, pooled_side], dim=1)
        else:
            pooled = pooled_side
        if action_side_detach:
            pooled = pooled.detach()
        group_action_logits = side_coeff_to_logits(pooled)
        action_side_representation = side_coefficients_seq
    elif action_side_input == "free_path_coeff":
        pooled = mean_pool_sequence_tensor(
            free_path_coefficients_seq, sequence_shape, mask=valid_mask,
        )
        if pooled is None:
            pooled = free_path_coefficients
        if action_side_detach:
            pooled = pooled.detach()
        group_action_logits = action_usage_to_side(pooled)
    else:  # free_path_usage
        pooled = mean_pool_sequence_tensor(
            free_path_usage_seq, sequence_shape, mask=valid_mask,
        )
        if pooled is None:
            pooled = free_path_usage
        if action_side_detach:
            pooled = pooled.detach()
        group_action_logits = action_usage_to_side(pooled)

    return ActionSideOutputs(
        group_action_logits=group_action_logits,
        action_side_representation=action_side_representation,
    )


__all__ = [
    "ActionSideOutputs",
    "SideResidualOutputs",
    "build_action_side_outputs",
    "build_side_residual_outputs",
]
