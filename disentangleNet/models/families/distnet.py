"""Recovered `v31` main model family implementation fragment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from disentangleNet.models.basis_ops import (
    basis_l1_loss,
    get_joint_structured_basis,
    get_structured_basis,
    load_action_basis_init,
    load_side_basis_init,
    orthogonality_loss,
    project_basis_abs_max_,
    split_basis,
)
from disentangleNet.models.basis_pipeline.correction import (
    project_symmetric_zero_diagonal,
)
from disentangleNet.models.encoders import (
    SemanticBranchingEncoder,
    build_branch_adapter,
    build_branch_pool,
    build_motion_encoder,
)
from disentangleNet.models.heads import (
    build_discrete_side_classifier,
    build_free_head,
    build_group_severity_classifier,
    build_group_side_classifier,
    build_private_dataset_classifier,
    build_private_decoder,
    build_private_head,
    build_shared_basis_heads,
    build_shared_coeff_heads,
    build_shared_coeff_net,
    build_shared_dataset_adversary,
    build_static_side_encoder,
    build_side_classifier,
    build_side_head,
    build_side_semantic_basis_head,
    build_side_semantic_coeff_head,
)
from disentangleNet.models.quantizers import (
    build_shared_quantizer,
    decode_latent_indices,
    quantize_shared_latent,
)
from disentangleNet.models.sequence_utils import (
    flatten_sequence_labels,
    reshape_sequence_tensor,
)
from .v31_forward import compute_v31_forward_outputs


class DistNet(nn.Module):
    """
    Recovered high-confidence `v31` family fragment.

    This file is intentionally kept as a preserved source fragment even though
    some imported dependencies are not yet reconstructed.
    """

    SIDE_FIXED_REGION_BLOCKS = (
        (slice(0, 3), slice(0, 3)),
        (slice(3, 6), slice(3, 6)),
        (slice(6, 10), slice(6, 10)),
        (slice(10, 15), slice(10, 15)),
    )

    def __init__(
        self,
        side_label=None,
        levels=(2, 3, 6),
        basis_size=119,
        hidden_dim=32,
        pool_size=1,
        shared_dim=None,
        private_dim=32,
        private_decoder_hidden_dim=None,
        num_side_classes=3,
        num_severity_classes=3,
        num_dataset_classes=2,
        target_label_mode="side",
        private_residual_weight=0.25,
        grl_lambda=1.0,
        use_dataset_aux=False,
        action_basis_init_path=None,
        side_basis_init_path=None,
        lq_commitment_loss_weight=0.1,
        lq_quantization_loss_weight=0.1,
        lq_optimize_values=True,
        quantizer_type="latent_quantize",
        fsq_preserve_symmetry=True,
        basis_orthogonalization="normalize",
        discrete_side_loss_enabled=True,
        private_residual_max_l1=None,
        shared_basis_soft_mixing=False,
        shared_basis_anchor_bias=1.0,
        shared_basis_topk=None,
        basis_abs_max: float | None = None,
        side_semantic_enabled=False,
        side_basis_count=0,
        side_pooling="masked_mean",
        static_side_input_enabled=False,
        static_side_fusion_mode="add",
        side_subspace_dim=None,
        side_free_frame_qr=False,
        free_side_grl_lambda=1.0,
        early_branch_factorization=False,
        free_pool_size=2,
        side_pool_size=2,
        private_pool_size=1,
        free_z_dim=None,
        side_z_dim=None,
        private_adapter_enabled=False,
    ):
        super().__init__()

        self.levels = tuple(levels)
        self.total_basis_num = sum(self.levels)
        self.labels = side_label
        self.basis_size = basis_size
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.pooled_dim = hidden_dim * pool_size * pool_size
        self.shared_dim = shared_dim if shared_dim is not None else hidden_dim
        self.private_dim = private_dim
        self.private_decoder_hidden_dim = (
            private_decoder_hidden_dim
            if private_decoder_hidden_dim is not None
            else hidden_dim * 2
        )
        self.num_side_classes = num_side_classes
        self.num_severity_classes = num_severity_classes
        self.num_dataset_classes = num_dataset_classes
        self.target_label_mode = "side"
        self.requested_target_label_mode = str(target_label_mode)
        self.private_residual_weight = private_residual_weight
        self.grl_lambda = grl_lambda
        self.use_dataset_aux = bool(use_dataset_aux)
        self.action_basis_init_path = action_basis_init_path
        self.side_basis_init_path = side_basis_init_path
        self.lq_commitment_loss_weight = lq_commitment_loss_weight
        self.lq_quantization_loss_weight = lq_quantization_loss_weight
        self.lq_optimize_values = lq_optimize_values
        self.quantizer_type = quantizer_type
        self.fsq_preserve_symmetry = fsq_preserve_symmetry
        self.basis_orthogonalization = basis_orthogonalization
        self.discrete_side_loss_enabled = bool(discrete_side_loss_enabled)
        self.private_residual_max_l1 = private_residual_max_l1
        self.shared_basis_soft_mixing = shared_basis_soft_mixing
        self.shared_basis_anchor_bias = shared_basis_anchor_bias
        self.shared_basis_topk = shared_basis_topk
        self.basis_abs_max = basis_abs_max
        self.side_semantic_enabled = side_semantic_enabled
        self.side_basis_count = int(side_basis_count)
        self.side_pooling = side_pooling
        self.static_side_input_enabled = bool(static_side_input_enabled)
        self.static_side_fusion_mode = str(static_side_fusion_mode)
        self.side_free_frame_qr = side_free_frame_qr
        self.free_side_grl_lambda = free_side_grl_lambda
        self.early_branch_factorization = bool(early_branch_factorization)
        self.free_pool_size = int(free_pool_size)
        self.side_pool_size = int(side_pool_size)
        self.private_pool_size = int(private_pool_size)
        self.free_z_dim = int(
            free_z_dim if free_z_dim is not None else hidden_dim
        )
        self.side_z_dim = int(
            side_z_dim if side_z_dim is not None else hidden_dim
        )
        self.private_adapter_enabled = bool(private_adapter_enabled)

        if self.use_dataset_aux:
            raise ValueError("disentangleNet v31 does not support dataset auxiliary heads")
        if self.quantizer_type != "residual_fsq":
            raise ValueError(
                "disentangleNet v31 requires quantizer_type='residual_fsq', got "
                f"{self.quantizer_type!r}"
            )
        if self.discrete_side_loss_enabled:
            raise ValueError("disentangleNet v31 requires discrete_side_loss_enabled=False")
        if not self.early_branch_factorization:
            raise ValueError("disentangleNet v31 requires early_branch_factorization=True")
        if not self.side_semantic_enabled and self.side_basis_count > 0:
            raise ValueError("disentangleNet v31 requires side_semantic_enabled=True when side_basis_count > 0")

        self.shared_dim = self.free_z_dim
        self.side_subspace_dim = self.side_z_dim
        self.free_subspace_dim = self.free_z_dim
        self.side_classifier_dim = self.side_z_dim
