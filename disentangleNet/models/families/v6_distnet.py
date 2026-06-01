"""
V6: Action-Usage based Side Supervision.

Changes from DistNet v31:
1. Removes entire side branch (side_adapter, side_pool, side_head,
   side_semantic_coeff_head, side_semantic_basis_head, group_side_classifier,
   side_basis_bank).
2. A lightweight `ActionUsageToSideClassifier` predicts action-side labels
   from group-pooled action usage.
3. Reconstruction: identical to distnet forward (soft-mixing + shared_coeff_heads)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import (
    ReflexBasisBank,
    enforce_matrix_constraints,
    load_action_basis_init,
    split_basis,
)
from .encoder import build_branch_adapter, build_branch_pool, build_motion_encoder
from .heads import (
    build_private_decoder,
    build_private_head,
    build_shared_basis_heads,
    build_shared_coeff_heads,
    build_shared_coeff_net,
)
from .quantizers import build_shared_quantizer, quantize_shared_latent


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


class ActionUsageToSideClassifier(nn.Module):
    """
    Predict action-side label (left_affected / bilateral_normal / right_affected)
    from group-pooled action representation.

    Input:  [B, total_levels]  — mean-pooled action usage over the sequence
    Output: [B, num_side_classes] — logits
    """

    def __init__(self, total_levels: int, num_side_classes: int, hidden_dim: int = 32,
                 classifier_type: str = "linear"):
        super().__init__()
        self.classifier_type = classifier_type
        if classifier_type == "linear":
            # Simple Linear: good for dim=2 (free_path_coeff)
            self.net = nn.Linear(total_levels, num_side_classes)
        elif classifier_type == "mlp":
            # MLP: good for dim=8 (free_path_usage)
            self.net = nn.Sequential(
                nn.Linear(total_levels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_side_classes),
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def forward(self, pooled_coefficients: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_coefficients)


class V6DistNet(nn.Module):
    """
    V6 variant: removes the side branch, uses action usage for action-side prediction.
    Reconstruction logic is identical to DistNet forward.
    """

    def __init__(
        self,
        levels=(2, 6),
        basis_size=119,
        hidden_dim=32,
        pool_size=1,
        shared_dim=None,
        private_dim=32,
        private_decoder_hidden_dim=None,
        num_side_classes=3,
        num_severity_classes=3,
        num_dataset_classes=2,
        private_residual_weight=0.25,
        action_basis_init_path=None,
        lq_commitment_loss_weight=0.0,
        lq_quantization_loss_weight=0.0,
        lq_optimize_values=False,
        quantizer_type="residual_fsq",
        fsq_preserve_symmetry=True,
        basis_orthogonalization="joint_global_qr",
        private_residual_max_l1=None,
        shared_basis_soft_mixing=False,
        shared_basis_anchor_bias=1.0,
        shared_basis_topk=None,
        early_branch_factorization=True,
        free_pool_size=2,
        private_pool_size=1,
        free_z_dim=None,
        private_adapter_enabled=False,
        private_branch_enabled: bool = True,
        action_side_input: str = "free_path_coeff",  # "free_path_coeff" or "free_path_usage"
        side_residual_enabled: bool = False,
        side_feature_mode: str = "none",
        side_residual_weight: float = 1.0,
        side_coeff_l1_weight: float = 0.0,
        side_private_orth_weight: float = 0.0,
        private_side_adv_weight: float = 0.0,
        private_side_grl_lambda: float = 1.0,
        reflex_basis_enabled: bool = False,
        mirror_perm: torch.Tensor | None = None,
    ):
        super().__init__()

        self.levels = tuple(levels)
        self.total_basis_num = sum(self.levels)
        self.basis_size = basis_size
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.private_dim = private_dim
        self.private_decoder_hidden_dim = (
            private_decoder_hidden_dim
            if private_decoder_hidden_dim is not None
            else hidden_dim * 2
        )
        self.num_side_classes = num_side_classes
        self.num_severity_classes = num_severity_classes
        self.num_dataset_classes = num_dataset_classes
        self.private_residual_weight = private_residual_weight
        self.lq_commitment_loss_weight = lq_commitment_loss_weight
        self.lq_quantization_loss_weight = lq_quantization_loss_weight
        self.lq_optimize_values = lq_optimize_values
        self.quantizer_type = quantizer_type
        self.fsq_preserve_symmetry = fsq_preserve_symmetry
        self.basis_orthogonalization = basis_orthogonalization
        self.private_residual_max_l1 = private_residual_max_l1
        self.shared_basis_soft_mixing = shared_basis_soft_mixing
        self.shared_basis_anchor_bias = shared_basis_anchor_bias
        self.shared_basis_topk = shared_basis_topk
        self.early_branch_factorization = bool(early_branch_factorization)
        self.free_pool_size = int(free_pool_size)
        self.private_pool_size = int(private_pool_size)
        self.private_branch_enabled = bool(private_branch_enabled)
        self.action_side_input = action_side_input
        self.side_residual_enabled = bool(side_residual_enabled)
        self.side_feature_mode = str(side_feature_mode)
        self.side_residual_weight = float(side_residual_weight)
        self.side_coeff_l1_weight = float(side_coeff_l1_weight)
        self.side_private_orth_weight = float(side_private_orth_weight)
        self.private_side_adv_weight = float(private_side_adv_weight)
        self.private_side_grl_lambda = float(private_side_grl_lambda)
        self.reflex_basis_enabled = bool(reflex_basis_enabled)
        self.mirror_perm = mirror_perm

        self.free_z_dim = int(free_z_dim if free_z_dim is not None else hidden_dim)
        self.shared_dim = self.free_z_dim

        if self.quantizer_type != "residual_fsq":
            raise ValueError("V6 requires quantizer_type='residual_fsq'")

        # Encoder
        (
            self.initial_conv,
            self.pre_layer1_block,
            self.layer1,
            self.pre_layer2_block,
            self.layer2,
            self.layer3,
            self.avg_pool,
        ) = build_motion_encoder(hidden_dim, pool_size)

        # Free branch
        self.free_adapter = build_branch_adapter(hidden_dim)
        self.side_adapter = build_branch_adapter(hidden_dim) if self.side_residual_enabled else None
        self.private_adapter = (
            build_branch_adapter(hidden_dim) if private_adapter_enabled else None
        )
        self.free_pool = build_branch_pool(self.free_pool_size)
        self.side_pool = build_branch_pool(self.free_pool_size) if self.side_residual_enabled else None
        self.private_pool = build_branch_pool(self.private_pool_size)
        self.free_pooled_dim = hidden_dim * self.free_pool_size * self.free_pool_size
        self.side_pooled_dim = self.free_pooled_dim
        self.private_pooled_dim = hidden_dim * self.private_pool_size * self.private_pool_size

        self.free_head = nn.Sequential(
            nn.Linear(self.free_pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.free_z_dim),
        )
        if self.private_branch_enabled:
            self.private_head = nn.Sequential(
                nn.Linear(self.private_pooled_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, private_dim),
            )
        else:
            self.private_head = None
        self.side_fold_feature_dim = 6 if self.side_feature_mode == "folded_mouth_chunks" else 0
        if self.side_feature_mode not in {"none", "folded_mouth_chunks"}:
            raise ValueError(f"Unsupported side_feature_mode: {self.side_feature_mode!r}")
        if self.side_residual_enabled:
            self.side_head = nn.Sequential(
                nn.Linear(self.side_pooled_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.side_coeff_head = nn.Sequential(
                nn.Linear(hidden_dim + self.side_fold_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            side_logit_input_dim = 4 if action_side_input == "shared_side_coeff" else 2
            self.side_coeff_to_logits = nn.Linear(side_logit_input_dim, num_side_classes)
            self.side_basis_bank = nn.Parameter(
                torch.randn(2, basis_size, basis_size) * 0.002
            )
            if self.private_branch_enabled:
                self.private_side_adversary = nn.Sequential(
                    nn.Linear(private_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_side_classes),
                )
            else:
                self.private_side_adversary = None
        else:
            self.side_head = None
            self.side_coeff_head = None
            self.side_coeff_to_logits = None
            self.side_basis_bank = None
            self.private_side_adversary = None

        # LQ quantizer
        self.lq, self.residual_fsq_layers = build_shared_quantizer(
            quantizer_type=quantizer_type,
            levels=self.levels,
            shared_dim=self.shared_dim,
            lq_commitment_loss_weight=lq_commitment_loss_weight,
            lq_quantization_loss_weight=lq_quantization_loss_weight,
            lq_optimize_values=lq_optimize_values,
            fsq_preserve_symmetry=fsq_preserve_symmetry,
        )

        # Action basis bank
        if self.reflex_basis_enabled:
            self.action_basis_bank = nn.Parameter(
                torch.randn(self.total_basis_num, basis_size, basis_size) * 0.02,
                requires_grad=False,
            )
            self.reflex_basis_bank = ReflexBasisBank(
                levels=self.levels,
                basis_size=basis_size,
                init_path=action_basis_init_path,
                mirror_perm=self.mirror_perm,
            )
        else:
            self.action_basis_bank = nn.Parameter(
                torch.randn(self.total_basis_num, basis_size, basis_size) * 0.02
            )
            if action_basis_init_path is not None:
                self._load_action_basis_init(action_basis_init_path)

        # Shared coefficient heads
        self.shared_coeff_net = build_shared_coeff_net(
            self.free_z_dim, hidden_dim, len(self.levels)
        )
        self.shared_coeff_heads = build_shared_coeff_heads(
            shared_dim=self.free_z_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.shared_basis_heads = build_shared_basis_heads(
            shared_dim=self.free_z_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )

        # Private decoder
        if self.private_branch_enabled:
            self.private_decoder = build_private_decoder(
                private_dim=private_dim,
                private_decoder_hidden_dim=self.private_decoder_hidden_dim,
                basis_size=basis_size,
            )
        else:
            self.private_decoder = None

        # V6: action usage → action-side classifier
        # free_path_coeff (dim=2): use Linear classifier
        # free_path_usage (dim=8): use MLP classifier
        if action_side_input in {"free_path_coeff", "shared_side_coeff"}:
            clf_type = "linear"
            input_dim = len(self.levels) if action_side_input == "free_path_coeff" else 4
        elif action_side_input == "free_path_usage":
            clf_type = "mlp"
            input_dim = self.total_basis_num
        else:
            raise ValueError(f"Unsupported action_side_input: {action_side_input!r}")
        self.action_usage_to_side = ActionUsageToSideClassifier(
            total_levels=input_dim,
            num_side_classes=num_side_classes,
            hidden_dim=hidden_dim,
            classifier_type=clf_type,
        )

    def _enforce_matrix_constraints(self, mats: torch.Tensor) -> torch.Tensor:
        return enforce_matrix_constraints(mats)

    def _load_action_basis_init(self, init_path: str) -> None:
        load_action_basis_init(
            self.action_basis_bank,
            init_path=init_path,
            total_basis_num=self.total_basis_num,
            basis_size=self.basis_size,
        )

    def get_structured_basis(self) -> torch.Tensor:
        if self.reflex_basis_enabled:
            return self.reflex_basis_bank()
        return self.action_basis_bank

    def _limit_private_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.private_residual_max_l1 is None:
            return residual
        mean_abs = residual.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(mean_abs / float(self.private_residual_max_l1), min=1.0)
        return residual / scale

    def split_basis(self, all_basis: torch.Tensor):
        return split_basis(all_basis, self.levels)

    def _flatten_sequence_input(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
            return x, (B, T)
        return x, None

    def _flatten_sequence_labels(self, labels, sequence_shape):
        if labels is None or sequence_shape is None:
            return labels
        B, T = sequence_shape
        return labels.reshape(B * T) if labels.ndim > 1 else labels

    def _reshape_sequence_tensor(self, tensor, sequence_shape):
        if sequence_shape is None:
            return tensor
        B, T = sequence_shape
        if tensor.ndim == 1:
            return tensor.reshape(B, T)
        return tensor.reshape(B, T, *tensor.shape[1:])

    def _mean_pool_sequence_tensor(self, tensor, sequence_shape, mask=None):
        if tensor is None or sequence_shape is None:
            return None
        B, T = sequence_shape
        if mask is not None:
            mask = mask.to(device=tensor.device, dtype=tensor.dtype)
            if tensor.ndim == 1:
                tensor = tensor.reshape(B, T)
                return (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            tensor = tensor.reshape(B, T, -1)
            expanded_mask = mask.unsqueeze(-1)
            return (tensor * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp_min(1.0)
        if tensor.ndim == 1:
            return tensor.reshape(B, T).mean(dim=1)
        return tensor.reshape(B, T, -1).mean(dim=1)

    def _apply_sparse_basis_topk(self, logits):
        if self.shared_basis_topk is None or not self.training:
            return logits
        k = min(self.shared_basis_topk, logits.shape[-1])
        vals, idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return logits * mask

    def _fold_mouth_chunk_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.side_feature_mode == "none":
            return x.new_zeros((x.shape[0], 0))
        if self.basis_size != 119:
            raise ValueError("folded_mouth_chunks expects the 119x119 mouth crop")

        chunk_slices = (
            slice(0, 22),    # around_mouth_left
            slice(22, 45),   # around_mouth_right
            slice(45, 82),   # mouth_left
            slice(82, 119),  # mouth_right
        )
        matrix = x[:, 0] if x.ndim == 4 else x
        abs_matrix = matrix.abs()

        row_means = [abs_matrix[:, current, :].mean(dim=(1, 2)) for current in chunk_slices]
        col_means = [abs_matrix[:, :, current].mean(dim=(1, 2)) for current in chunk_slices]
        block_means = [
            abs_matrix[:, chunk_slices[0], chunk_slices[0]].mean(dim=(1, 2)),
            abs_matrix[:, chunk_slices[1], chunk_slices[1]].mean(dim=(1, 2)),
            abs_matrix[:, chunk_slices[2], chunk_slices[2]].mean(dim=(1, 2)),
            abs_matrix[:, chunk_slices[3], chunk_slices[3]].mean(dim=(1, 2)),
        ]
        around_left = 0.5 * (row_means[0] + col_means[0])
        around_right = 0.5 * (row_means[1] + col_means[1])
        mouth_left = 0.5 * (row_means[2] + col_means[2])
        mouth_right = 0.5 * (row_means[3] + col_means[3])
        around_contrast = around_left - around_right
        mouth_contrast = mouth_left - mouth_right
        around_sum = around_left + around_right
        mouth_sum = mouth_left + mouth_right
        within_around_contrast = block_means[0] - block_means[1]
        within_mouth_contrast = block_means[2] - block_means[3]
        return torch.stack(
            [
                around_contrast,
                mouth_contrast,
                around_sum,
                mouth_sum,
                within_around_contrast,
                within_mouth_contrast,
            ],
            dim=1,
        )

    def _side_private_orthogonality_loss(
        self,
        side_residual: torch.Tensor,
        private_residual: torch.Tensor,
    ) -> torch.Tensor:
        side_flat = F.normalize(side_residual.reshape(side_residual.shape[0], -1), dim=1)
        private_flat = F.normalize(private_residual.reshape(private_residual.shape[0], -1), dim=1)
        return (side_flat * private_flat).sum(dim=1).abs().mean()

    def forward(
        self,
        x,
        side_labels=None,
        label5_labels=None,
        dataset_labels=None,
        valid_mask=None,
        return_group_pooled: bool = False,
    ):
        x, sequence_shape = self._flatten_sequence_input(x)
        side_labels = self._flatten_sequence_labels(side_labels, sequence_shape)

        # Encoder
        feats = self.initial_conv(x)
        feats = self.pre_layer1_block(feats)
        feats = self.layer1(feats)
        feats = self.pre_layer2_block(feats)
        feats = self.layer2(feats)
        feats = self.layer3(feats)

        # Free + private branches
        free_feats = self.free_adapter(feats)
        side_feats = self.side_adapter(feats) if self.side_adapter is not None else None
        private_feats = (
            self.private_adapter(feats) if self.private_adapter is not None else feats
        )

        free_pooled = self.free_pool(free_feats).flatten(1)
        side_pooled = self.side_pool(side_feats).flatten(1) if side_feats is not None else None
        private_pooled = (
            self.private_pool(private_feats).flatten(1)
            if self.private_branch_enabled
            else None
        )

        free_raw = self.free_head(free_pooled)
        side_z = self.side_head(side_pooled) if side_pooled is not None else None
        if self.private_branch_enabled:
            private_z = self.private_head(private_pooled)
        else:
            private_z = x.new_zeros((x.shape[0], 0))

        # Quantization
        free_quantized, indices, stage_quantized = quantize_shared_latent(
            free_raw,
            quantizer_type=self.quantizer_type,
            lq=self.lq,
            residual_fsq_layers=self.residual_fsq_layers,
        )
        free_latent = free_quantized

        # Basis
        basis = self.get_structured_basis()
        basis_list = self.split_basis(basis)

        # Coefficients
        coeffs = (
            None
            if stage_quantized is not None
            else self.shared_coeff_net(free_latent)
        )
        level_quantized_list = (
            [stage_quantized[:, i] for i in range(stage_quantized.shape[1])]
            if stage_quantized is not None
            else [free_latent for _ in self.levels]
        )

        # Reconstruction loop — identical structure to distnet forward
        shared_reconstruction = torch.zeros(
            x.shape[0], self.basis_size, self.basis_size,
            device=x.device, dtype=x.dtype,
        )
        free_path_coeff_levels = []
        free_path_usage_levels = []
        free_path_rep_levels = []

        for level_idx, (basis_i, level_quantized_i) in enumerate(
            zip(basis_list, level_quantized_list)
        ):
            if self.shared_basis_soft_mixing:
                level_logits = self.shared_basis_heads[level_idx](level_quantized_i)
                if self.shared_basis_anchor_bias != 0.0:
                    # Anchor toward discrete index (reconstruction target)
                    pass
                level_logits = self._apply_sparse_basis_topk(level_logits)
                level_weights = F.softmax(level_logits, dim=-1)
                selected_basis = torch.einsum("bl,lxy->bxy", level_weights, basis_i)
            else:
                raise NotImplementedError("V6 only supports shared_basis_soft_mixing=True")

            if coeffs is None:
                coeff = self.shared_coeff_heads[level_idx](level_quantized_i)
                coeff = coeff.view(x.shape[0], 1, 1)
            else:
                coeff = coeffs[:, level_idx].view(x.shape[0], 1, 1)

            shared_reconstruction = shared_reconstruction + coeff * selected_basis
            free_path_coeff_levels.append(coeff.view(x.shape[0], 1))
            free_path_usage_levels.append(level_weights)
            free_path_rep_levels.append(level_weights * coeff.view(x.shape[0], 1))

        free_path_coefficients = torch.cat(free_path_coeff_levels, dim=1)
        free_path_usage = torch.cat(free_path_usage_levels, dim=1)
        free_path_rep = torch.cat(free_path_rep_levels, dim=1)
        free_level2_usage = free_path_usage_levels[1] if len(free_path_usage_levels) >= 2 else None
        free_level2_rep = free_path_rep_levels[1] if len(free_path_rep_levels) >= 2 else None
        free_level2_coefficients = (
            free_path_coeff_levels[1] if len(free_path_coeff_levels) >= 2 else None
        )

        # Private decoder
        if self.private_branch_enabled:
            private_residual = self.private_decoder(private_z).reshape(
                x.shape[0], self.basis_size, self.basis_size
            )
            private_residual = self._limit_private_residual(private_residual)
        else:
            private_residual = torch.zeros(
                x.shape[0], self.basis_size, self.basis_size,
                device=x.device, dtype=x.dtype,
            )
        if self.side_residual_enabled:
            fold_features = self._fold_mouth_chunk_features(x)
            side_coeff_input = (
                side_z
                if self.side_fold_feature_dim == 0
                else torch.cat([side_z, fold_features], dim=1)
            )
            side_coefficients = self.side_coeff_head(side_coeff_input)
            side_basis = self._enforce_matrix_constraints(self.side_basis_bank)
            side_residual = torch.einsum("bc,cxy->bxy", side_coefficients, side_basis)
            private_side_logits = (
                self.private_side_adversary(
                    grad_reverse(private_z, self.private_side_grl_lambda)
                )
                if self.private_side_adversary is not None
                else None
            )
            side_coeff_l1 = side_coefficients.abs().mean()
            side_private_orth = self._side_private_orthogonality_loss(
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

        reconstructed = (
            shared_reconstruction
            + self.side_residual_weight * side_residual
            - self.private_residual_weight * private_residual
        )

        # Reshape to sequence form
        side_residual_seq = self._reshape_sequence_tensor(
            side_residual.unsqueeze(1), sequence_shape
        )
        private_residual_seq = self._reshape_sequence_tensor(
            private_residual.unsqueeze(1), sequence_shape
        )
        shared_recon_seq = self._reshape_sequence_tensor(
            shared_reconstruction.unsqueeze(1), sequence_shape
        )
        reconstructed_seq = self._reshape_sequence_tensor(
            reconstructed.unsqueeze(1), sequence_shape
        )
        free_path_coefficients_seq = self._reshape_sequence_tensor(
            free_path_coefficients, sequence_shape,
        )
        free_path_usage_seq = self._reshape_sequence_tensor(
            free_path_usage, sequence_shape,
        )
        free_level2_coefficients_seq = self._reshape_sequence_tensor(
            free_level2_coefficients, sequence_shape,
        )
        side_coefficients_seq = self._reshape_sequence_tensor(
            side_coefficients, sequence_shape,
        )
        private_z_seq = self._reshape_sequence_tensor(private_z, sequence_shape)
        private_side_logits_seq = self._reshape_sequence_tensor(
            private_side_logits, sequence_shape,
        ) if private_side_logits is not None else None

        if free_path_usage_seq is None:
            action_side_representation = None
        elif free_path_usage_seq.ndim == 2:
            action_side_representation = free_path_usage_seq.unsqueeze(1)
        else:
            action_side_representation = free_path_usage_seq

        # V6: group-pool action usage → side prediction
        if self.side_residual_enabled:
            pooled_side = self._mean_pool_sequence_tensor(
                side_coefficients_seq, sequence_shape, mask=valid_mask,
            )
            if pooled_side is None:
                pooled_side = side_coefficients
            if self.action_side_input == "shared_side_coeff":
                pooled_shared = self._mean_pool_sequence_tensor(
                    free_path_coefficients_seq, sequence_shape, mask=valid_mask,
                )
                if pooled_shared is None:
                    pooled_shared = free_path_coefficients
                pooled = torch.cat([pooled_shared, pooled_side], dim=1)
            else:
                pooled = pooled_side
            if getattr(self, "action_side_detach", False):
                pooled = pooled.detach()
            group_action_logits = self.side_coeff_to_logits(pooled)
            action_side_representation = side_coefficients_seq
        elif self.action_side_input == "free_path_coeff":
            pooled = self._mean_pool_sequence_tensor(
                free_path_coefficients_seq, sequence_shape, mask=valid_mask,
            )
            if pooled is None:
                pooled = free_path_coefficients
        else:  # free_path_usage
            pooled = self._mean_pool_sequence_tensor(
                free_path_usage_seq, sequence_shape, mask=valid_mask,
            )
            if pooled is None:
                pooled = free_path_usage
        if not self.side_residual_enabled and getattr(self, "action_side_detach", False):
            pooled = pooled.detach()
        if not self.side_residual_enabled:
            group_action_logits = self.action_usage_to_side(pooled)

        # Group pooling for outputs
        free_latent_seq = self._reshape_sequence_tensor(free_latent, sequence_shape)
        group_pooled_free_rep = (
            self._mean_pool_sequence_tensor(free_latent_seq, sequence_shape, mask=valid_mask)
            if return_group_pooled else None
        )

        lq_loss = x.new_zeros(())
        orth_loss = x.new_zeros(())
        shared_basis_l1 = x.new_zeros(())
        side_basis_l1 = x.new_zeros(())
        basis_l1 = x.new_zeros(())
        residual_l1 = x.new_zeros(())
        lq_loss_per_sample = x.new_ones(x.shape[0]) * 1e-6
        residual_l1_per_sample = private_residual.abs().mean(dim=(1, 2))
        lq_loss_per_sample = self._reshape_sequence_tensor(lq_loss_per_sample, sequence_shape)
        residual_l1_per_sample = self._reshape_sequence_tensor(
            residual_l1_per_sample, sequence_shape
        )

        outputs = {
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
        }
        if self.reflex_basis_enabled:
            b = self.reflex_basis_bank
            outputs["v9_freq_loss"] = b.frequency_loss()
            outputs["reflex_orth_loss"] = b.orthogonality_loss()
            d = b.diagnostics()
            outputs["reflex_max_diag_abs"] = d.max_diag_abs
            outputs["reflex_max_symmetry_error"] = d.max_symmetry_error
            outputs["reflex_max_offdiag_gram_abs"] = d.max_offdiag_gram_abs
        return outputs

    @property
    def action_side_supervision_enabled(self):
        return True

    @property
    def side_semantic_enabled(self):
        return True  # Legacy compatibility for shared trainer checks

    @property
    def side_basis_count(self):
        return 0
