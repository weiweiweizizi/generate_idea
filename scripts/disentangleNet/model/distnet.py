"""
Core LQ network for shared action-basis learning.

High-level decomposition implemented here:

input signed ΔD matrix
  -> CNN encoder
  -> shared branch  -> LQ -> discrete motion code -> action basis reconstruction
  -> private branch -> residual decoder          -> identity / nuisance residual

final reconstruction
  = shared action reconstruction
  + weighted private residual
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import (
    basis_l1_loss,
    enforce_matrix_constraints,
    get_joint_structured_basis,
    get_structured_basis,
    load_action_basis_init,
    load_side_basis_init,
    orthogonality_loss,
    split_basis,
)
from .encoder import build_branch_adapter, build_branch_pool, build_motion_encoder
from .heads import (
    build_free_head,
    build_group_side_classifier,
    build_private_decoder,
    build_private_head,
    build_shared_basis_heads,
    build_shared_coeff_heads,
    build_shared_coeff_net,
    build_side_classifier,
    build_side_head,
    build_side_semantic_basis_head,
    build_side_semantic_coeff_head,
)
from .quantizers import (
    build_shared_quantizer,
    decode_latent_indices,
    quantize_shared_latent,
)


class DistNet(nn.Module):
    """
    LQ-based motion decomposition network.

    Important conventions:
    - `levels=(2, 3, 6)` means the discrete latent is factorized into 3 groups.
    - `action_basis_bank` must be stored in the same order as those levels.
    - `mode=x` and `mode=y` are handled outside this model; each branch loads
      its own dataset and its own basis initialization tensor.
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
        side_semantic_enabled=False,
        side_basis_count=0,
        side_pooling="masked_mean",
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
        self.side_semantic_enabled = side_semantic_enabled
        self.side_basis_count = int(side_basis_count)
        self.side_pooling = side_pooling
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
        if not self.side_semantic_enabled:
            raise ValueError("disentangleNet v31 requires side_semantic_enabled=True")

        self.shared_dim = self.free_z_dim
        self.side_subspace_dim = self.side_z_dim
        self.free_subspace_dim = self.free_z_dim
        self.side_classifier_dim = self.side_z_dim

        if self.side_basis_count < 0:
            raise ValueError("side_basis_count must be >= 0")
        if self.side_semantic_enabled and self.side_basis_count <= 0:
            raise ValueError(
                "side_basis_count must be > 0 when side_semantic_enabled=True"
            )
        if not self.side_pooling:
            raise ValueError("side_pooling must be a non-empty string")

        (
            self.initial_conv,
            self.layer1,
            self.layer2,
            self.layer3,
            self.avg_pool,
        ) = build_motion_encoder(hidden_dim, pool_size)

        self.free_adapter = build_branch_adapter(hidden_dim)
        self.side_adapter = build_branch_adapter(hidden_dim)
        self.private_adapter = (
            build_branch_adapter(hidden_dim) if self.private_adapter_enabled else None
        )
        self.free_pool = build_branch_pool(self.free_pool_size)
        self.side_pool = build_branch_pool(self.side_pool_size)
        self.private_pool = build_branch_pool(self.private_pool_size)
        self.free_pooled_dim = hidden_dim * self.free_pool_size * self.free_pool_size
        if self.side_pooling == "fixed_block4_diag":
            self.side_pooled_dim = hidden_dim * 4
        elif self.side_pooling == "fixed_region2_contrast":
            self.side_pooled_dim = hidden_dim * 2
        else:
            self.side_pooled_dim = hidden_dim * self.side_pool_size * self.side_pool_size
        self.private_pooled_dim = hidden_dim * self.private_pool_size * self.private_pool_size
        self.shared_head = None
        self.free_head = build_free_head(
            self.free_pooled_dim,
            hidden_dim,
            self.free_z_dim,
        )
        self.side_head = build_side_head(
            self.side_pooled_dim,
            hidden_dim,
            self.side_z_dim,
        )
        self.private_head = build_private_head(
            self.private_pooled_dim,
            hidden_dim,
            private_dim,
        )
        self.lq, self.residual_fsq_layers = build_shared_quantizer(
            quantizer_type=quantizer_type,
            levels=self.levels,
            shared_dim=self.shared_dim,
            lq_commitment_loss_weight=lq_commitment_loss_weight,
            lq_quantization_loss_weight=lq_quantization_loss_weight,
            lq_optimize_values=lq_optimize_values,
            fsq_preserve_symmetry=fsq_preserve_symmetry,
        )

        self.action_basis_bank = nn.Parameter(
            torch.randn(self.total_basis_num, basis_size, basis_size) * 0.02
        )
        self.side_basis_bank = nn.Parameter(
            torch.randn(self.side_basis_count, basis_size, basis_size) * 0.02
        )
        if action_basis_init_path is not None:
            self._load_action_basis_init(action_basis_init_path)
        if side_basis_init_path is not None and self.side_basis_count > 0:
            self._load_side_basis_init(side_basis_init_path)

        self.shared_coeff_net = build_shared_coeff_net(
            self.free_subspace_dim, hidden_dim, len(self.levels)
        )
        self.shared_coeff_heads = build_shared_coeff_heads(
            shared_dim=self.free_subspace_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.shared_basis_heads = build_shared_basis_heads(
            shared_dim=self.free_subspace_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.side_semantic_coeff_head = (
            build_side_semantic_coeff_head(self.side_subspace_dim, hidden_dim)
            if self.side_basis_count > 0
            else None
        )
        self.side_semantic_basis_head = (
            build_side_semantic_basis_head(
                self.side_subspace_dim,
                hidden_dim,
                self.side_basis_count,
            )
            if self.side_basis_count > 0
            else None
        )
        self.group_side_classifier = (
            build_group_side_classifier(self.side_basis_count, num_side_classes)
            if self.side_basis_count > 0
            else None
        )
        self.private_decoder = build_private_decoder(
            private_dim=private_dim,
            private_decoder_hidden_dim=self.private_decoder_hidden_dim,
            basis_size=basis_size,
        )
        self.side_classifier = build_side_classifier(self.side_classifier_dim, num_side_classes)
        self.group_severity_classifier = None
        self.free_side_adversary = None
        self.discrete_side_classifier = None
        self.private_dataset_classifier = None
        self.shared_dataset_adversary = None

    def _enforce_matrix_constraints(self, mats: torch.Tensor) -> torch.Tensor:
        return enforce_matrix_constraints(mats)

    def _load_action_basis_init(self, init_path: str) -> None:
        load_action_basis_init(
            self.action_basis_bank,
            init_path=init_path,
            total_basis_num=self.total_basis_num,
            basis_size=self.basis_size,
        )

    def _load_side_basis_init(self, init_path: str) -> None:
        load_side_basis_init(
            self.side_basis_bank,
            init_path=init_path,
            side_basis_count=self.side_basis_count,
            basis_size=self.basis_size,
        )

    def get_structured_basis(self) -> torch.Tensor:
        shared_basis, _ = self._get_structured_basis_pair()
        return shared_basis

    def _get_structured_basis_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        return get_joint_structured_basis(
            self.action_basis_bank,
            self.side_basis_bank,
            levels=self.levels,
            total_basis_num=self.total_basis_num,
            side_basis_count=self.side_basis_count,
            basis_size=self.basis_size,
            basis_orthogonalization=self.basis_orthogonalization,
        )

    def get_side_basis(self) -> torch.Tensor:
        if self.side_basis_count == 0:
            return self.side_basis_bank
        _, side_basis = self._get_structured_basis_pair()
        return side_basis

    def _limit_private_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Cap private residual mean absolute magnitude per sample when configured.

        This prevents the model from bypassing a small `private_residual_weight`
        by simply inflating the raw residual amplitude.
        """

        if self.private_residual_max_l1 is None:
            return residual

        mean_abs = residual.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(mean_abs / float(self.private_residual_max_l1), min=1.0)
        return residual / scale

    def split_basis(self, all_basis: torch.Tensor):
        return split_basis(all_basis, self.levels)

    def decode_indices(self, indices: torch.Tensor):
        """
        Decode the flattened LQ index back into one index per latent level.

        This must stay consistent with `LatentQuantize.codes_to_indices()`.
        """

        return decode_latent_indices(
            indices,
            quantizer_type=self.quantizer_type,
            levels=self.levels,
            lq=self.lq,
        )

    def _quantize_shared(self, shared_raw: torch.Tensor):
        return quantize_shared_latent(
            shared_raw,
            quantizer_type=self.quantizer_type,
            lq=self.lq,
            residual_fsq_layers=self.residual_fsq_layers,
        )

    def _pool_side_tokens_fixed_blocks(self, side_feats: torch.Tensor) -> torch.Tensor:
        """Pool four fixed diagonal blocks from the early side feature map."""

        if side_feats.ndim != 4 or side_feats.shape[-2:] != (15, 15):
            raise ValueError(
                "fixed_block4_diag expects side_feats with shape [N, C, 15, 15], got "
                f"{tuple(side_feats.shape)}"
            )

        tokens = self._pool_side_tokens_fixed_regions(side_feats)
        return torch.cat(tokens, dim=1)

    def _pool_side_tokens_fixed_regions(
        self, side_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool four fixed region tokens from the early side feature map."""

        if side_feats.ndim != 4 or side_feats.shape[-2:] != (15, 15):
            raise ValueError(
                "fixed side region pooling expects side_feats with shape [N, C, 15, 15], got "
                f"{tuple(side_feats.shape)}"
            )

        tokens = []
        for row_slice, col_slice in self.SIDE_FIXED_REGION_BLOCKS:
            block = side_feats[:, :, row_slice, col_slice]
            tokens.append(block.mean(dim=(2, 3)))
        return tuple(tokens)

    def _pool_side_tokens_region_contrast(self, side_feats: torch.Tensor) -> torch.Tensor:
        """Pool explicit left-right contrast tokens for around-mouth and mouth regions."""

        (
            around_left,
            around_right,
            mouth_left,
            mouth_right,
        ) = self._pool_side_tokens_fixed_regions(side_feats)
        around_contrast = around_left - around_right
        mouth_contrast = mouth_left - mouth_right
        return torch.cat([around_contrast, mouth_contrast], dim=1)

    def orthogonality_loss(
        self,
        basis: torch.Tensor,
        side_basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if side_basis is None or side_basis.numel() == 0:
            return orthogonality_loss(basis, self.total_basis_num)
        all_basis = torch.cat([basis, side_basis], dim=0)
        return orthogonality_loss(all_basis, all_basis.shape[0])

    def basis_l1_loss(self, basis: torch.Tensor) -> torch.Tensor:
        return basis_l1_loss(basis)

    def basis_l1_components(
        self,
        basis: torch.Tensor,
        side_basis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_basis_l1 = self.basis_l1_loss(basis)
        if side_basis is None or side_basis.numel() == 0:
            side_basis_l1 = shared_basis_l1.new_zeros(())
        else:
            side_basis_l1 = self.basis_l1_loss(side_basis)
        return shared_basis_l1, side_basis_l1, shared_basis_l1 + side_basis_l1

    def _apply_sparse_basis_topk(self, level_logits: torch.Tensor) -> torch.Tensor:
        """
        Keep only the top-k basis logits per sample before softmax, if enabled.

        This preserves the anchor-guided discrete routing signal while allowing
        each level to mix a small number of nearby bases instead of exactly one.
        """

        if self.shared_basis_topk is None:
            return level_logits

        topk = int(self.shared_basis_topk)
        if topk <= 0:
            raise ValueError(f"shared_basis_topk must be positive, got {topk}")
        if topk >= level_logits.shape[-1]:
            return level_logits

        topk_indices = level_logits.topk(topk, dim=-1).indices
        keep_mask = torch.zeros_like(level_logits, dtype=torch.bool)
        keep_mask.scatter_(1, topk_indices, True)
        masked_value = torch.finfo(level_logits.dtype).min
        return level_logits.masked_fill(~keep_mask, masked_value)

    @staticmethod
    def _flatten_sequence_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        """Accept either frame or grouped-sequence input and flatten to frame axis."""

        if x.ndim == 4:
            return x, None
        if x.ndim == 5:
            batch_size, seq_len, channels, height, width = x.shape
            return x.reshape(batch_size * seq_len, channels, height, width), (batch_size, seq_len)
        raise ValueError(f"Expected input rank 4 or 5, got shape {tuple(x.shape)}")

    @staticmethod
    def _flatten_sequence_labels(
        labels: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        """Broadcast sequence-level labels across time when needed."""

        if labels is None or sequence_shape is None:
            return labels

        batch_size, seq_len = sequence_shape
        if labels.ndim == 1:
            if labels.shape[0] != batch_size:
                raise ValueError(
                    f"Expected {batch_size} labels for sequence batch, got {tuple(labels.shape)}"
                )
            labels = labels.unsqueeze(1).expand(batch_size, seq_len)
        elif labels.ndim == 2:
            expected = (batch_size, seq_len)
            if tuple(labels.shape) != expected:
                raise ValueError(
                    f"Expected sequence labels with shape {expected}, got {tuple(labels.shape)}"
                )
        else:
            raise ValueError(
                f"Expected sequence labels rank 1 or 2, got shape {tuple(labels.shape)}"
            )

        return labels.reshape(batch_size * seq_len)

    @staticmethod
    def _reshape_sequence_tensor(
        tensor: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        """Restore flattened frame-major tensors back to `B x T x ...`."""

        if tensor is None or sequence_shape is None:
            return tensor
        batch_size, seq_len = sequence_shape
        return tensor.reshape(batch_size, seq_len, *tensor.shape[1:])

    def _reshape_sequence_index_list(
        self,
        indices_list: list[torch.Tensor],
        sequence_shape: tuple[int, int] | None,
    ) -> list[torch.Tensor]:
        """Restore decoded per-level indices back to sequence form."""

        if sequence_shape is None:
            return indices_list
        return [
            self._reshape_sequence_tensor(level_indices, sequence_shape)
            for level_indices in indices_list
        ]

    @staticmethod
    def _mean_pool_sequence_tensor(
        tensor: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        """Optionally expose simple unmasked group pooling for later tasks."""

        if tensor is None or sequence_shape is None:
            return None
        batch_size, seq_len = sequence_shape
        if tensor.ndim >= 2 and tensor.shape[:2] == (batch_size, seq_len):
            return tensor.mean(dim=1)
        return tensor.reshape(batch_size, seq_len, *tensor.shape[1:]).mean(dim=1)

    def classify_side_group(self, group_rep: torch.Tensor) -> torch.Tensor:
        """Predict group-level side labels from pooled side-path representations."""

        if self.group_side_classifier is None:
            raise RuntimeError("group_side_classifier is unavailable when side_basis_count=0")
        return self.group_side_classifier(group_rep)

    def forward(
        self,
        x,
        side_labels=None,
        dataset_labels=None,
        return_group_pooled: bool = False,
    ):
        """
        Forward pass for one direction-specific motion matrix batch.

        Input shape:
        - frame batch: `(B, 1, H, W)`
        - grouped sequence batch: `(B, T, 1, H, W)`
        """
        x, sequence_shape = self._flatten_sequence_input(x)
        side_labels = self._flatten_sequence_labels(side_labels, sequence_shape)
        dataset_labels = self._flatten_sequence_labels(dataset_labels, sequence_shape)

        feats = self.initial_conv(x)
        feats = self.layer1(feats)
        feats = self.layer2(feats)
        feats = self.layer3(feats)

        return self._forward_early_branch(
            x=x,
            feats=feats,
            side_labels=side_labels,
            dataset_labels=dataset_labels,
            sequence_shape=sequence_shape,
            return_group_pooled=return_group_pooled,
        )

    def _forward_early_branch(
        self,
        *,
        x: torch.Tensor,
        feats: torch.Tensor,
        side_labels: torch.Tensor | None,
        dataset_labels: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
        return_group_pooled: bool,
    ):
        free_feats = self.free_adapter(feats)
        side_feats = self.side_adapter(feats)
        private_feats = self.private_adapter(feats) if self.private_adapter is not None else feats

        free_pooled = self.free_pool(free_feats).flatten(1)
        if self.side_pooling == "fixed_block4_diag":
            side_pooled = self._pool_side_tokens_fixed_blocks(side_feats)
        elif self.side_pooling == "fixed_region2_contrast":
            side_pooled = self._pool_side_tokens_region_contrast(side_feats)
        else:
            side_pooled = self.side_pool(side_feats).flatten(1)
        private_pooled = self.private_pool(private_feats).flatten(1)

        free_raw = self.free_head(free_pooled)
        side_latent = self.side_head(side_pooled)
        private_z = self.private_head(private_pooled)

        free_quantized, indices, stage_quantized = self._quantize_shared(free_raw)
        free_latent = free_quantized
        basis, side_basis = self._get_structured_basis_pair()
        basis_list = self.split_basis(basis)
        d_list = self.decode_indices(indices)

        coeffs = None if stage_quantized is not None else self.shared_coeff_net(free_latent)
        level_quantized_list = (
            [stage_quantized[:, i] for i in range(stage_quantized.shape[1])]
            if stage_quantized is not None
            else [free_latent for _ in self.levels]
        )
        shared_free_recon = torch.zeros(
            x.shape[0], self.basis_size, self.basis_size, device=x.device, dtype=x.dtype
        )
        free_path_coeff_levels = []
        free_path_usage_levels = []
        free_path_rep_levels = []

        for level_idx, (basis_i, d_i, level_quantized_i) in enumerate(
            zip(basis_list, d_list, level_quantized_list)
        ):
            if self.shared_basis_soft_mixing:
                level_logits = self.shared_basis_heads[level_idx](level_quantized_i)
                if self.shared_basis_anchor_bias != 0.0:
                    anchor = F.one_hot(d_i, num_classes=basis_i.shape[0]).to(level_logits.dtype)
                    level_logits = level_logits + self.shared_basis_anchor_bias * anchor
                level_logits = self._apply_sparse_basis_topk(level_logits)
                level_weights = F.softmax(level_logits, dim=-1)
                selected_basis = torch.einsum("bl,lxy->bxy", level_weights, basis_i)
            else:
                level_weights = F.one_hot(d_i, num_classes=basis_i.shape[0]).to(
                    device=x.device,
                    dtype=free_quantized.dtype,
                )
                selected_basis = basis_i[d_i]
            if coeffs is None:
                coeff = self.shared_coeff_heads[level_idx](level_quantized_i)
                coeff = coeff.view(x.shape[0], 1, 1)
            else:
                coeff = coeffs[:, level_idx].view(x.shape[0], 1, 1)
            shared_free_recon = shared_free_recon + coeff * selected_basis
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

        shared_side_recon = torch.zeros_like(shared_free_recon)
        side_path_usage = free_quantized.new_zeros((x.shape[0], self.side_basis_count))
        side_path_rep = free_quantized.new_zeros((x.shape[0], self.side_basis_count))
        side_path_coefficients = free_quantized.new_zeros((x.shape[0], 1))
        side_basis_logits = None
        if self.side_semantic_enabled and self.side_basis_count > 0:
            side_basis_logits = self.side_semantic_basis_head(side_latent)
            side_path_usage = F.softmax(side_basis_logits, dim=-1)
            side_coeff = self.side_semantic_coeff_head(side_latent).view(x.shape[0], 1, 1)
            side_path_coefficients = side_coeff.view(x.shape[0], 1)
            side_path_rep = side_path_usage * side_path_coefficients
            selected_side_basis = torch.einsum("bs,sxy->bxy", side_path_usage, side_basis)
            shared_side_recon = side_coeff * selected_side_basis

        shared_recon = shared_side_recon + shared_free_recon

        id_nuisance_residual = self.private_decoder(private_z).reshape(
            x.shape[0], self.basis_size, self.basis_size
        )
        id_nuisance_residual = self._enforce_matrix_constraints(id_nuisance_residual)
        id_nuisance_residual = self._limit_private_residual(id_nuisance_residual)
        recon = shared_recon + self.private_residual_weight * id_nuisance_residual
        recon = self._enforce_matrix_constraints(recon).unsqueeze(1)

        lq_loss_per_sample = free_raw.new_zeros(free_raw.shape[0])
        lq_loss = lq_loss_per_sample.mean()

        side_logits = None
        side_loss = None
        side_loss_cont = None
        side_loss_per_sample = None
        side_loss_cont_per_sample = None
        free_side_logits = None
        free_side_adv_loss = None
        free_side_adv_loss_per_sample = None
        if side_labels is not None:
            side_logits = self.side_classifier(side_latent)
            side_loss_cont_per_sample = F.cross_entropy(
                side_logits, side_labels, reduction="none"
            )
            side_loss_cont = side_loss_cont_per_sample.mean()
            side_loss_per_sample = side_loss_cont_per_sample
            side_loss = side_loss_cont

        private_dataset_logits = None
        shared_dataset_logits = None
        dataset_private_loss = None
        dataset_adv_loss = None
        dataset_private_loss_per_sample = None
        dataset_adv_loss_per_sample = None

        orth_loss = self.orthogonality_loss(basis, side_basis)
        shared_basis_l1, side_basis_l1, basis_l1 = self.basis_l1_components(
            basis,
            side_basis,
        )
        residual_l1_per_sample = id_nuisance_residual.abs().mean(dim=(1, 2))
        residual_l1 = residual_l1_per_sample.mean()

        reconstructed = self._reshape_sequence_tensor(recon, sequence_shape)
        action_reconstruction = self._reshape_sequence_tensor(
            self._enforce_matrix_constraints(shared_recon).unsqueeze(1),
            sequence_shape,
        )
        shared_side_reconstruction = self._reshape_sequence_tensor(
            self._enforce_matrix_constraints(shared_side_recon).unsqueeze(1),
            sequence_shape,
        )
        shared_free_reconstruction = self._reshape_sequence_tensor(
            self._enforce_matrix_constraints(shared_free_recon).unsqueeze(1),
            sequence_shape,
        )
        private_residual = self._reshape_sequence_tensor(
            id_nuisance_residual.unsqueeze(1),
            sequence_shape,
        )
        shared_quantized = self._reshape_sequence_tensor(free_quantized, sequence_shape)
        side_latent = self._reshape_sequence_tensor(side_latent, sequence_shape)
        free_latent = self._reshape_sequence_tensor(free_latent, sequence_shape)
        private_z = self._reshape_sequence_tensor(private_z, sequence_shape)
        indices = self._reshape_sequence_tensor(indices, sequence_shape)
        decoded_indices = self._reshape_sequence_index_list(d_list, sequence_shape)
        side_path_usage = self._reshape_sequence_tensor(side_path_usage, sequence_shape)
        free_path_usage = self._reshape_sequence_tensor(free_path_usage, sequence_shape)
        side_path_rep = self._reshape_sequence_tensor(side_path_rep, sequence_shape)
        free_path_rep = self._reshape_sequence_tensor(free_path_rep, sequence_shape)
        side_path_coefficients = self._reshape_sequence_tensor(
            side_path_coefficients,
            sequence_shape,
        )
        free_path_coefficients = self._reshape_sequence_tensor(
            free_path_coefficients,
            sequence_shape,
        )
        free_level2_usage = self._reshape_sequence_tensor(free_level2_usage, sequence_shape)
        free_level2_rep = self._reshape_sequence_tensor(free_level2_rep, sequence_shape)
        free_level2_coefficients = self._reshape_sequence_tensor(
            free_level2_coefficients,
            sequence_shape,
        )
        side_basis_logits = self._reshape_sequence_tensor(side_basis_logits, sequence_shape)
        side_logits = self._reshape_sequence_tensor(side_logits, sequence_shape)
        free_side_logits = self._reshape_sequence_tensor(free_side_logits, sequence_shape)
        private_dataset_logits = self._reshape_sequence_tensor(
            private_dataset_logits, sequence_shape
        )
        shared_dataset_logits = self._reshape_sequence_tensor(
            shared_dataset_logits, sequence_shape
        )
        lq_loss_per_sample = self._reshape_sequence_tensor(lq_loss_per_sample, sequence_shape)
        residual_l1_per_sample = self._reshape_sequence_tensor(
            residual_l1_per_sample, sequence_shape
        )
        side_loss_per_sample = self._reshape_sequence_tensor(
            side_loss_per_sample, sequence_shape
        )
        side_loss_cont_per_sample = self._reshape_sequence_tensor(
            side_loss_cont_per_sample, sequence_shape
        )
        free_side_adv_loss_per_sample = self._reshape_sequence_tensor(
            free_side_adv_loss_per_sample,
            sequence_shape,
        )
        dataset_private_loss_per_sample = self._reshape_sequence_tensor(
            dataset_private_loss_per_sample, sequence_shape
        )
        dataset_adv_loss_per_sample = self._reshape_sequence_tensor(
            dataset_adv_loss_per_sample, sequence_shape
        )
        group_pooled_side_rep = (
            self._mean_pool_sequence_tensor(side_latent, sequence_shape)
            if return_group_pooled
            else None
        )
        group_pooled_free_rep = (
            self._mean_pool_sequence_tensor(free_latent, sequence_shape)
            if return_group_pooled
            else None
        )
        group_pooled_side_latent = group_pooled_side_rep
        group_pooled_free_latent = group_pooled_free_rep
        group_pooled_side_latent_raw = None
        group_pooled_free_latent_raw = None

        return {
            "reconstructed": reconstructed,
            "action_reconstruction": action_reconstruction,
            "shared_reconstruction": action_reconstruction,
            "shared_side_reconstruction": shared_side_reconstruction,
            "shared_free_reconstruction": shared_free_reconstruction,
            "id_nuisance_residual": private_residual,
            "private_residual": private_residual,
            "shared_quantized": shared_quantized,
            "side_latent_raw": None,
            "free_latent_raw": None,
            "side_latent": side_latent,
            "free_latent": free_latent,
            "private_z": private_z,
            "indices": indices,
            "decoded_indices": decoded_indices,
            "action_basis": basis,
            "basis": basis,
            "side_basis": side_basis,
            "lq_loss": lq_loss,
            "lq_loss_per_sample": lq_loss_per_sample,
            "orth_loss": orth_loss,
            "shared_basis_l1": shared_basis_l1,
            "side_basis_l1": side_basis_l1,
            "basis_l1": basis_l1,
            "residual_l1": residual_l1,
            "residual_l1_per_sample": residual_l1_per_sample,
            "side_path_usage": side_path_usage,
            "free_path_usage": free_path_usage,
            "free_level2_usage": free_level2_usage,
            "side_path_representation": side_path_rep,
            "free_path_representation": free_path_rep,
            "free_level2_representation": free_level2_rep,
            "side_path_coefficients": side_path_coefficients,
            "free_path_coefficients": free_path_coefficients,
            "free_level2_coefficients": free_level2_coefficients,
            "side_basis_logits": side_basis_logits,
            "group_pooled_side_rep": group_pooled_side_rep,
            "group_pooled_free_rep": group_pooled_free_rep,
            "group_pooled_side_latent_raw": group_pooled_side_latent_raw,
            "group_pooled_free_latent_raw": group_pooled_free_latent_raw,
            "group_pooled_side_latent": group_pooled_side_latent,
            "group_pooled_free_latent": group_pooled_free_latent,
            "side_loss": {
                "side_loss": side_loss,
                "side_loss_cont": side_loss_cont,
                "side_loss_disc": None,
                "side_loss_per_sample": side_loss_per_sample,
                "side_loss_cont_per_sample": side_loss_cont_per_sample,
                "side_loss_disc_per_sample": None,
                "free_side_adv_loss": free_side_adv_loss,
                "free_side_adv_loss_per_sample": free_side_adv_loss_per_sample,
            },
            "dataset_loss": {
                "private_dataset_loss": dataset_private_loss,
                "shared_dataset_adv_loss": dataset_adv_loss,
                "private_dataset_loss_per_sample": dataset_private_loss_per_sample,
                "shared_dataset_adv_loss_per_sample": dataset_adv_loss_per_sample,
            },
            "side_logits": side_logits,
            "free_side_logits": free_side_logits,
            "discrete_side_logits": None,
            "private_dataset_logits": private_dataset_logits,
            "shared_dataset_logits": shared_dataset_logits,
        }
