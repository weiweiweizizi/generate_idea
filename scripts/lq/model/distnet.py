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

try:
    from .basis import (
        basis_l1_loss,
        enforce_matrix_constraints,
        get_structured_basis,
        load_action_basis_init,
        orthogonality_loss,
        split_basis,
    )
    from .encoder import build_motion_encoder
    from .heads import (
        build_discrete_side_classifier,
        build_group_side_classifier,
        build_private_dataset_classifier,
        build_private_decoder,
        build_private_head,
        build_side_semantic_basis_head,
        build_side_semantic_coeff_head,
        build_shared_basis_heads,
        build_shared_coeff_heads,
        build_shared_coeff_net,
        build_shared_dataset_adversary,
        build_shared_head,
        build_side_classifier,
    )
    from .quantizers import (
        build_shared_quantizer,
        decode_latent_indices,
        quantize_shared_latent,
    )
except ImportError:
    from basis import (
        basis_l1_loss,
        enforce_matrix_constraints,
        get_structured_basis,
        load_action_basis_init,
        orthogonality_loss,
        split_basis,
    )
    from encoder import build_motion_encoder
    from heads import (
        build_discrete_side_classifier,
        build_group_side_classifier,
        build_private_dataset_classifier,
        build_private_decoder,
        build_private_head,
        build_side_semantic_basis_head,
        build_side_semantic_coeff_head,
        build_shared_basis_heads,
        build_shared_coeff_heads,
        build_shared_coeff_net,
        build_shared_dataset_adversary,
        build_shared_head,
        build_side_classifier,
    )
    from quantizers import (
        build_shared_quantizer,
        decode_latent_indices,
        quantize_shared_latent,
    )


class GradientReversalFn(torch.autograd.Function):
    """Straightforward gradient reversal layer used by the optional dataset head."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Convenience wrapper for optional adversarial dataset supervision."""

    return GradientReversalFn.apply(x, lambd)


class DistNet(nn.Module):
    """
    LQ-based motion decomposition network.

    Important conventions:
    - `levels=(2, 3, 6)` means the discrete latent is factorized into 3 groups.
    - `action_basis_bank` must be stored in the same order as those levels.
    - `mode=x` and `mode=y` are handled outside this model; each branch loads
      its own dataset and its own basis initialization tensor.
    """

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
        num_dataset_classes=2,
        private_residual_weight=0.25,
        grl_lambda=1.0,
        use_dataset_aux=False,
        action_basis_init_path=None,
        lq_commitment_loss_weight=0.1,
        lq_quantization_loss_weight=0.1,
        lq_optimize_values=True,
        quantizer_type="latent_quantize",
        fsq_preserve_symmetry=True,
        basis_orthogonalization="normalize",
        private_residual_max_l1=None,
        shared_basis_soft_mixing=False,
        shared_basis_anchor_bias=1.0,
        shared_basis_topk=None,
        side_semantic_enabled=False,
        side_basis_count=0,
        side_pooling="masked_mean",
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
        self.num_dataset_classes = num_dataset_classes
        self.private_residual_weight = private_residual_weight
        self.grl_lambda = grl_lambda
        self.use_dataset_aux = use_dataset_aux
        self.action_basis_init_path = action_basis_init_path
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
        self.side_semantic_enabled = side_semantic_enabled
        self.side_basis_count = int(side_basis_count)
        self.side_pooling = side_pooling

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

        self.shared_head = build_shared_head(self.pooled_dim, hidden_dim, self.shared_dim)
        self.private_head = build_private_head(self.pooled_dim, hidden_dim, private_dim)
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

        self.shared_coeff_net = build_shared_coeff_net(
            self.shared_dim, hidden_dim, len(self.levels)
        )
        self.shared_coeff_heads = build_shared_coeff_heads(
            shared_dim=self.shared_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.shared_basis_heads = build_shared_basis_heads(
            shared_dim=self.shared_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.side_semantic_coeff_head = (
            build_side_semantic_coeff_head(self.shared_dim, hidden_dim)
            if self.side_basis_count > 0
            else None
        )
        self.side_semantic_basis_head = (
            build_side_semantic_basis_head(
                self.shared_dim,
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
        self.side_classifier = build_side_classifier(self.shared_dim, num_side_classes)
        self.discrete_side_classifier = build_discrete_side_classifier(
            self.levels[1], num_side_classes
        )
        self.private_dataset_classifier = build_private_dataset_classifier(
            private_dim, num_dataset_classes
        )
        self.shared_dataset_adversary = build_shared_dataset_adversary(
            self.shared_dim, num_dataset_classes
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
        return get_structured_basis(
            self.action_basis_bank,
            levels=self.levels,
            total_basis_num=self.total_basis_num,
            basis_size=self.basis_size,
            basis_orthogonalization=self.basis_orthogonalization,
        )

    def get_side_basis(self) -> torch.Tensor:
        if self.side_basis_count == 0:
            return self.side_basis_bank
        side_basis = self._enforce_matrix_constraints(self.side_basis_bank)
        side_basis_flat = F.normalize(
            side_basis.reshape(self.side_basis_count, -1),
            dim=1,
            eps=1e-8,
        )
        return side_basis_flat.reshape(self.side_basis_count, self.basis_size, self.basis_size)

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

    def orthogonality_loss(self, basis: torch.Tensor) -> torch.Tensor:
        return orthogonality_loss(basis, self.total_basis_num)

    def basis_l1_loss(self, basis: torch.Tensor) -> torch.Tensor:
        return basis_l1_loss(basis)

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

        pooled = self.avg_pool(feats).flatten(1)
        shared_raw = self.shared_head(pooled)
        private_z = self.private_head(pooled)

        shared_quantized, indices, stage_quantized = self._quantize_shared(shared_raw)
        basis = self.get_structured_basis()
        side_basis = self.get_side_basis()
        basis_list = self.split_basis(basis)
        d_list = self.decode_indices(indices)

        coeffs = None if stage_quantized is not None else self.shared_coeff_net(shared_quantized)
        level_quantized_list = (
            [stage_quantized[:, i] for i in range(stage_quantized.shape[1])]
            if stage_quantized is not None
            else [shared_quantized for _ in self.levels]
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
                    dtype=shared_quantized.dtype,
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

        shared_side_recon = torch.zeros_like(shared_free_recon)
        side_path_usage = shared_quantized.new_zeros((x.shape[0], self.side_basis_count))
        side_path_rep = shared_quantized.new_zeros((x.shape[0], self.side_basis_count))
        side_path_coefficients = shared_quantized.new_zeros((x.shape[0], 1))
        side_basis_logits = None
        if self.side_semantic_enabled and self.side_basis_count > 0:
            side_basis_logits = self.side_semantic_basis_head(shared_quantized)
            side_path_usage = F.softmax(side_basis_logits, dim=-1)
            side_coeff = self.side_semantic_coeff_head(shared_quantized).view(x.shape[0], 1, 1)
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

        commitment_loss_per_sample = F.mse_loss(
            shared_raw.detach(),
            shared_quantized,
            reduction="none",
        ).mean(dim=1)
        quantization_loss_per_sample = F.mse_loss(
            shared_quantized.detach(),
            shared_raw,
            reduction="none",
        ).mean(dim=1)
        if self.quantizer_type == "latent_quantize":
            lq_loss_per_sample = (
                self.lq.commitment_loss_weight * commitment_loss_per_sample
                + self.lq.quantization_loss_weight * quantization_loss_per_sample
            )
        else:
            lq_loss_per_sample = shared_raw.new_zeros(shared_raw.shape[0])
        lq_loss = lq_loss_per_sample.mean()

        side_logits = None
        discrete_side_logits = None
        side_loss = None
        side_loss_cont = None
        side_loss_disc = None
        side_loss_per_sample = None
        side_loss_cont_per_sample = None
        side_loss_disc_per_sample = None
        if side_labels is not None:
            side_logits = self.side_classifier(shared_quantized)
            side_loss_cont_per_sample = F.cross_entropy(
                side_logits, side_labels, reduction="none"
            )
            side_loss_cont = side_loss_cont_per_sample.mean()
            discrete_side_logits = self.discrete_side_classifier(d_list[1])
            side_loss_disc_per_sample = F.cross_entropy(
                discrete_side_logits, side_labels, reduction="none"
            )
            side_loss_disc = side_loss_disc_per_sample.mean()
            side_loss_per_sample = side_loss_cont_per_sample + side_loss_disc_per_sample
            side_loss = side_loss_cont + side_loss_disc

        private_dataset_logits = None
        shared_dataset_logits = None
        dataset_private_loss = None
        dataset_adv_loss = None
        dataset_private_loss_per_sample = None
        dataset_adv_loss_per_sample = None
        if self.use_dataset_aux and dataset_labels is not None:
            private_dataset_logits = self.private_dataset_classifier(private_z)
            dataset_private_loss_per_sample = F.cross_entropy(
                private_dataset_logits,
                dataset_labels,
                reduction="none",
            )
            dataset_private_loss = dataset_private_loss_per_sample.mean()

            shared_dataset_logits = self.shared_dataset_adversary(
                grad_reverse(shared_quantized, self.grl_lambda)
            )
            dataset_adv_loss_per_sample = F.cross_entropy(
                shared_dataset_logits,
                dataset_labels,
                reduction="none",
            )
            dataset_adv_loss = dataset_adv_loss_per_sample.mean()

        orth_loss = self.orthogonality_loss(basis)
        basis_l1 = self.basis_l1_loss(basis)
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
        shared_quantized = self._reshape_sequence_tensor(shared_quantized, sequence_shape)
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
        side_basis_logits = self._reshape_sequence_tensor(side_basis_logits, sequence_shape)
        side_logits = self._reshape_sequence_tensor(side_logits, sequence_shape)
        discrete_side_logits = self._reshape_sequence_tensor(
            discrete_side_logits, sequence_shape
        )
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
        side_loss_disc_per_sample = self._reshape_sequence_tensor(
            side_loss_disc_per_sample, sequence_shape
        )
        dataset_private_loss_per_sample = self._reshape_sequence_tensor(
            dataset_private_loss_per_sample, sequence_shape
        )
        dataset_adv_loss_per_sample = self._reshape_sequence_tensor(
            dataset_adv_loss_per_sample, sequence_shape
        )
        group_pooled_side_rep = (
            self._mean_pool_sequence_tensor(side_path_rep, sequence_shape)
            if return_group_pooled
            else None
        )
        group_pooled_free_rep = (
            self._mean_pool_sequence_tensor(free_path_rep, sequence_shape)
            if return_group_pooled
            else None
        )

        return {
            "reconstructed": reconstructed,
            "action_reconstruction": action_reconstruction,
            "shared_reconstruction": action_reconstruction,
            "shared_side_reconstruction": shared_side_reconstruction,
            "shared_free_reconstruction": shared_free_reconstruction,
            "id_nuisance_residual": private_residual,
            "private_residual": private_residual,
            "shared_quantized": shared_quantized,
            "private_z": private_z,
            "indices": indices,
            "decoded_indices": decoded_indices,
            "action_basis": basis,
            "basis": basis,
            "side_basis": side_basis,
            "lq_loss": lq_loss,
            "lq_loss_per_sample": lq_loss_per_sample,
            "orth_loss": orth_loss,
            "basis_l1": basis_l1,
            "residual_l1": residual_l1,
            "residual_l1_per_sample": residual_l1_per_sample,
            "side_path_usage": side_path_usage,
            "free_path_usage": free_path_usage,
            "side_path_representation": side_path_rep,
            "free_path_representation": free_path_rep,
            "side_path_coefficients": side_path_coefficients,
            "free_path_coefficients": free_path_coefficients,
            "side_basis_logits": side_basis_logits,
            "group_pooled_side_rep": group_pooled_side_rep,
            "group_pooled_free_rep": group_pooled_free_rep,
            "side_loss": {
                "side_loss": side_loss,
                "side_loss_cont": side_loss_cont,
                "side_loss_disc": side_loss_disc,
                "side_loss_per_sample": side_loss_per_sample,
                "side_loss_cont_per_sample": side_loss_cont_per_sample,
                "side_loss_disc_per_sample": side_loss_disc_per_sample,
            },
            "dataset_loss": {
                "private_dataset_loss": dataset_private_loss,
                "shared_dataset_adv_loss": dataset_adv_loss,
                "private_dataset_loss_per_sample": dataset_private_loss_per_sample,
                "shared_dataset_adv_loss_per_sample": dataset_adv_loss_per_sample,
            },
            "side_logits": side_logits,
            "discrete_side_logits": discrete_side_logits,
            "private_dataset_logits": private_dataset_logits,
            "shared_dataset_logits": shared_dataset_logits,
        }
