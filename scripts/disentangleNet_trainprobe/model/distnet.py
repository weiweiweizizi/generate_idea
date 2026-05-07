from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import (
    basis_l1_loss,
    enforce_matrix_constraints,
    get_joint_structured_basis,
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
    build_side_head,
    build_side_semantic_basis_head,
    build_side_semantic_coeff_head,
)
from .quantizers import (
    build_shared_quantizer,
    decode_latent_indices,
    quantize_shared_latent,
)
from ..regions import (
    CHEEK_REGIONS,
    OTHERS_REGIONS,
    UPPER_FACE_REGIONS,
    build_branch_masks,
    project_region_half_to_feature_map,
)


class RegionBranch(nn.Module):
    """One masked branch with its own encoder, free path, and side path."""

    def __init__(
        self,
        *,
        name: str,
        input_mask: torch.Tensor,
        side_token_specs: list[dict[str, object]],
        levels: tuple[int, ...],
        basis_size: int,
        hidden_dim: int,
        free_pool_size: int,
        free_z_dim: int,
        side_z_dim: int,
        side_basis_count: int,
        shared_basis_soft_mixing: bool,
        shared_basis_anchor_bias: float,
        shared_basis_topk: int | None,
        basis_orthogonalization: str,
        quantizer_type: str,
        lq_commitment_loss_weight: float,
        lq_quantization_loss_weight: float,
        lq_optimize_values: bool,
        fsq_preserve_symmetry: bool,
        num_side_classes: int,
        action_basis_init: torch.Tensor | None,
        side_basis_init: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.name = name
        self.levels = tuple(levels)
        self.total_basis_num = sum(self.levels)
        self.basis_size = basis_size
        self.hidden_dim = hidden_dim
        self.free_z_dim = free_z_dim
        self.side_z_dim = side_z_dim
        self.side_basis_count = side_basis_count
        self.shared_basis_soft_mixing = shared_basis_soft_mixing
        self.shared_basis_anchor_bias = shared_basis_anchor_bias
        self.shared_basis_topk = shared_basis_topk
        self.basis_orthogonalization = basis_orthogonalization
        self.side_token_specs = side_token_specs

        self.register_buffer("input_mask", input_mask.unsqueeze(0).unsqueeze(0))
        self.register_buffer("basis_mask", input_mask)

        (
            self.initial_conv,
            self.pre_layer1_block,
            self.layer1,
            self.pre_layer2_block,
            self.layer2,
            self.layer3,
            self.avg_pool,
        ) = build_motion_encoder(hidden_dim, 1)

        self.free_adapter = build_branch_adapter(hidden_dim)
        self.side_adapter = build_branch_adapter(hidden_dim)
        self.free_pool = build_branch_pool(free_pool_size)
        self.free_pooled_dim = hidden_dim * free_pool_size * free_pool_size
        self.side_pooled_dim = hidden_dim * len(self.side_token_specs)

        self.free_head = build_free_head(self.free_pooled_dim, hidden_dim, free_z_dim)
        self.side_head = build_side_head(self.side_pooled_dim, hidden_dim, side_z_dim)
        self.lq, self.residual_fsq_layers = build_shared_quantizer(
            quantizer_type=quantizer_type,
            levels=self.levels,
            shared_dim=free_z_dim,
            lq_commitment_loss_weight=lq_commitment_loss_weight,
            lq_quantization_loss_weight=lq_quantization_loss_weight,
            lq_optimize_values=lq_optimize_values,
            fsq_preserve_symmetry=fsq_preserve_symmetry,
        )

        self.action_basis_bank = nn.Parameter(
            torch.randn(self.total_basis_num, basis_size, basis_size) * 0.02
        )
        self.side_basis_bank = nn.Parameter(
            torch.randn(side_basis_count, basis_size, basis_size) * 0.02
        )
        if action_basis_init is not None:
            with torch.no_grad():
                self.action_basis_bank.copy_(action_basis_init * self.basis_mask)
        if side_basis_init is not None:
            with torch.no_grad():
                self.side_basis_bank.copy_(side_basis_init * self.basis_mask)

        self.shared_coeff_net = build_shared_coeff_net(free_z_dim, hidden_dim, len(self.levels))
        self.shared_coeff_heads = build_shared_coeff_heads(
            shared_dim=free_z_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.shared_basis_heads = build_shared_basis_heads(
            shared_dim=free_z_dim,
            hidden_dim=hidden_dim,
            levels=self.levels,
        )
        self.side_semantic_coeff_head = build_side_semantic_coeff_head(side_z_dim, hidden_dim)
        self.side_semantic_basis_head = build_side_semantic_basis_head(
            side_z_dim,
            hidden_dim,
            side_basis_count,
        )
        self.group_side_classifier = build_group_side_classifier(side_basis_count, num_side_classes)

    def _masked_basis_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_basis_bank * self.basis_mask, self.side_basis_bank * self.basis_mask

    def get_structured_basis_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        action_basis_bank, side_basis_bank = self._masked_basis_banks()
        return get_joint_structured_basis(
            action_basis_bank,
            side_basis_bank,
            levels=self.levels,
            total_basis_num=self.total_basis_num,
            side_basis_count=self.side_basis_count,
            basis_size=self.basis_size,
            basis_orthogonalization=self.basis_orthogonalization,
        )

    def get_masked_raw_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._masked_basis_banks()

    def _apply_sparse_basis_topk(self, level_logits: torch.Tensor) -> torch.Tensor:
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

    def _mean_region_blocks(
        self,
        side_feats: torch.Tensor,
        row_regions: tuple[str, ...],
        col_regions: tuple[str, ...],
        half: str,
    ) -> torch.Tensor:
        feature_size = side_feats.shape[-1]
        pooled = []
        for row_region in row_regions:
            row_slice = project_region_half_to_feature_map(row_region, half, feature_size)
            for col_region in col_regions:
                col_slice = project_region_half_to_feature_map(col_region, half, feature_size)
                block = side_feats[:, :, row_slice, col_slice]
                pooled.append(block.mean(dim=(2, 3)))
        return torch.stack(pooled, dim=0).mean(dim=0)

    def _pool_side_tokens(self, side_feats: torch.Tensor) -> torch.Tensor:
        if side_feats.ndim != 4 or side_feats.shape[-1] != side_feats.shape[-2]:
            raise ValueError(
                f"Expected square side feature map [N, C, H, W], got {tuple(side_feats.shape)}"
            )

        tokens = []
        for spec in self.side_token_specs:
            spec_type = str(spec["type"])
            if spec_type == "self":
                region_group = tuple(spec["regions"])
                left = self._mean_region_blocks(side_feats, region_group, region_group, "left")
                right = self._mean_region_blocks(side_feats, region_group, region_group, "right")
            elif spec_type == "pair":
                row_regions = tuple(spec["row_regions"])
                col_regions = tuple(spec["col_regions"])
                left = 0.5 * (
                    self._mean_region_blocks(side_feats, row_regions, col_regions, "left")
                    + self._mean_region_blocks(side_feats, col_regions, row_regions, "left")
                )
                right = 0.5 * (
                    self._mean_region_blocks(side_feats, row_regions, col_regions, "right")
                    + self._mean_region_blocks(side_feats, col_regions, row_regions, "right")
                )
            else:
                raise ValueError(f"Unsupported side token spec type: {spec_type}")
            tokens.append(left - right)

        return torch.cat(tokens, dim=1)

    def _mean_pool_sequence_tensor(
        self,
        tensor: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if sequence_shape is None:
            return tensor
        batch_size, seq_len = sequence_shape
        return tensor.reshape(batch_size, seq_len, *tensor.shape[1:]).mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sequence_shape: tuple[int, int] | None,
    ) -> dict[str, torch.Tensor | None]:
        x = x * self.input_mask.to(device=x.device, dtype=x.dtype)

        feats = self.initial_conv(x)
        feats = self.pre_layer1_block(feats)
        feats = self.layer1(feats)
        feats = self.pre_layer2_block(feats)
        feats = self.layer2(feats)
        feats = self.layer3(feats)

        free_feats = self.free_adapter(feats)
        side_feats = self.side_adapter(feats)
        free_pooled = self.free_pool(free_feats).flatten(1)
        side_pooled = self._pool_side_tokens(side_feats)

        free_raw = self.free_head(free_pooled)
        side_latent = self.side_head(side_pooled)

        free_quantized, indices, stage_quantized = self._quantize_shared(free_raw)
        free_latent = free_quantized
        basis, side_basis = self.get_structured_basis_pair()
        basis_list = split_basis(basis, self.levels)
        d_list = decode_latent_indices(
            indices,
            quantizer_type="residual_fsq",
            levels=self.levels,
            lq=self.lq,
        )

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
                coeff = self.shared_coeff_heads[level_idx](level_quantized_i).view(x.shape[0], 1, 1)
            else:
                coeff = coeffs[:, level_idx].view(x.shape[0], 1, 1)

            shared_free_recon = shared_free_recon + coeff * selected_basis
            free_path_coeff_levels.append(coeff.view(x.shape[0], 1))
            free_path_usage_levels.append(level_weights)
            free_path_rep_levels.append(level_weights * coeff.view(x.shape[0], 1))

        side_basis_logits = self.side_semantic_basis_head(side_latent)
        side_path_usage = F.softmax(side_basis_logits, dim=-1)
        side_coeff = self.side_semantic_coeff_head(side_latent).view(x.shape[0], 1, 1)
        side_path_coefficients = side_coeff.view(x.shape[0], 1)
        side_path_rep = side_path_usage * side_path_coefficients
        selected_side_basis = torch.einsum("bs,sxy->bxy", side_path_usage, side_basis)
        shared_side_recon = side_coeff * selected_side_basis

        shared_recon = shared_free_recon + shared_side_recon
        lq_loss_per_sample = free_raw.new_zeros(free_raw.shape[0])

        side_path_rep_group = self._mean_pool_sequence_tensor(side_path_rep, sequence_shape)
        group_side_logits = self.group_side_classifier(side_path_rep_group)

        return {
            "shared_reconstruction_raw": shared_recon,
            "shared_free_reconstruction_raw": shared_free_recon,
            "shared_side_reconstruction_raw": shared_side_recon,
            "lq_loss_per_sample": lq_loss_per_sample,
            "side_path_representation": side_path_rep,
            "group_side_logits": group_side_logits,
            "side_path_usage": side_path_usage,
            "free_path_usage": torch.cat(free_path_usage_levels, dim=1),
            "free_path_representation": torch.cat(free_path_rep_levels, dim=1),
            "free_path_coefficients": torch.cat(free_path_coeff_levels, dim=1),
            "side_path_coefficients": side_path_coefficients,
            "side_latent": side_latent,
            "free_latent": free_latent,
            "indices": indices,
        }

    def _quantize_shared(self, shared_raw: torch.Tensor):
        return quantize_shared_latent(
            shared_raw,
            quantizer_type="residual_fsq",
            lq=self.lq,
            residual_fsq_layers=self.residual_fsq_layers,
        )


class DistNet(nn.Module):
    """Tri-region masked trainprobe network with independent free/side branches."""

    BRANCH_NAMES = ("mouth_self", "mouth_cross_other", "other_self")

    def __init__(
        self,
        side_label=None,
        levels=(2, 6),
        basis_size=341,
        hidden_dim=32,
        pool_size=1,
        shared_dim=None,
        private_dim=32,
        private_decoder_hidden_dim=None,
        num_side_classes=3,
        num_severity_classes=3,
        num_dataset_classes=2,
        target_label_mode="side",
        private_residual_weight=0.05,
        grl_lambda=1.0,
        use_dataset_aux=False,
        action_basis_init_path=None,
        side_basis_init_path=None,
        lq_commitment_loss_weight=0.1,
        lq_quantization_loss_weight=0.1,
        lq_optimize_values=True,
        quantizer_type="residual_fsq",
        fsq_preserve_symmetry=True,
        basis_orthogonalization="joint_global_qr",
        discrete_side_loss_enabled=False,
        private_residual_max_l1=0.5,
        shared_basis_soft_mixing=True,
        shared_basis_anchor_bias=2.0,
        shared_basis_topk=2,
        side_semantic_enabled=True,
        side_basis_count=3,
        side_pooling="tri_region_contrast",
        side_subspace_dim=None,
        side_free_frame_qr=False,
        free_side_grl_lambda=1.0,
        early_branch_factorization=True,
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
        self.num_side_classes = num_side_classes
        self.num_severity_classes = num_severity_classes
        self.num_dataset_classes = num_dataset_classes
        self.target_label_mode = "side"
        self.requested_target_label_mode = str(target_label_mode)
        self.private_residual_weight = float(private_residual_weight)
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
        self.side_semantic_enabled = bool(side_semantic_enabled)
        self.side_basis_count = int(side_basis_count)
        self.side_pooling = side_pooling
        self.free_side_grl_lambda = free_side_grl_lambda
        self.early_branch_factorization = bool(early_branch_factorization)
        self.free_pool_size = int(free_pool_size)
        self.private_pool_size = int(private_pool_size)
        self.free_z_dim = int(free_z_dim if free_z_dim is not None else hidden_dim)
        self.side_z_dim = int(side_z_dim if side_z_dim is not None else hidden_dim)
        self.private_dim = int(private_dim)
        self.private_decoder_hidden_dim = (
            int(private_decoder_hidden_dim)
            if private_decoder_hidden_dim is not None
            else hidden_dim * 2
        )
        self.private_adapter_enabled = bool(private_adapter_enabled)

        if self.use_dataset_aux:
            raise ValueError("disentangleNet_trainprobe does not support dataset auxiliary heads")
        if self.basis_size != 341:
            raise ValueError("disentangleNet_trainprobe requires basis_size=341")
        if self.quantizer_type != "residual_fsq":
            raise ValueError("disentangleNet_trainprobe requires quantizer_type='residual_fsq'")
        if self.discrete_side_loss_enabled:
            raise ValueError("disentangleNet_trainprobe requires discrete_side_loss_enabled=False")
        if not self.early_branch_factorization:
            raise ValueError("disentangleNet_trainprobe requires early_branch_factorization=True")
        if not self.side_semantic_enabled:
            raise ValueError("disentangleNet_trainprobe requires side_semantic_enabled=True")
        if self.side_basis_count != 3:
            raise ValueError("disentangleNet_trainprobe requires side_basis_count=3")

        action_basis_init = None
        side_basis_init = None
        if action_basis_init_path is not None:
            action_basis_init = torch.from_numpy(np.load(action_basis_init_path)).float()
        if side_basis_init_path is not None:
            side_basis_init = torch.from_numpy(np.load(side_basis_init_path)).float()
        if action_basis_init is not None and action_basis_init.shape[0] == self.total_basis_num + self.side_basis_count:
            full_action_basis_init = action_basis_init[: self.total_basis_num]
        else:
            full_action_basis_init = action_basis_init
        if side_basis_init is not None and side_basis_init.shape[0] == self.total_basis_num + self.side_basis_count:
            full_side_basis_init = side_basis_init[-self.side_basis_count :]
        else:
            full_side_basis_init = side_basis_init
        if full_action_basis_init is not None and tuple(full_action_basis_init.shape) != (
            self.total_basis_num,
            self.basis_size,
            self.basis_size,
        ):
            raise ValueError(
                "action_basis_init_path must contain either "
                f"{(self.total_basis_num, self.basis_size, self.basis_size)} or "
                f"{(self.total_basis_num + self.side_basis_count, self.basis_size, self.basis_size)}, "
                f"got {tuple(action_basis_init.shape)}"
            )
        if full_side_basis_init is not None and tuple(full_side_basis_init.shape) != (
            self.side_basis_count,
            self.basis_size,
            self.basis_size,
        ):
            raise ValueError(
                "side_basis_init_path must contain either "
                f"{(self.side_basis_count, self.basis_size, self.basis_size)} or "
                f"{(self.total_basis_num + self.side_basis_count, self.basis_size, self.basis_size)}, "
                f"got {tuple(side_basis_init.shape)}"
            )

        branch_masks = build_branch_masks()
        branch_side_specs = {
            "mouth_self": [
                {"type": "self", "regions": ("around_mouth",)},
                {"type": "self", "regions": ("mouth",)},
            ],
            "mouth_cross_other": [
                {
                    "type": "pair",
                    "row_regions": ("around_mouth",),
                    "col_regions": UPPER_FACE_REGIONS,
                },
                {
                    "type": "pair",
                    "row_regions": ("mouth",),
                    "col_regions": UPPER_FACE_REGIONS,
                },
                {
                    "type": "pair",
                    "row_regions": ("around_mouth",),
                    "col_regions": CHEEK_REGIONS,
                },
                {
                    "type": "pair",
                    "row_regions": ("mouth",),
                    "col_regions": CHEEK_REGIONS,
                },
                {
                    "type": "pair",
                    "row_regions": ("around_mouth",),
                    "col_regions": OTHERS_REGIONS,
                },
                {
                    "type": "pair",
                    "row_regions": ("mouth",),
                    "col_regions": OTHERS_REGIONS,
                },
            ],
            "other_self": [
                {"type": "self", "regions": UPPER_FACE_REGIONS},
                {"type": "self", "regions": CHEEK_REGIONS},
                {"type": "self", "regions": OTHERS_REGIONS},
            ],
        }

        self.branches = nn.ModuleDict(
            {
                name: RegionBranch(
                    name=name,
                    input_mask=torch.from_numpy(branch_masks[name]).float(),
                    side_token_specs=branch_side_specs[name],
                    levels=self.levels,
                    basis_size=basis_size,
                    hidden_dim=hidden_dim,
                    free_pool_size=self.free_pool_size,
                    free_z_dim=self.free_z_dim,
                    side_z_dim=self.side_z_dim,
                    side_basis_count=self.side_basis_count,
                    shared_basis_soft_mixing=self.shared_basis_soft_mixing,
                    shared_basis_anchor_bias=self.shared_basis_anchor_bias,
                    shared_basis_topk=self.shared_basis_topk,
                    basis_orthogonalization=self.basis_orthogonalization,
                    quantizer_type=self.quantizer_type,
                    lq_commitment_loss_weight=self.lq_commitment_loss_weight,
                    lq_quantization_loss_weight=self.lq_quantization_loss_weight,
                    lq_optimize_values=self.lq_optimize_values,
                    fsq_preserve_symmetry=self.fsq_preserve_symmetry,
                    num_side_classes=num_side_classes,
                    action_basis_init=full_action_basis_init,
                    side_basis_init=full_side_basis_init,
                )
                for name in self.BRANCH_NAMES
            }
        )

        (
            self.private_initial_conv,
            self.private_pre_layer1_block,
            self.private_layer1,
            self.private_pre_layer2_block,
            self.private_layer2,
            self.private_layer3,
            self.private_avg_pool,
        ) = build_motion_encoder(hidden_dim, 1)
        self.private_adapter = (
            build_branch_adapter(hidden_dim) if self.private_adapter_enabled else None
        )
        self.private_pool = build_branch_pool(self.private_pool_size)
        self.private_pooled_dim = hidden_dim * self.private_pool_size * self.private_pool_size
        self.private_head = build_private_head(
            self.private_pooled_dim,
            hidden_dim,
            self.private_dim,
        )
        self.private_decoder = build_private_decoder(
            private_dim=self.private_dim,
            private_decoder_hidden_dim=self.private_decoder_hidden_dim,
            basis_size=basis_size,
        )

    @staticmethod
    def _flatten_sequence_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
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
        if labels is None or sequence_shape is None:
            return labels
        batch_size, seq_len = sequence_shape
        if labels.ndim == 1:
            labels = labels.unsqueeze(1).expand(batch_size, seq_len)
        elif labels.ndim != 2 or tuple(labels.shape) != (batch_size, seq_len):
            raise ValueError(
                f"Expected sequence labels with shape {(batch_size, seq_len)}, got {tuple(labels.shape)}"
            )
        return labels.reshape(batch_size * seq_len)

    @staticmethod
    def _reshape_sequence_tensor(
        tensor: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if tensor is None or sequence_shape is None:
            return tensor
        batch_size, seq_len = sequence_shape
        return tensor.reshape(batch_size, seq_len, *tensor.shape[1:])

    @staticmethod
    def _reshape_sequence_index_list(
        index_list: list[torch.Tensor] | None,
        sequence_shape: tuple[int, int] | None,
    ) -> list[torch.Tensor] | None:
        if index_list is None or sequence_shape is None:
            return index_list
        batch_size, seq_len = sequence_shape
        return [
            indices.reshape(batch_size, seq_len, *indices.shape[1:])
            for indices in index_list
        ]

    @staticmethod
    def _mean_pool_sequence_tensor(
        tensor: torch.Tensor | None,
        sequence_shape: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if tensor is None or sequence_shape is None:
            return tensor
        batch_size, seq_len = sequence_shape
        return tensor.reshape(batch_size, seq_len, *tensor.shape[1:]).mean(dim=1)

    def _stack_branch_feature(
        self,
        branch_outputs: dict[str, dict[str, torch.Tensor | None]],
        key: str,
    ) -> torch.Tensor | None:
        values = []
        for branch_name in self.BRANCH_NAMES:
            value = branch_outputs[branch_name].get(key)
            if value is None:
                return None
            values.append(value)
        return torch.stack(values, dim=0)

    def _aggregate_branch_usage(
        self,
        branch_outputs: dict[str, dict[str, torch.Tensor | None]],
        key: str,
    ) -> torch.Tensor | None:
        stacked = self._stack_branch_feature(branch_outputs, key)
        if stacked is None:
            return None
        return stacked.mean(dim=0)

    def _aggregate_branch_representation(
        self,
        branch_outputs: dict[str, dict[str, torch.Tensor | None]],
        key: str,
    ) -> torch.Tensor | None:
        stacked = self._stack_branch_feature(branch_outputs, key)
        if stacked is None:
            return None
        return stacked.sum(dim=0)

    def _derive_decoded_indices_from_usage(
        self,
        free_path_usage: torch.Tensor | None,
    ) -> list[torch.Tensor] | None:
        if free_path_usage is None:
            return None

        indices = []
        start = 0
        for level_size in self.levels:
            level_usage = free_path_usage[:, start : start + level_size]
            indices.append(level_usage.argmax(dim=-1))
            start += level_size
        return indices

    def _limit_private_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.private_residual_max_l1 is None:
            return residual

        mean_abs = residual.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(mean_abs / float(self.private_residual_max_l1), min=1.0)
        return residual / scale

    def _forward_private_branch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.private_initial_conv(x)
        feats = self.private_pre_layer1_block(feats)
        feats = self.private_layer1(feats)
        feats = self.private_pre_layer2_block(feats)
        feats = self.private_layer2(feats)
        feats = self.private_layer3(feats)

        private_feats = self.private_adapter(feats) if self.private_adapter is not None else feats
        private_pooled = self.private_pool(private_feats).flatten(1)
        private_z = self.private_head(private_pooled)
        private_residual = self.private_decoder(private_z).reshape(
            x.shape[0],
            self.basis_size,
            self.basis_size,
        )
        private_residual = enforce_matrix_constraints(private_residual)
        private_residual = self._limit_private_residual(private_residual)
        return private_residual, private_z

    def _branch_structured_basis_pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [branch.get_structured_basis_pair() for branch in self.branches.values()]

    def _combined_basis_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        shared_basis = None
        side_basis = None
        for branch_shared, branch_side in self._branch_structured_basis_pairs():
            shared_basis = branch_shared if shared_basis is None else shared_basis + branch_shared
            side_basis = branch_side if side_basis is None else side_basis + branch_side
        assert shared_basis is not None
        assert side_basis is not None
        return shared_basis, side_basis

    def _branch_orthogonality_loss(self) -> torch.Tensor:
        branch_losses = []
        for branch in self.branches.values():
            branch_shared, branch_side = branch.get_structured_basis_pair()
            branch_all = torch.cat([branch_shared, branch_side], dim=0)
            branch_losses.append(orthogonality_loss(branch_all, branch_all.shape[0]))
        return torch.stack(branch_losses).mean()

    def _mean_branch_group_logits(
        self,
        branch_outputs: dict[str, dict[str, torch.Tensor | None]],
    ) -> torch.Tensor | None:
        logits = [
            branch_out["group_side_logits"]
            for branch_out in branch_outputs.values()
            if branch_out["group_side_logits"] is not None
        ]
        if not logits:
            return None
        return torch.stack(logits, dim=0).mean(dim=0)

    def get_structured_basis(self) -> torch.Tensor:
        shared_basis, _ = self._combined_basis_pair()
        return shared_basis

    def get_side_basis(self) -> torch.Tensor:
        _, side_basis = self._combined_basis_pair()
        return side_basis

    def forward(
        self,
        x,
        side_labels=None,
        label5_labels=None,
        dataset_labels=None,
        return_group_pooled: bool = False,
    ):
        x, sequence_shape = self._flatten_sequence_input(x)
        side_labels = self._flatten_sequence_labels(side_labels, sequence_shape)
        _ = self._flatten_sequence_labels(label5_labels, sequence_shape)
        _ = self._flatten_sequence_labels(dataset_labels, sequence_shape)

        branch_outputs = {
            name: branch(x, sequence_shape=sequence_shape)
            for name, branch in self.branches.items()
        }

        shared_recon = sum(branch_out["shared_reconstruction_raw"] for branch_out in branch_outputs.values())
        shared_free_recon = sum(
            branch_out["shared_free_reconstruction_raw"] for branch_out in branch_outputs.values()
        )
        shared_side_recon = sum(
            branch_out["shared_side_reconstruction_raw"] for branch_out in branch_outputs.values()
        )
        private_residual_raw, private_z = self._forward_private_branch(x)
        recon_raw = shared_recon + self.private_residual_weight * private_residual_raw

        combined_shared_basis, combined_side_basis = self._combined_basis_pair()
        orth_loss = self._branch_orthogonality_loss()
        shared_basis_l1 = basis_l1_loss(combined_shared_basis)
        side_basis_l1 = basis_l1_loss(combined_side_basis)
        basis_l1 = shared_basis_l1 + side_basis_l1

        reconstructed = self._reshape_sequence_tensor(
            enforce_matrix_constraints(recon_raw).unsqueeze(1),
            sequence_shape,
        )
        action_reconstruction = self._reshape_sequence_tensor(
            enforce_matrix_constraints(shared_recon).unsqueeze(1),
            sequence_shape,
        )
        shared_side_reconstruction = self._reshape_sequence_tensor(
            enforce_matrix_constraints(shared_side_recon).unsqueeze(1),
            sequence_shape,
        )
        shared_free_reconstruction = self._reshape_sequence_tensor(
            enforce_matrix_constraints(shared_free_recon).unsqueeze(1),
            sequence_shape,
        )

        private_residual = self._reshape_sequence_tensor(
            private_residual_raw.unsqueeze(1),
            sequence_shape,
        )
        lq_loss_per_sample = torch.stack(
            [branch_out["lq_loss_per_sample"] for branch_out in branch_outputs.values()],
            dim=0,
        ).mean(dim=0)
        residual_l1_per_sample = private_residual_raw.abs().mean(dim=(1, 2))

        branch_group_side_logits = {
            name: branch_out["group_side_logits"] for name, branch_out in branch_outputs.items()
        }
        group_side_logits = self._mean_branch_group_logits(branch_outputs)

        free_path_usage_raw = self._aggregate_branch_usage(
            branch_outputs,
            "free_path_usage",
        )
        side_path_usage_raw = self._aggregate_branch_usage(
            branch_outputs,
            "side_path_usage",
        )
        free_path_rep_raw = self._aggregate_branch_representation(
            branch_outputs,
            "free_path_representation",
        )
        side_path_rep_raw = self._aggregate_branch_representation(
            branch_outputs,
            "side_path_representation",
        )
        free_path_coefficients_raw = self._aggregate_branch_usage(
            branch_outputs,
            "free_path_coefficients",
        )
        side_path_coefficients_raw = self._aggregate_branch_usage(
            branch_outputs,
            "side_path_coefficients",
        )
        decoded_indices_raw = self._derive_decoded_indices_from_usage(free_path_usage_raw)
        free_level2_usage_raw = None
        free_level2_rep_raw = None
        free_level2_coefficients_raw = None
        if free_path_usage_raw is not None and len(self.levels) >= 2:
            level2_start = int(self.levels[0])
            level2_end = level2_start + int(self.levels[1])
            free_level2_usage_raw = free_path_usage_raw[:, level2_start:level2_end]
        if free_path_rep_raw is not None and len(self.levels) >= 2:
            level2_start = int(self.levels[0])
            level2_end = level2_start + int(self.levels[1])
            free_level2_rep_raw = free_path_rep_raw[:, level2_start:level2_end]
        if free_path_coefficients_raw is not None and free_path_coefficients_raw.shape[1] >= 2:
            free_level2_coefficients_raw = free_path_coefficients_raw[:, 1:2]

        branch_basis_pairs = {
            name: self.branches[name].get_structured_basis_pair() for name in self.BRANCH_NAMES
        }
        branch_action_basis = {
            name: pair[0] for name, pair in branch_basis_pairs.items()
        }
        branch_side_basis = {
            name: pair[1] for name, pair in branch_basis_pairs.items()
        }
        branch_free_path_usage = {
            name: self._reshape_sequence_tensor(branch_outputs[name]["free_path_usage"], sequence_shape)
            for name in self.BRANCH_NAMES
        }
        branch_side_path_usage = {
            name: self._reshape_sequence_tensor(branch_outputs[name]["side_path_usage"], sequence_shape)
            for name in self.BRANCH_NAMES
        }
        branch_free_path_representation = {
            name: self._reshape_sequence_tensor(
                branch_outputs[name]["free_path_representation"],
                sequence_shape,
            )
            for name in self.BRANCH_NAMES
        }
        branch_side_path_representation = {
            name: self._reshape_sequence_tensor(
                branch_outputs[name]["side_path_representation"],
                sequence_shape,
            )
            for name in self.BRANCH_NAMES
        }
        branch_free_path_coefficients = {
            name: self._reshape_sequence_tensor(
                branch_outputs[name]["free_path_coefficients"],
                sequence_shape,
            )
            for name in self.BRANCH_NAMES
        }
        branch_side_path_coefficients = {
            name: self._reshape_sequence_tensor(
                branch_outputs[name]["side_path_coefficients"],
                sequence_shape,
            )
            for name in self.BRANCH_NAMES
        }
        branch_decoded_indices = {
            name: self._reshape_sequence_index_list(
                decode_latent_indices(
                    branch_outputs[name]["indices"],
                    quantizer_type="residual_fsq",
                    levels=self.levels,
                    lq=self.branches[name].lq,
                ),
                sequence_shape,
            )
            for name in self.BRANCH_NAMES
        }

        free_path_usage = self._reshape_sequence_tensor(free_path_usage_raw, sequence_shape)
        side_path_usage = self._reshape_sequence_tensor(side_path_usage_raw, sequence_shape)
        free_path_representation = self._reshape_sequence_tensor(free_path_rep_raw, sequence_shape)
        side_path_representation = self._reshape_sequence_tensor(side_path_rep_raw, sequence_shape)
        free_path_coefficients = self._reshape_sequence_tensor(
            free_path_coefficients_raw,
            sequence_shape,
        )
        side_path_coefficients = self._reshape_sequence_tensor(
            side_path_coefficients_raw,
            sequence_shape,
        )
        decoded_indices = self._reshape_sequence_index_list(decoded_indices_raw, sequence_shape)
        free_level2_usage = self._reshape_sequence_tensor(free_level2_usage_raw, sequence_shape)
        free_level2_representation = self._reshape_sequence_tensor(
            free_level2_rep_raw,
            sequence_shape,
        )
        free_level2_coefficients = self._reshape_sequence_tensor(
            free_level2_coefficients_raw,
            sequence_shape,
        )
        group_pooled_side_rep = (
            self._mean_pool_sequence_tensor(side_path_rep_raw, sequence_shape)
            if return_group_pooled
            else None
        )
        group_pooled_free_rep = (
            self._mean_pool_sequence_tensor(free_path_rep_raw, sequence_shape)
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
            "shared_quantized": None,
            "side_latent_raw": None,
            "free_latent_raw": None,
            "side_latent": None,
            "free_latent": None,
            "private_z": self._reshape_sequence_tensor(private_z, sequence_shape),
            "indices": None,
            "decoded_indices": decoded_indices,
            "action_basis": combined_shared_basis,
            "basis": combined_shared_basis,
            "side_basis": combined_side_basis,
            "lq_loss": lq_loss_per_sample.mean(),
            "lq_loss_per_sample": self._reshape_sequence_tensor(lq_loss_per_sample, sequence_shape),
            "orth_loss": orth_loss,
            "shared_basis_l1": shared_basis_l1,
            "side_basis_l1": side_basis_l1,
            "basis_l1": basis_l1,
            "residual_l1": residual_l1_per_sample.mean(),
            "residual_l1_per_sample": self._reshape_sequence_tensor(residual_l1_per_sample, sequence_shape),
            "side_path_usage": side_path_usage,
            "free_path_usage": free_path_usage,
            "free_level2_usage": free_level2_usage,
            "side_path_representation": side_path_representation,
            "free_path_representation": free_path_representation,
            "free_level2_representation": free_level2_representation,
            "side_path_coefficients": side_path_coefficients,
            "free_path_coefficients": free_path_coefficients,
            "free_level2_coefficients": free_level2_coefficients,
            "side_basis_logits": None,
            "side_logits": None,
            "group_side_logits": group_side_logits,
            "branch_group_side_logits": branch_group_side_logits,
            "group_pooled_side_rep": group_pooled_side_rep,
            "group_pooled_free_rep": group_pooled_free_rep,
            "group_pooled_side_latent_raw": None,
            "group_pooled_free_latent_raw": None,
            "group_pooled_side_latent": None,
            "group_pooled_free_latent": None,
            "branch_action_basis": branch_action_basis,
            "branch_side_basis": branch_side_basis,
            "branch_free_path_usage": branch_free_path_usage,
            "branch_side_path_usage": branch_side_path_usage,
            "branch_free_path_representation": branch_free_path_representation,
            "branch_side_path_representation": branch_side_path_representation,
            "branch_free_path_coefficients": branch_free_path_coefficients,
            "branch_side_path_coefficients": branch_side_path_coefficients,
            "branch_decoded_indices": branch_decoded_indices,
            "side_loss": {
                "side_loss": None,
                "side_loss_cont": None,
                "side_loss_disc": None,
                "side_loss_per_sample": None,
                "side_loss_cont_per_sample": None,
                "side_loss_disc_per_sample": None,
                "free_side_adv_loss": None,
                "free_side_adv_loss_per_sample": None,
            },
            "dataset_loss": {
                "private_dataset_loss": None,
                "shared_dataset_adv_loss": None,
                "private_dataset_loss_per_sample": None,
                "shared_dataset_adv_loss_per_sample": None,
            },
            "free_side_logits": None,
            "discrete_side_logits": None,
            "private_dataset_logits": None,
            "shared_dataset_logits": None,
        }
