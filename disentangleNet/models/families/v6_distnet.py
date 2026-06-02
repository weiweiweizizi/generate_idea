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

from ..basis import (
    collect_runtime_diagnostics,
    enforce_matrix_constraints,
    load_action_basis_init,
)
from ..encoder import build_branch_adapter, build_branch_pool, build_motion_encoder
from ..heads import (
    build_private_decoder,
    build_private_head,
    build_shared_basis_heads,
    build_shared_coeff_heads,
    build_shared_coeff_net,
)
from ..quantizers import build_shared_quantizer, quantize_shared_latent
from ..reconstruction import build_phaseab_outputs, build_shared_reconstruction
from ..sequence_utils import (
    flatten_sequence_input,
    flatten_sequence_labels,
    mean_pool_sequence_tensor,
    reshape_sequence_tensor,
)
from ..side_heads import build_action_side_outputs, build_side_residual_outputs
from ..side_heads.features import fold_mouth_chunk_features


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
        shared_trunk_attention_enabled: bool = False,
        shared_trunk_attention_layers: int = 2,
        shared_trunk_attention_heads: int = 4,
        shared_trunk_attention_ffn_dim: int = 64,
        shared_selection_mode: str = "mlp_coeff",
        use_dataset_aux: bool = False,
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
        basis_provider=None,
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
        if action_side_input == "side_pair_choice_coeff":
            action_side_input = "shared_side_coeff"
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
        self.use_dataset_aux = bool(use_dataset_aux)
        self.shared_trunk_attention_enabled = bool(shared_trunk_attention_enabled)
        self.shared_trunk_attention_layers = int(shared_trunk_attention_layers)
        self.shared_trunk_attention_heads = int(shared_trunk_attention_heads)
        self.shared_trunk_attention_ffn_dim = int(shared_trunk_attention_ffn_dim)
        self.shared_selection_mode = str(shared_selection_mode)
        self.basis_provider = basis_provider
        self.shared_basis_runtime = None

        # TODO(recovery): shared trunk attention and external basis-provider
        # paths are not re-implemented for the current PhaseAB short-run
        # recovery. These fields are accepted here to keep the recovered config
        # interface loadable without pretending the features are active.

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
            self.action_basis_bank = None
            if self.basis_provider is not None:
                self.shared_basis_runtime = self.basis_provider
                self.reflex_basis_bank = self.shared_basis_runtime
            else:
                # TODO(recovery): restore a dense/direct reflex runtime only if a
                # non-low-rank PhaseAB path is brought back intentionally.
                raise RuntimeError(
                    "reflex_basis_enabled requires a basis_provider in the recovered "
                    "PhaseAB path; dense ReflexBasisBank fallback is disabled"
                )
        else:
            if self.basis_provider is not None:
                self.action_basis_bank = None
                self.shared_basis_runtime = self.basis_provider
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
        if self.shared_basis_runtime is not None:
            return self.shared_basis_runtime.get_structured_basis()
        if self.reflex_basis_enabled:
            return self.reflex_basis_bank()
        return self.action_basis_bank

    def _limit_private_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.private_residual_max_l1 is None:
            return residual
        mean_abs = residual.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(mean_abs / float(self.private_residual_max_l1), min=1.0)
        return residual / scale

    def _apply_sparse_basis_topk(self, logits):
        if self.shared_basis_topk is None or not self.training:
            return logits
        k = min(self.shared_basis_topk, logits.shape[-1])
        vals, idx = logits.topk(k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return logits * mask

    def forward(
        self,
        x,
        side_labels=None,
        label5_labels=None,
        dataset_labels=None,
        valid_mask=None,
        static_side_input=None,
        return_group_pooled: bool = False,
    ):
        _ = label5_labels, dataset_labels, static_side_input
        # TODO(recovery): wire `static_side_input` back into the model if future
        # stages re-enable static-side fusion. The current PhaseAB path keeps
        # the argument for interface compatibility only.
        x, sequence_shape = flatten_sequence_input(x)
        side_labels = flatten_sequence_labels(side_labels, sequence_shape)

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

        shared_outputs = build_shared_reconstruction(
            basis=basis,
            levels=self.levels,
            level_quantized_list=level_quantized_list,
            coeffs=coeffs,
            shared_basis_heads=self.shared_basis_heads,
            shared_coeff_heads=self.shared_coeff_heads,
            shared_basis_soft_mixing=self.shared_basis_soft_mixing,
            shared_basis_anchor_bias=self.shared_basis_anchor_bias,
            apply_sparse_basis_topk=self._apply_sparse_basis_topk,
        )
        shared_reconstruction = shared_outputs.shared_reconstruction
        free_path_coefficients = shared_outputs.free_path_coefficients
        free_path_usage = shared_outputs.free_path_usage
        free_path_rep = shared_outputs.free_path_rep
        free_level2_usage = shared_outputs.free_level2_usage
        free_level2_rep = shared_outputs.free_level2_rep
        free_level2_coefficients = shared_outputs.free_level2_coefficients

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
        side_outputs = build_side_residual_outputs(
            x=x,
            side_z=side_z,
            private_z=private_z,
            private_residual=private_residual,
            side_residual_enabled=self.side_residual_enabled,
            side_fold_feature_dim=self.side_fold_feature_dim,
            side_head_input_builder=lambda current_x: fold_mouth_chunk_features(
                current_x,
                side_feature_mode=self.side_feature_mode,
                basis_size=self.basis_size,
            ),
            side_coeff_head=self.side_coeff_head,
            side_basis_bank=self.side_basis_bank,
            private_side_adversary=self.private_side_adversary,
            private_side_grl_lambda=self.private_side_grl_lambda,
            grad_reverse=grad_reverse,
            enforce_matrix_constraints=self._enforce_matrix_constraints,
        )
        fold_features = side_outputs.fold_features
        side_coefficients = side_outputs.side_coefficients
        side_residual = side_outputs.side_residual
        private_side_logits = side_outputs.private_side_logits
        side_coeff_l1 = side_outputs.side_coeff_l1
        side_private_orth = side_outputs.side_private_orth

        reconstructed = (
            shared_reconstruction
            + self.side_residual_weight * side_residual
            - self.private_residual_weight * private_residual
        )

        action_side_outputs = build_action_side_outputs(
            side_residual_enabled=self.side_residual_enabled,
            action_side_input=self.action_side_input,
            action_side_detach=getattr(self, "action_side_detach", False),
            side_coefficients=side_coefficients,
            side_coefficients_seq=reshape_sequence_tensor(
                side_coefficients, sequence_shape,
            ),
            free_path_coefficients=free_path_coefficients,
            free_path_coefficients_seq=reshape_sequence_tensor(
                free_path_coefficients, sequence_shape,
            ),
            free_path_usage=free_path_usage,
            free_path_usage_seq=reshape_sequence_tensor(
                free_path_usage, sequence_shape,
            ),
            valid_mask=valid_mask,
            sequence_shape=sequence_shape,
            mean_pool_sequence_tensor=mean_pool_sequence_tensor,
            side_coeff_to_logits=self.side_coeff_to_logits,
            action_usage_to_side=self.action_usage_to_side,
        )
        group_action_logits = action_side_outputs.group_action_logits
        action_side_representation = action_side_outputs.action_side_representation

        outputs = build_phaseab_outputs(
            sequence_shape=sequence_shape,
            valid_mask=valid_mask,
            return_group_pooled=return_group_pooled,
            reshape_sequence_tensor=reshape_sequence_tensor,
            mean_pool_sequence_tensor=mean_pool_sequence_tensor,
            reconstructed=reconstructed,
            shared_reconstruction=shared_reconstruction,
            side_residual=side_residual,
            private_residual=private_residual,
            free_path_coefficients=free_path_coefficients,
            free_path_usage=free_path_usage,
            free_level2_coefficients=free_level2_coefficients,
            side_coefficients=side_coefficients,
            fold_features=fold_features,
            private_side_logits=private_side_logits,
            free_latent=free_latent,
            side_z=side_z,
            private_z=private_z,
            action_side_representation=action_side_representation,
            group_action_logits=group_action_logits,
            stage_quantized=stage_quantized,
            side_coeff_l1=side_coeff_l1,
            side_private_orth=side_private_orth,
        )
        outputs.pop("_free_path_coefficients_seq", None)
        outputs.pop("_free_path_usage_seq", None)
        outputs.pop("_free_level2_coefficients_seq", None)
        outputs.pop("_side_coefficients_seq", None)
        if self.reflex_basis_enabled:
            outputs.update(
                collect_runtime_diagnostics(
                    self.reflex_basis_bank,
                    orth_key="reflex_orth_loss",
                    diag_prefix="reflex",
                )
            )
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
