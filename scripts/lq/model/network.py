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

The design goal is not perfect disentanglement yet; it is a pragmatic first
step that keeps the "shared interpretable motion basis" idea explicit while
still giving the model somewhere to put patient-specific leftovers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .BasicBlock import BasicBlock
    from .latent_quantization import LatentQuantize
except ImportError:
    try:
        from model.BasicBlock import BasicBlock
        from model.latent_quantization import LatentQuantize
    except ImportError:
        from BasicBlock import BasicBlock
        from latent_quantization import LatentQuantize


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
        private_dim=32,
        num_side_classes=3,
        num_dataset_classes=2,
        private_residual_weight=0.25,
        grl_lambda=1.0,
        use_dataset_aux=False,
        action_basis_init_path=None,
    ):
        super().__init__()

        self.levels = tuple(levels)
        self.total_basis_num = sum(self.levels)
        self.labels = side_label
        self.basis_size = basis_size
        self.hidden_dim = hidden_dim
        self.private_dim = private_dim
        self.num_side_classes = num_side_classes
        self.num_dataset_classes = num_dataset_classes
        self.private_residual_weight = private_residual_weight
        self.grl_lambda = grl_lambda
        self.use_dataset_aux = use_dataset_aux
        self.action_basis_init_path = action_basis_init_path

        # Small CNN stem: enough to compress a 119x119 motion matrix into a
        # compact latent without immediately introducing a heavy backbone.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # Residual encoder blocks. This encoder is intentionally shallow for now
        # because interpretability experiments matter more than raw scale.
        self.layer1 = BasicBlock(
            8,
            16,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(16),
            ),
        )
        self.layer2 = BasicBlock(
            16,
            hidden_dim,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(16, hidden_dim, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ),
        )
        self.layer3 = BasicBlock(hidden_dim, hidden_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared branch feeds the quantizer. This branch is where we want the
        # model to encode cross-patient motion patterns that can be explained by
        # a small set of shared action bases.
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Private branch captures whatever the shared bases cannot explain well:
        # identity variation, nuisance factors, local idiosyncrasies, etc.
        self.private_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, private_dim),
        )

        self.lq = LatentQuantize(
            levels=self.levels,
            dim=hidden_dim,
            commitment_loss_weight=0.1,
            quantization_loss_weight=0.1,
        )

        # Shared action basis bank. Each row is one candidate motion prototype.
        # During forward, one basis is selected from each level partition and
        # then weighted by a learned scalar coefficient.
        self.action_basis_bank = nn.Parameter(
            torch.randn(self.total_basis_num, basis_size, basis_size) * 0.02
        )
        if action_basis_init_path is not None:
            self._load_action_basis_init(action_basis_init_path)

        self.shared_coeff_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.levels)),
        )

        # Residual decoder maps the private latent back into matrix space.
        # This branch deliberately stays simple: it is meant as a correction
        # term, not a second full generator competing with the action bases.
        self.private_decoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, basis_size * basis_size),
        )

        # Continuous side supervision on the quantized shared latent.
        self.side_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_side_classes),
        )

        # Discrete side supervision on the second level code. This keeps the
        # "one level ~ one interpretable factor" idea explicit.
        self.discrete_side_classifier = nn.Embedding(self.levels[1], num_side_classes)

        # Optional dataset auxiliary heads. They are disabled by default because
        # source separation is treated as a helper constraint, not the final
        # goal of the model.
        self.private_dataset_classifier = nn.Sequential(
            nn.Linear(private_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_dataset_classes),
        )

        self.shared_dataset_adversary = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_dataset_classes),
        )

    def _enforce_matrix_constraints(self, mats: torch.Tensor) -> torch.Tensor:
        """
        Project matrices into the structured space expected for distance-diff data.

        This is differentiable: it does not block gradients, it only restricts
        the solution to symmetric matrices with zero diagonal.
        """

        mats = 0.5 * (mats + mats.transpose(-1, -2))
        diag = torch.diagonal(mats, dim1=-2, dim2=-1)
        mats = mats - torch.diag_embed(diag)
        return mats

    def _load_action_basis_init(self, init_path: str) -> None:
        """Load a prebuilt `(sum(levels), H, W)` basis tensor from disk."""

        basis = torch.from_numpy(__import__("numpy").load(init_path)).float()
        expected_shape = (self.total_basis_num, self.basis_size, self.basis_size)
        if tuple(basis.shape) != expected_shape:
            raise ValueError(
                f"Action basis init shape mismatch: got {tuple(basis.shape)}, expected {expected_shape}"
            )
        with torch.no_grad():
            self.action_basis_bank.copy_(basis)

    def get_structured_basis(self) -> torch.Tensor:
        """
        Return the current shared action basis bank in model-ready form.

        We do not hard-orthogonalize it with QR every forward anymore. Instead
        we:
        1. enforce matrix structure
        2. normalize each basis
        3. use an explicit orthogonality loss
        """

        basis = self._enforce_matrix_constraints(self.action_basis_bank)
        basis_flat = basis.reshape(self.total_basis_num, -1)
        basis_flat = F.normalize(basis_flat, dim=1, eps=1e-8)
        return basis_flat.reshape(self.total_basis_num, self.basis_size, self.basis_size)

    def split_basis(self, all_basis: torch.Tensor):
        """Split the full basis bank according to `levels`."""

        basis_list = []
        start = 0
        for level in self.levels:
            basis_list.append(all_basis[start : start + level])
            start += level
        return basis_list

    def decode_indices(self, indices: torch.Tensor):
        """
        Decode the flattened LQ index back into one index per latent level.

        This must stay consistent with `LatentQuantize.codes_to_indices()`.
        """

        indices = indices.long()
        basis = self.lq._basis.to(indices.device).long()
        levels = self.lq._levels.to(indices.device).long()
        return [(indices // basis[i]) % levels[i] for i in range(len(self.levels))]

    def orthogonality_loss(self, basis: torch.Tensor) -> torch.Tensor:
        """Soft orthogonality penalty on the action bases."""

        flat = basis.reshape(self.total_basis_num, -1)
        gram = flat @ flat.T
        eye = torch.eye(self.total_basis_num, device=flat.device, dtype=flat.dtype)
        return ((gram - eye) ** 2).mean()

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

    def forward(self, x, side_labels=None, dataset_labels=None):
        """
        Forward pass for one direction-specific motion matrix batch.

        Input shape:
        - frame batch: `(B, 1, H, W)`
        - grouped sequence batch: `(B, T, 1, H, W)`

        Returns both reconstructions and auxiliary signals so training code can
        decide which losses to use.
        """

        x, sequence_shape = self._flatten_sequence_input(x)
        side_labels = self._flatten_sequence_labels(side_labels, sequence_shape)
        dataset_labels = self._flatten_sequence_labels(dataset_labels, sequence_shape)

        feats = self.initial_conv(x)
        feats = self.layer1(feats)
        feats = self.layer2(feats)
        feats = self.layer3(feats)

        # Global pooling produces one compact representation per input matrix.
        pooled = self.avg_pool(feats).flatten(1)
        shared_raw = self.shared_head(pooled)
        private_z = self.private_head(pooled)

        # LQ returns:
        # - `shared_quantized`: continuous vector with straight-through gradient
        # - `indices`: flattened discrete code
        # - `lq_loss`: commitment / quantization loss
        shared_quantized, indices, _ = self.lq(shared_raw)
        basis = self.get_structured_basis()
        basis_list = self.split_basis(basis)
        d_list = self.decode_indices(indices)

        # One scalar coefficient per level. The selected basis from each level
        # is scaled and accumulated into the shared action reconstruction.
        coeffs = self.shared_coeff_net(shared_quantized)
        shared_recon = torch.zeros(
            x.shape[0], self.basis_size, self.basis_size, device=x.device, dtype=x.dtype
        )

        for level_idx, (basis_i, d_i) in enumerate(zip(basis_list, d_list)):
            selected_basis = basis_i[d_i]
            coeff = coeffs[:, level_idx].view(x.shape[0], 1, 1)
            shared_recon = shared_recon + coeff * selected_basis

        # Private residual explains the leftover matrix content after the shared
        # action bases have accounted for the main motion pattern.
        id_nuisance_residual = self.private_decoder(private_z).reshape(
            x.shape[0], self.basis_size, self.basis_size
        )
        id_nuisance_residual = self._enforce_matrix_constraints(id_nuisance_residual)
        recon = shared_recon + self.private_residual_weight * id_nuisance_residual
        recon = self._enforce_matrix_constraints(recon).unsqueeze(1)

        zero_loss = shared_raw.new_zeros(shared_raw.shape[0])
        commitment_loss_per_sample = zero_loss
        quantization_loss_per_sample = zero_loss
        if self.training:
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
        lq_loss_per_sample = (
            self.lq.commitment_loss_weight * commitment_loss_per_sample
            + self.lq.quantization_loss_weight * quantization_loss_per_sample
        )
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
            # Two complementary side losses:
            # 1. continuous loss on the shared latent
            # 2. discrete loss on one selected level code
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
            # Optional helper objective:
            # - private branch may keep dataset-specific information
            # - shared branch is weakly discouraged from encoding it
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
        residual_l1_per_sample = id_nuisance_residual.abs().mean(dim=(1, 2))
        residual_l1 = residual_l1_per_sample.mean()

        reconstructed = self._reshape_sequence_tensor(recon, sequence_shape)
        action_reconstruction = self._reshape_sequence_tensor(
            self._enforce_matrix_constraints(shared_recon).unsqueeze(1),
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

        return {
            "reconstructed": reconstructed,
            # `action_reconstruction` is the interpretable shared-motion part.
            "action_reconstruction": action_reconstruction,
            # Kept for backward compatibility with earlier code that used the
            # older `shared_reconstruction` naming.
            "shared_reconstruction": action_reconstruction,
            "id_nuisance_residual": private_residual,
            # Kept for backward compatibility with earlier code that used the
            # older `private_residual` naming.
            "private_residual": private_residual,
            "shared_quantized": shared_quantized,
            "private_z": private_z,
            "indices": indices,
            "decoded_indices": decoded_indices,
            "action_basis": basis,
            # Backward-compatible alias.
            "basis": basis,
            "lq_loss": lq_loss,
            "lq_loss_per_sample": lq_loss_per_sample,
            "orth_loss": orth_loss,
            "residual_l1": residual_l1,
            "residual_l1_per_sample": residual_l1_per_sample,
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
