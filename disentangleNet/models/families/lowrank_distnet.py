from __future__ import annotations

import torch

from .v6_distnet import V6DistNet


class LowRankDistNet(V6DistNet):
    """Recovered lowrank family fragment built on top of `V6DistNet`."""

    def __init__(
        self,
        *args,
        lowrank_level_ranks: tuple[int, ...] = (3, 5),
        action_basis_init_path: str | None = None,
        action_side_detach: bool = False,
        basis_provider=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, action_basis_init_path=None, basis_provider=basis_provider, **kwargs)
        self.action_side_detach = bool(action_side_detach)
        self.lowrank_basis_bank = self.shared_basis_runtime
        self.shared_basis_runtime = self.lowrank_basis_bank
        self.lowrank_level_ranks = tuple(int(v) for v in lowrank_level_ranks)

    def get_structured_basis(self) -> torch.Tensor:
        return self.shared_basis_runtime.get_structured_basis()

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        diagnostics = self.shared_basis_runtime.diagnostics()
        outputs["v9_freq_loss"] = self.shared_basis_runtime.frequency_loss()
        outputs["lowrank_freq_loss"] = outputs["v9_freq_loss"]
        outputs["lowrank_orth_loss"] = self.shared_basis_runtime.orthogonality_loss()
        if isinstance(diagnostics, dict):
            outputs["lowrank_max_diag_abs"] = diagnostics["max_diag_abs"]
            outputs["lowrank_max_symmetry_error"] = diagnostics["max_symmetry_error"]
            outputs["lowrank_max_offdiag_gram_abs"] = diagnostics["max_offdiag_gram_abs"]
        else:
            outputs["lowrank_max_diag_abs"] = diagnostics.max_diag_abs
            outputs["lowrank_max_symmetry_error"] = diagnostics.max_symmetry_error
            outputs["lowrank_max_offdiag_gram_abs"] = diagnostics.max_offdiag_gram_abs
        return outputs
