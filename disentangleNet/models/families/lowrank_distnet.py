from __future__ import annotations

import torch

from ..basis import collect_runtime_diagnostics
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
        outputs.update(
            collect_runtime_diagnostics(
                self.shared_basis_runtime,
                orth_key="lowrank_orth_loss",
                diag_prefix="lowrank",
                extra_freq_key="lowrank_freq_loss",
            )
        )
        return outputs
