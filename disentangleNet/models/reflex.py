from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ReflexLayout:
    reflex_count: int
    pair_count: int

    @property
    def paired_seed_count(self) -> int:
        if self.pair_count < 0:
            raise ValueError(f"pair_count must be >= 0, got {self.pair_count}")
        return int(self.pair_count)

    @property
    def total_output_count(self) -> int:
        return self.reflex_count + self.paired_seed_count * 2

    @property
    def total_source_count(self) -> int:
        return self.reflex_count + self.paired_seed_count


def infer_reflex_layout(*, self_count: int, pair_count: int) -> ReflexLayout:
    if self_count < 0:
        raise ValueError(f"self_count must be >= 0, got {self_count}")
    if pair_count < 0:
        raise ValueError(f"pair_count must be >= 0, got {pair_count}")
    return ReflexLayout(reflex_count=int(self_count), pair_count=int(pair_count))


class ReflexBasisComposer(nn.Module):
    """Recovered reflex/pair basis composition helper."""

    def __init__(self, *, self_count: int, pair_count: int, mirror_perm: torch.Tensor | None) -> None:
        super().__init__()
        self.layout = infer_reflex_layout(self_count=self_count, pair_count=pair_count)
        if mirror_perm is not None:
            self.register_buffer("mirror_perm", mirror_perm.clone().long())
        else:
            self.mirror_perm = None

    def _mirror(self, bases: torch.Tensor) -> torch.Tensor:
        if self.mirror_perm is None:
            return bases
        mirror_perm = self.mirror_perm.to(device=bases.device)
        return bases[:, mirror_perm][:, :, mirror_perm]

    def forward(self, source_bases: torch.Tensor) -> torch.Tensor:
        if source_bases.shape[0] != self.layout.total_source_count:
            raise ValueError(
                "source basis count mismatch: "
                f"got {source_bases.shape[0]}, expected {self.layout.total_source_count}"
            )
        reflex_src = source_bases[: self.layout.reflex_count]
        seed_src = source_bases[self.layout.reflex_count :]

        if self.layout.reflex_count > 0:
            if self.mirror_perm is not None:
                reflex_out = 0.5 * (reflex_src + self._mirror(reflex_src))
            else:
                reflex_out = reflex_src
        else:
            reflex_out = source_bases.new_zeros((0, *source_bases.shape[1:]))

        if self.layout.paired_seed_count > 0:
            if self.mirror_perm is not None:
                paired_mirror = self._mirror(seed_src)
            else:
                paired_mirror = seed_src
            paired_out = torch.stack([seed_src, paired_mirror], dim=1).reshape(
                self.layout.paired_seed_count * 2,
                *seed_src.shape[1:],
            )
        else:
            paired_out = source_bases.new_zeros((0, *source_bases.shape[1:]))

        return torch.cat([reflex_out, paired_out], dim=0)

    def diagnostics(self, basis: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = basis.reshape(basis.shape[0], -1)
        gram = flat @ flat.T
        offdiag = gram - torch.diag_embed(torch.diagonal(gram))
        return {
            "max_diag_abs": torch.diagonal(basis, dim1=-2, dim2=-1).abs().max(),
            "max_symmetry_error": (basis - basis.transpose(-1, -2)).abs().max(),
            "max_offdiag_gram_abs": offdiag.abs().max(),
        }


__all__ = [
    "ReflexBasisComposer",
    "ReflexLayout",
    "infer_reflex_layout",
]
