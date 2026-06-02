from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from disentangleNet.models.reflex import ReflexBasisComposer, infer_reflex_layout

from .correction import project_symmetric_zero_diagonal, scale_basis_abs_max
from .synthesis import synthesize_lowrank_basis


@dataclass(frozen=True)
class BasisDiagnostics:
    max_diag_abs: torch.Tensor
    max_symmetry_error: torch.Tensor
    max_offdiag_gram_abs: torch.Tensor


class LowRankBasisRuntime(nn.Module):
    """
    Low-rank basis runtime used by the reconstructed modular low-rank path.

    The recovered contract is:
    - synthesize per-source basis as sum_k coeff_k * u_k u_k^T
    - optionally expand reflex/paired mirrored outputs
    - enforce symmetric zero-diagonal structure
    - optionally scale each basis to basis_abs_max
    """

    def __init__(
        self,
        *,
        levels: tuple[int, ...],
        basis_size: int,
        lowrank_level_ranks: tuple[int, ...],
        init_path: str | None,
        mirror_perm: torch.Tensor | None,
        basis_abs_max: float = 0.05,
        reflex_self_count: int = 0,
        reflex_pair_count: int = 0,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(v) for v in levels)
        self.basis_size = int(basis_size)
        self.total_basis_num = sum(self.levels)
        self.level_ranks = tuple(int(v) for v in lowrank_level_ranks)
        if len(self.levels) != len(self.level_ranks):
            raise ValueError(
                "levels and lowrank_level_ranks length mismatch: "
                f"{self.levels} vs {self.level_ranks}"
            )
        self.basis_abs_max = float(basis_abs_max)
        self.layout = infer_reflex_layout(
            self_count=int(reflex_self_count),
            pair_count=int(reflex_pair_count),
        )
        if self.layout.total_output_count not in {0, self.total_basis_num}:
            raise ValueError(
                "reflex layout/output mismatch: "
                f"levels total={self.total_basis_num} layout={self.layout.total_output_count}"
            )
        self.source_counts = self._build_source_counts()
        self.source_slices = self._build_slices(self.source_counts)
        self.level_slices = self._build_slices(self.levels)
        total_source_count = sum(self.source_counts)
        max_rank = max(self.level_ranks, default=0)
        self.latents = nn.Parameter(
            torch.randn(total_source_count, max_rank, self.basis_size) * 0.02
        )
        self.coefficients = nn.Parameter(
            torch.randn(total_source_count, max_rank) * 0.02
        )
        self.register_buffer(
            "rank_mask",
            self._build_rank_mask(total_source_count, max_rank),
        )
        if mirror_perm is not None:
            mirror_tensor = torch.as_tensor(mirror_perm, dtype=torch.long).clone()
            self.register_buffer("mirror_perm", mirror_tensor)
        else:
            self.mirror_perm = None
        self.composer = (
            ReflexBasisComposer(
                self_count=self.layout.reflex_count,
                pair_count=self.layout.pair_count,
                mirror_perm=self.mirror_perm,
            )
            if self.layout.total_output_count > 0
            else None
        )
        if init_path is not None:
            self._load_init(Path(init_path))

    def _build_source_counts(self) -> tuple[int, ...]:
        if self.layout.total_output_count == 0:
            return self.levels
        if len(self.levels) != 2:
            raise ValueError(
                "reflex low-rank runtime expects two levels, got "
                f"{self.levels}"
            )
        return (self.layout.reflex_count, self.layout.pair_count)

    @staticmethod
    def _build_slices(counts: tuple[int, ...]) -> tuple[slice, ...]:
        slices: list[slice] = []
        start = 0
        for count in counts:
            slices.append(slice(start, start + int(count)))
            start += int(count)
        return tuple(slices)

    def _build_rank_mask(self, total_source_count: int, max_rank: int) -> torch.Tensor:
        mask = torch.zeros(total_source_count, max_rank, dtype=torch.float32)
        for source_slice, rank in zip(self.source_slices, self.level_ranks):
            if rank <= 0:
                raise ValueError(f"low-rank rank must be positive, got {rank}")
            mask[source_slice, :rank] = 1.0
        return mask

    def _effective_latents(self) -> torch.Tensor:
        return self.latents * self.rank_mask.unsqueeze(-1).to(self.latents.device)

    def _effective_coefficients(self) -> torch.Tensor:
        return self.coefficients * self.rank_mask.to(self.coefficients.device)

    def _compose_source_basis(self) -> torch.Tensor:
        source_basis = synthesize_lowrank_basis(
            self._effective_latents(),
            self._effective_coefficients(),
        )
        if self.composer is not None:
            source_basis = self.composer(source_basis)
        return source_basis

    def _load_init(self, init_path: Path) -> None:
        dense_basis = torch.from_numpy(np.load(str(init_path))).float()
        expected = (self.total_basis_num, self.basis_size, self.basis_size)
        if tuple(dense_basis.shape) != expected:
            raise ValueError(
                f"basis init shape mismatch: got {tuple(dense_basis.shape)}, expected {expected}"
            )
        if self.layout.total_output_count > 0:
            source_parts = []
            if self.layout.reflex_count > 0:
                source_parts.append(dense_basis[: self.layout.reflex_count])
            if self.layout.pair_count > 0:
                pair_basis = dense_basis[self.layout.reflex_count :]
                source_parts.append(pair_basis[::2])
            dense_basis = torch.cat(source_parts, dim=0) if source_parts else dense_basis[:0]
        if dense_basis.shape[0] != self.latents.shape[0]:
            raise ValueError(
                "source basis init count mismatch: "
                f"got {dense_basis.shape[0]}, expected {self.latents.shape[0]}"
            )
        sym_basis = project_symmetric_zero_diagonal(dense_basis)
        rank_mask = self.rank_mask
        with torch.no_grad():
            self.latents.zero_()
            self.coefficients.zero_()
            for source_index in range(sym_basis.shape[0]):
                rank = int(rank_mask[source_index].sum().item())
                eigvals, eigvecs = torch.linalg.eigh(sym_basis[source_index])
                order = torch.argsort(eigvals.abs(), descending=True)[:rank]
                self.coefficients[source_index, :rank] = eigvals[order]
                self.latents[source_index, :rank] = eigvecs[:, order].T

    def get_structured_basis(self) -> torch.Tensor:
        basis = self._compose_source_basis()
        basis = project_symmetric_zero_diagonal(basis)
        return scale_basis_abs_max(basis, target_abs_max=self.basis_abs_max)

    def forward(self) -> torch.Tensor:
        return self.get_structured_basis()

    def frequency_loss(self) -> torch.Tensor:
        basis = self.get_structured_basis()
        row_diff = basis[..., 1:, :] - basis[..., :-1, :]
        col_diff = basis[..., :, 1:] - basis[..., :, :-1]
        return 0.5 * (row_diff.square().mean() + col_diff.square().mean())

    def orthogonality_loss(self) -> torch.Tensor:
        basis = self.get_structured_basis()
        flat = F.normalize(basis.reshape(basis.shape[0], -1), dim=1, eps=1e-8)
        gram = flat @ flat.T
        eye = torch.eye(flat.shape[0], device=flat.device, dtype=flat.dtype)
        return (gram - eye).square().mean()

    def diagnostics(self) -> BasisDiagnostics:
        basis = self.get_structured_basis()
        flat = F.normalize(basis.reshape(basis.shape[0], -1), dim=1, eps=1e-8)
        gram = flat @ flat.T
        offdiag = gram - torch.diag_embed(torch.diagonal(gram))
        return BasisDiagnostics(
            max_diag_abs=torch.diagonal(basis, dim1=-2, dim2=-1).abs().max(),
            max_symmetry_error=(basis - basis.transpose(-1, -2)).abs().max(),
            max_offdiag_gram_abs=offdiag.abs().max(),
        )


__all__ = [
    "BasisDiagnostics",
    "LowRankBasisRuntime",
]
