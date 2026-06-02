from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis_pipeline.correction import (
    project_basis_abs_max_,
    project_symmetric_zero_diagonal,
    scale_basis_abs_max,
)
from .reflex import ReflexBasisComposer, infer_reflex_layout


def enforce_matrix_constraints(mats: torch.Tensor) -> torch.Tensor:
    return project_symmetric_zero_diagonal(mats)


def load_action_basis_init(
    action_basis_bank: torch.nn.Parameter,
    *,
    init_path: str,
    total_basis_num: int,
    basis_size: int,
) -> None:
    basis = torch.from_numpy(np.load(init_path)).float()
    expected_shape = (total_basis_num, basis_size, basis_size)
    if tuple(basis.shape) == (total_basis_num + 3, basis_size, basis_size):
        basis = basis[:total_basis_num]
    elif tuple(basis.shape) != expected_shape:
        raise ValueError(
            f"Action basis init shape mismatch: got {tuple(basis.shape)}, expected {expected_shape}"
        )
    with torch.no_grad():
        action_basis_bank.copy_(basis)


def load_side_basis_init(
    side_basis_bank: torch.nn.Parameter,
    *,
    init_path: str,
    side_basis_count: int,
    basis_size: int,
) -> None:
    basis = torch.from_numpy(np.load(init_path)).float()
    expected_shape = (side_basis_count, basis_size, basis_size)
    if tuple(basis.shape) == (side_basis_count + 8, basis_size, basis_size):
        basis = basis[-side_basis_count:]
    elif tuple(basis.shape) != expected_shape:
        raise ValueError(
            f"Side basis init shape mismatch: got {tuple(basis.shape)}, expected {expected_shape}"
        )
    with torch.no_grad():
        side_basis_bank.copy_(basis)


def qr_orthogonalize_rows(basis_flat: torch.Tensor) -> torch.Tensor:
    weight = basis_flat.T + 1e-8
    q, _ = torch.linalg.qr(weight, mode="reduced")
    return q.T


def get_structured_basis(
    action_basis_bank: torch.Tensor,
    *,
    levels: tuple[int, ...],
    total_basis_num: int,
    basis_size: int,
    basis_orthogonalization: str,
    target_abs_max: float = 0.05,
) -> torch.Tensor:
    basis = enforce_matrix_constraints(action_basis_bank)
    basis_flat = basis.reshape(total_basis_num, -1)

    if basis_orthogonalization in {"normalize", "none"}:
        basis_flat = F.normalize(basis_flat, dim=1, eps=1e-8)
        basis = basis_flat.reshape(total_basis_num, basis_size, basis_size)
        return scale_basis_abs_max(basis, target_abs_max=target_abs_max)

    if basis_orthogonalization == "global_qr":
        basis_flat = qr_orthogonalize_rows(basis_flat)
        basis = basis_flat.reshape(total_basis_num, basis_size, basis_size)
        return scale_basis_abs_max(basis, target_abs_max=target_abs_max)

    if basis_orthogonalization == "level_qr":
        orth_basis = []
        start = 0
        for level in levels:
            level_flat = basis_flat[start : start + level]
            orth_basis.append(qr_orthogonalize_rows(level_flat))
            start += level
        basis_flat = torch.cat(orth_basis, dim=0)
        basis = basis_flat.reshape(total_basis_num, basis_size, basis_size)
        return scale_basis_abs_max(basis, target_abs_max=target_abs_max)

    raise ValueError(f"Unsupported basis_orthogonalization: {basis_orthogonalization}")


def get_joint_structured_basis(
    action_basis_bank: torch.Tensor,
    side_basis_bank: torch.Tensor,
    *,
    levels: tuple[int, ...],
    total_basis_num: int,
    side_basis_count: int,
    basis_size: int,
    basis_orthogonalization: str,
    target_abs_max: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_basis = enforce_matrix_constraints(action_basis_bank)
    side_basis = enforce_matrix_constraints(side_basis_bank)

    if basis_orthogonalization != "joint_global_qr" or side_basis_count <= 0:
        shared_structured = get_structured_basis(
            shared_basis,
            levels=levels,
            total_basis_num=total_basis_num,
            basis_size=basis_size,
            basis_orthogonalization=basis_orthogonalization,
            target_abs_max=target_abs_max,
        )
        side_flat = F.normalize(side_basis.reshape(side_basis_count, -1), dim=1, eps=1e-8)
        side_structured = side_flat.reshape(side_basis_count, basis_size, basis_size)
        return shared_structured, scale_basis_abs_max(side_structured, target_abs_max=target_abs_max)

    joint_flat = torch.cat(
        [
            shared_basis.reshape(total_basis_num, -1),
            side_basis.reshape(side_basis_count, -1),
        ],
        dim=0,
    )
    joint_flat = qr_orthogonalize_rows(joint_flat)
    shared_flat = joint_flat[:total_basis_num]
    side_flat = joint_flat[total_basis_num:]
    return (
        scale_basis_abs_max(
            shared_flat.reshape(total_basis_num, basis_size, basis_size),
            target_abs_max=target_abs_max,
        ),
        scale_basis_abs_max(
            side_flat.reshape(side_basis_count, basis_size, basis_size),
            target_abs_max=target_abs_max,
        ),
    )


def orthogonality_loss(basis: torch.Tensor, total_basis_num: int) -> torch.Tensor:
    flat = basis.reshape(total_basis_num, -1)
    gram = flat @ flat.T
    eye = torch.eye(total_basis_num, device=flat.device, dtype=flat.dtype)
    return ((gram - eye) ** 2).mean()


def basis_l1_loss(basis: torch.Tensor) -> torch.Tensor:
    return basis.abs().mean()


def basis_frequency_loss(basis: torch.Tensor) -> torch.Tensor:
    row_diff = basis[..., 1:, :] - basis[..., :-1, :]
    col_diff = basis[..., :, 1:] - basis[..., :, :-1]
    return 0.5 * (row_diff.square().mean() + col_diff.square().mean())


def split_basis(all_basis: torch.Tensor, levels: tuple[int, ...]):
    basis_list = []
    start = 0
    for level in levels:
        basis_list.append(all_basis[start : start + level])
        start += level
    return basis_list


def collect_runtime_diagnostics(
    basis_runtime,
    *,
    orth_key: str,
    diag_prefix: str,
    include_v9_freq: bool = True,
    extra_freq_key: str | None = None,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    freq_loss = basis_runtime.frequency_loss()
    if include_v9_freq:
        outputs["v9_freq_loss"] = freq_loss
    if extra_freq_key is not None:
        outputs[extra_freq_key] = freq_loss
    outputs[orth_key] = basis_runtime.orthogonality_loss()
    diagnostics = basis_runtime.diagnostics()
    if isinstance(diagnostics, dict):
        outputs[f"{diag_prefix}_max_diag_abs"] = diagnostics["max_diag_abs"]
        outputs[f"{diag_prefix}_max_symmetry_error"] = diagnostics["max_symmetry_error"]
        outputs[f"{diag_prefix}_max_offdiag_gram_abs"] = diagnostics["max_offdiag_gram_abs"]
    else:
        outputs[f"{diag_prefix}_max_diag_abs"] = diagnostics.max_diag_abs
        outputs[f"{diag_prefix}_max_symmetry_error"] = diagnostics.max_symmetry_error
        outputs[f"{diag_prefix}_max_offdiag_gram_abs"] = diagnostics.max_offdiag_gram_abs
    return outputs


class ReflexBasisBank(nn.Module):
    """
    Recovered minimal reflex basis runtime for the current PhaseAB path.

    TODO(recovery): replace this dense-parameter version with the historical
    lowrank provider/runtime once the dedicated basis pipeline modules are
    restored. This class preserves the current constructor and diagnostics API.
    """

    def __init__(
        self,
        *,
        levels: tuple[int, ...],
        basis_size: int,
        init_path: str | None,
        mirror_perm: torch.Tensor | None,
        basis_abs_max: float = 0.05,
    ) -> None:
        super().__init__()
        self.levels = tuple(int(v) for v in levels)
        self.basis_size = int(basis_size)
        self.layout = infer_reflex_layout(
            self_count=int(self.levels[0]),
            pair_count=int(self.levels[1] // 2),
        )
        self.composer = ReflexBasisComposer(
            self_count=int(self.levels[0]),
            pair_count=int(self.levels[1] // 2),
            mirror_perm=mirror_perm,
        )
        source_count = self.layout.total_source_count
        self.source_basis_bank = nn.Parameter(
            torch.randn(source_count, self.basis_size, self.basis_size) * 0.02
        )
        self.basis_abs_max = float(basis_abs_max)
        if init_path is not None:
            dense_basis = torch.from_numpy(np.load(init_path)).float()
            expected = (sum(self.levels), self.basis_size, self.basis_size)
            if tuple(dense_basis.shape) != expected:
                raise ValueError(
                    f"Reflex basis init shape mismatch: got {tuple(dense_basis.shape)}, expected {expected}"
                )
            reflex_count = self.layout.reflex_count
            pair_count = self.layout.paired_seed_count
            source_parts = []
            if reflex_count > 0:
                source_parts.append(dense_basis[:reflex_count])
            if pair_count > 0:
                source_parts.append(dense_basis[reflex_count : reflex_count + pair_count])
            source_basis = torch.cat(source_parts, dim=0) if source_parts else dense_basis.new_zeros((0, self.basis_size, self.basis_size))
            with torch.no_grad():
                self.source_basis_bank.copy_(source_basis)

    def get_structured_basis(self) -> torch.Tensor:
        dense_basis = self.composer(self.source_basis_bank)
        dense_basis = enforce_matrix_constraints(dense_basis)
        return scale_basis_abs_max(dense_basis, target_abs_max=self.basis_abs_max)

    def forward(self) -> torch.Tensor:
        return self.get_structured_basis()

    def frequency_loss(self) -> torch.Tensor:
        return basis_frequency_loss(self.get_structured_basis())

    def orthogonality_loss(self) -> torch.Tensor:
        basis = self.get_structured_basis()
        return orthogonality_loss(basis, basis.shape[0])

    def diagnostics(self):
        basis = self.get_structured_basis()
        flat = basis.reshape(basis.shape[0], -1)
        gram = flat @ flat.T
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        diag = torch.diagonal(basis, dim1=-2, dim2=-1).abs().max()
        symmetry = (basis - basis.transpose(-1, -2)).abs().max()
        offdiag = (gram - eye).masked_fill(eye.bool(), 0.0).abs().max()
        return {
            "max_diag_abs": diag,
            "max_symmetry_error": symmetry,
            "max_offdiag_gram_abs": offdiag,
        }
