from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def enforce_matrix_constraints(mats: torch.Tensor) -> torch.Tensor:
    """Project matrices to symmetric zero-diagonal form."""

    mats = 0.5 * (mats + mats.transpose(-1, -2))
    diag = torch.diagonal(mats, dim1=-2, dim2=-1)
    return mats - torch.diag_embed(diag)


def load_action_basis_init(
    action_basis_bank: torch.nn.Parameter,
    *,
    init_path: str,
    total_basis_num: int,
    basis_size: int,
) -> None:
    """Load a prebuilt `(sum(levels), H, W)` basis tensor from disk."""

    basis = torch.from_numpy(np.load(init_path)).float()
    expected_shape = (total_basis_num, basis_size, basis_size)
    if tuple(basis.shape) != expected_shape:
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
    """Load a prebuilt `(side_basis_count, H, W)` side-basis tensor from disk."""

    basis = torch.from_numpy(np.load(init_path)).float()
    expected_shape = (side_basis_count, basis_size, basis_size)
    if tuple(basis.shape) != expected_shape:
        raise ValueError(
            f"Side basis init shape mismatch: got {tuple(basis.shape)}, expected {expected_shape}"
        )
    with torch.no_grad():
        side_basis_bank.copy_(basis)


def qr_orthogonalize_rows(basis_flat: torch.Tensor) -> torch.Tensor:
    """Orthonormalize row vectors with a differentiable QR projection."""

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
) -> torch.Tensor:
    """
    Return the current shared action basis bank in model-ready form.

    Supported modes:
    - `normalize`: keep the current soft-constraint behavior
    - `level_qr`: enforce strict orthonormality within each level via QR
    - `global_qr`: enforce strict orthonormality across all bases via one QR
    """

    basis = enforce_matrix_constraints(action_basis_bank)
    basis_flat = basis.reshape(total_basis_num, -1)

    if basis_orthogonalization == "normalize":
        basis_flat = F.normalize(basis_flat, dim=1, eps=1e-8)
        return basis_flat.reshape(total_basis_num, basis_size, basis_size)

    if basis_orthogonalization == "global_qr":
        basis_flat = qr_orthogonalize_rows(basis_flat)
        return basis_flat.reshape(total_basis_num, basis_size, basis_size)

    if basis_orthogonalization == "level_qr":
        orth_basis = []
        start = 0
        for level in levels:
            level_flat = basis_flat[start : start + level]
            orth_basis.append(qr_orthogonalize_rows(level_flat))
            start += level
        basis_flat = torch.cat(orth_basis, dim=0)
        return basis_flat.reshape(total_basis_num, basis_size, basis_size)

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return jointly-orthogonalized shared / side basis banks when configured.

    `joint_global_qr` projects all shared and side bases into one common
    orthonormal basis family so no pair across the two banks can collapse.
    """

    shared_basis = enforce_matrix_constraints(action_basis_bank)
    side_basis = enforce_matrix_constraints(side_basis_bank)

    if basis_orthogonalization != "joint_global_qr" or side_basis_count <= 0:
        shared_structured = get_structured_basis(
            shared_basis,
            levels=levels,
            total_basis_num=total_basis_num,
            basis_size=basis_size,
            basis_orthogonalization=basis_orthogonalization,
        )
        side_flat = F.normalize(side_basis.reshape(side_basis_count, -1), dim=1, eps=1e-8)
        return shared_structured, side_flat.reshape(side_basis_count, basis_size, basis_size)

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
        shared_flat.reshape(total_basis_num, basis_size, basis_size),
        side_flat.reshape(side_basis_count, basis_size, basis_size),
    )


def orthogonality_loss(basis: torch.Tensor, total_basis_num: int) -> torch.Tensor:
    """Soft orthogonality penalty on the action bases."""

    flat = basis.reshape(total_basis_num, -1)
    gram = flat @ flat.T
    eye = torch.eye(total_basis_num, device=flat.device, dtype=flat.dtype)
    return ((gram - eye) ** 2).mean()


def basis_l1_loss(basis: torch.Tensor) -> torch.Tensor:
    """Sparse structured-basis penalty applied after QR / normalization."""

    return basis.abs().mean()


def split_basis(all_basis: torch.Tensor, levels: tuple[int, ...]):
    """Split the full basis bank according to `levels`."""

    basis_list = []
    start = 0
    for level in levels:
        basis_list.append(all_basis[start : start + level])
        start += level
    return basis_list
