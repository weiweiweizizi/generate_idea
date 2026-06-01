from __future__ import annotations

import torch


def fold_mouth_chunk_features(
    x: torch.Tensor,
    *,
    side_feature_mode: str,
    basis_size: int,
) -> torch.Tensor:
    if side_feature_mode == "none":
        return x.new_zeros((x.shape[0], 0))
    if basis_size != 119:
        raise ValueError("folded_mouth_chunks expects the 119x119 mouth crop")

    chunk_slices = (
        slice(0, 22),
        slice(22, 45),
        slice(45, 82),
        slice(82, 119),
    )
    matrix = x[:, 0] if x.ndim == 4 else x
    abs_matrix = matrix.abs()

    row_means = [abs_matrix[:, current, :].mean(dim=(1, 2)) for current in chunk_slices]
    col_means = [abs_matrix[:, :, current].mean(dim=(1, 2)) for current in chunk_slices]
    block_means = [
        abs_matrix[:, chunk_slices[0], chunk_slices[0]].mean(dim=(1, 2)),
        abs_matrix[:, chunk_slices[1], chunk_slices[1]].mean(dim=(1, 2)),
        abs_matrix[:, chunk_slices[2], chunk_slices[2]].mean(dim=(1, 2)),
        abs_matrix[:, chunk_slices[3], chunk_slices[3]].mean(dim=(1, 2)),
    ]
    around_left = 0.5 * (row_means[0] + col_means[0])
    around_right = 0.5 * (row_means[1] + col_means[1])
    mouth_left = 0.5 * (row_means[2] + col_means[2])
    mouth_right = 0.5 * (row_means[3] + col_means[3])
    around_contrast = around_left - around_right
    mouth_contrast = mouth_left - mouth_right
    around_sum = around_left + around_right
    mouth_sum = mouth_left + mouth_right
    within_around_contrast = block_means[0] - block_means[1]
    within_mouth_contrast = block_means[2] - block_means[3]
    return torch.stack(
        [
            around_contrast,
            mouth_contrast,
            around_sum,
            mouth_sum,
            within_around_contrast,
            within_mouth_contrast,
        ],
        dim=1,
    )


__all__ = ["fold_mouth_chunk_features"]
