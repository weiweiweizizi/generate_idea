from __future__ import annotations

import torch.nn as nn


def build_shared_head(pooled_dim: int, hidden_dim: int, shared_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(pooled_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, shared_dim),
    )


def build_private_head(pooled_dim: int, hidden_dim: int, private_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(pooled_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, private_dim),
    )


def build_free_head(pooled_dim: int, hidden_dim: int, free_z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(pooled_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, free_z_dim),
    )


def build_side_head(pooled_dim: int, hidden_dim: int, side_z_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(pooled_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, side_z_dim),
    )


def build_shared_coeff_net(shared_dim: int, hidden_dim: int, num_levels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(shared_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, num_levels),
    )


def build_shared_coeff_heads(
    *,
    shared_dim: int,
    hidden_dim: int,
    levels: tuple[int, ...],
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            for _ in levels
        ]
    )


def build_shared_basis_heads(
    *,
    shared_dim: int,
    hidden_dim: int,
    levels: tuple[int, ...],
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, level),
            )
            for level in levels
        ]
    )


def build_private_decoder(
    *,
    private_dim: int,
    private_decoder_hidden_dim: int,
    basis_size: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(private_dim, private_decoder_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(private_decoder_hidden_dim, basis_size * basis_size),
    )


def build_side_classifier(side_dim: int, num_side_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(side_dim, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, num_side_classes),
    )


def build_side_semantic_coeff_head(side_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(side_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 1),
    )


def build_side_semantic_basis_head(
    side_dim: int,
    hidden_dim: int,
    side_basis_count: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(side_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, side_basis_count),
    )


def build_group_side_classifier(
    side_basis_count: int,
    num_side_classes: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(side_basis_count, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, num_side_classes),
    )


def build_group_severity_classifier(
    severity_input_dim: int,
    num_severity_classes: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(severity_input_dim, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, num_severity_classes),
    )


def build_discrete_side_classifier(level_size: int, num_side_classes: int) -> nn.Embedding:
    return nn.Embedding(level_size, num_side_classes)


def build_private_dataset_classifier(
    private_dim: int,
    num_dataset_classes: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(private_dim, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, num_dataset_classes),
    )


def build_shared_dataset_adversary(
    shared_dim: int,
    num_dataset_classes: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(shared_dim, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, num_dataset_classes),
    )
