from __future__ import annotations

import torch.nn as nn

from .BasicBlock import BasicBlock


def build_motion_encoder(hidden_dim: int, pool_size: int):
    """Construct the shallow CNN encoder used by DistNet."""

    initial_conv = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
    )
    pre_layer1_block = BasicBlock(8, 8)
    layer1 = BasicBlock(
        8,
        16,
        stride=2,
        downsample=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
        ),
    )
    pre_layer2_block = BasicBlock(16, 16)
    layer2 = BasicBlock(
        16,
        hidden_dim,
        stride=2,
        downsample=nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
        ),
    )
    layer3 = BasicBlock(hidden_dim, hidden_dim)
    avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
    return (
        initial_conv,
        pre_layer1_block,
        layer1,
        pre_layer2_block,
        layer2,
        layer3,
        avg_pool,
    )


def build_branch_adapter(hidden_dim: int) -> nn.Sequential:
    """Construct a lightweight feature adapter for an early factorized branch."""

    return nn.Sequential(
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
    )


def build_branch_pool(pool_size: int) -> nn.AdaptiveAvgPool2d:
    """Construct branch-local spatial pooling used before branch heads."""

    return nn.AdaptiveAvgPool2d((pool_size, pool_size))
