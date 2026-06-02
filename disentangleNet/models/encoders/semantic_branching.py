from __future__ import annotations

import torch.nn as nn


class SemanticBranchingEncoder(nn.Module):
    """
    TODO(recovery): recover the historical semantic branching encoder used by
    legacy families. The current PhaseAB path does not instantiate it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        _ = args, kwargs

    def forward(self, x):
        return x
