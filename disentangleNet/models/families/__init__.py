try:
    from .distnet import DistNet
except Exception as exc:  # pragma: no cover - recovery compatibility path
    # TODO(recovery): restore the legacy v31 family dependencies
    # (`basis_ops.py`, encoder helpers, and v31 forward helpers). The current
    # PhaseAB recovery path does not construct `DistNet`, so keep this import
    # failure isolated instead of breaking modular model builds.
    DistNet = None
    _DISTNET_IMPORT_ERROR = exc
from .v6_distnet import V6DistNet
from .lowrank_distnet import LowRankDistNet
from .lowrank_reflex_distnet import LowRankReflexDistNet

__all__ = [
    "DistNet",
    "LowRankDistNet",
    "LowRankReflexDistNet",
    "V6DistNet",
]
