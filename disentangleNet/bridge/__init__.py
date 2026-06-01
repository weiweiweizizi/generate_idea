"""
Bridge layer between disentangleNet exports and matrix_vis.

Reconstructed from:
- scripts/matrix_vis/docs/disentanglenet_matrix_vis_contract.md  (Section 8: bridge helper API)
"""
from .matrix_vis import (
    PatientBundleBridge,
    load_patient_bundle_bridge,
    restore_physical_observation_scale,
)

__all__ = [
    "PatientBundleBridge",
    "load_patient_bundle_bridge",
    "restore_physical_observation_scale",
]
