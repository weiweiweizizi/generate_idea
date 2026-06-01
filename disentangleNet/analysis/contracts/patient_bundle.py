"""
Patient bundle contract for disentangleNet bridge.

Reconstructed from:
- scripts/matrix_vis/docs/disentanglenet_matrix_vis_contract.md  (Section 3.2: bundle_contract, Section 5: patient bundle export)
"""
from __future__ import annotations

from typing import Any


def build_patient_bundle_contract(
    *,
    framework: str,
    mode: str,
    region: str,
    matrix_size: int,
    signed_normalize: str = "per_sample",
) -> dict[str, Any]:
    """
    Build the ``bundle_contract`` dict that must be written into every
    ``patient_*_summary.json``.
    """
    return {
        "framework": framework,
        "mode": mode,
        "region": region,
        "matrix_size": matrix_size,
        "signed_normalize": signed_normalize,
        "value_semantics": "mean_distance_delta",
        "observation_matrix_space": "normalized_input_space",
        "observation_scale_semantics": "per_window_restore_scale",
        "composition_rule": "shared_coeff_weighted_basis_plus_decoded_side_reconstruction_plus_private_residual",
        "includes_private_residual": True,
    }


def build_patient_bundle_summary(
    *,
    framework: str,
    checkpoint_path: str,
    dataset_name: str,
    subject: str,
    mode: str,
    region: str,
    matrix_size: int,
    num_valid_windows: int,
    point_layout: str,
    point_layout_region_names: list[str],
    bundle_contract: dict[str, Any],
    bundle_path: str = "",
    side_predictions_csv: str = "",
    model_family: str = "unknown",
) -> dict[str, Any]:
    """
    Build the full ``patient_*_summary.json`` payload.

    Field order matches the actual output at:
    ``outputs/disentangleNet_frame/.../patients/TT_851519/patient_bundle/patient_851519_summary.json``
    """
    return {
        "framework": framework,
        "checkpoint_path": str(checkpoint_path),
        "dataset_name": dataset_name,
        "subject": subject,
        "mode": mode,
        "region": region,
        "point_layout": point_layout,
        "point_layout_region_names": point_layout_region_names,
        "matrix_size": matrix_size,
        "num_valid_windows": num_valid_windows,
        "signed_normalize": bundle_contract.get("signed_normalize", "per_sample"),
        "composition_rule": bundle_contract["composition_rule"],
        "bundle_path": str(bundle_path),
        "side_predictions_csv": str(side_predictions_csv),
        "bundle_contract": bundle_contract,
        "model_family": model_family,
    }


__all__ = [
    "build_patient_bundle_contract",
    "build_patient_bundle_summary",
]
