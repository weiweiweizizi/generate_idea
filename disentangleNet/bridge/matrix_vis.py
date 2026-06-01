"""
Bridge between disentangleNet patient bundles and matrix_vis.

Reconstructed from:
- scripts/matrix_vis/docs/disentanglenet_matrix_vis_contract.md  (Sections 3.3, 6, 8)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PatientBundleBridge:
    """
    Runtime object consumed by ``scripts/matrix_vis/pipelines/patient_sequence.py``.

    Attributes mirror the fields documented in the bridge contract.
    """

    bundle_path: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    contract: dict[str, Any]
    mode: str
    matrix_size: int
    dataset_name: str
    subject: str
    group_ids: Any = None
    observation_scales: Any = None


def _find_summary_json(bundle_npz_path: Path) -> Path | None:
    """Locate the ``patient_*_summary.json`` sibling of a bundle NPZ."""
    parent = bundle_npz_path.parent
    stem = bundle_npz_path.stem
    # patient_<subject>_x_sequence.npz  ->  patient_<subject>_summary.json
    # Try replacing the _x_sequence suffix with _summary
    if "_x_sequence" in stem:
        summary_stem = stem.replace("_x_sequence", "_summary")
    elif "_sequence" in stem:
        summary_stem = stem.replace("_sequence", "_summary")
    else:
        summary_stem = stem + "_summary"
    candidate = parent / f"{summary_stem}.json"
    if candidate.exists():
        return candidate
    # Fallback: look for any *_summary.json in the same directory
    for f in parent.glob("*_summary.json"):
        return f
    return None


def _load_summary(summary_path: Path) -> dict[str, Any]:
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def _safe_default_contract() -> dict[str, Any]:
    """Safe default bundle contract for legacy outputs that lack an explicit contract."""
    return {
        "framework": "disentangleNet_lowrank",
        "mode": "x",
        "region": "mouth",
        "matrix_size": 119,
        "value_semantics": "mean_distance_delta",
        "observation_matrix_space": "normalized_input_space",
        "observation_scale_semantics": "per_window_restore_scale",
        "composition_rule": "shared_coeff_weighted_basis_plus_decoded_side_reconstruction_plus_private_residual",
        "includes_private_residual": True,
    }


def load_patient_bundle_bridge(bundle_npz_path: str | Path) -> PatientBundleBridge:
    """
    Load a patient bundle NPZ and its companion summary JSON into a
    ``PatientBundleBridge``.
    """
    npz_path = Path(bundle_npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Patient bundle not found: {npz_path}")

    data_raw = np.load(npz_path, allow_pickle=True)
    data = {k: data_raw[k] for k in data_raw.files}

    summary_path = _find_summary_json(npz_path)
    if summary_path is not None:
        metadata = _load_summary(summary_path)
    else:
        metadata = {}

    contract = metadata.get("bundle_contract", None)
    if contract is None or not isinstance(contract, dict):
        contract = _safe_default_contract()

    mode = str(contract.get("mode", metadata.get("mode", "x")))
    matrix_size = int(contract.get("matrix_size", metadata.get("matrix_size", 119)))
    dataset_name = str(metadata.get("dataset_name", ""))
    subject = str(metadata.get("subject", ""))

    group_ids = data.get("group_id", None)
    observation_scales = data.get("observation_scales", None)

    return PatientBundleBridge(
        bundle_path=str(npz_path),
        data=data,
        metadata=metadata,
        contract=contract,
        mode=mode,
        matrix_size=matrix_size,
        dataset_name=dataset_name,
        subject=subject,
        group_ids=group_ids,
        observation_scales=observation_scales,
    )


def restore_physical_observation_scale(
    observation_matrix: np.ndarray,
    scale: float | np.ndarray,
) -> np.ndarray:
    """
    Multiply a normalized observation matrix by the per-window scale factor
    to restore physical values.

    Used when ``contract["observation_matrix_space"] == "normalized_input_space"``.
    """
    if scale is None or (isinstance(scale, (int, float)) and scale == 1.0):
        return observation_matrix.astype(np.float32, copy=False)
    return (observation_matrix.astype(np.float32) * np.float32(scale)).astype(np.float32)


__all__ = [
    "PatientBundleBridge",
    "load_patient_bundle_bridge",
    "restore_physical_observation_scale",
]
