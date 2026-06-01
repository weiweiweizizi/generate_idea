from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from disentangleNet.analysis.contracts import build_patient_bundle_contract, build_patient_bundle_summary
from disentangleNet.analysis.loaders import infer_checkpoint_contract, load_model_for_analysis
from disentangleNet.analysis.utils import (
    compute_level_boundaries,
    compute_shared_frame_weights,
    compose_window_matrix,
    expand_level_coefficients_to_basis_weights,
    get_shared_basis_bank,
    get_side_basis_bank,
    parse_levels,
    resolve_bridge_point_layout,
    resolve_bridge_point_layout_region_names,
    resolve_motion_normalization_scale,
    resolve_target_spec,
    reshape_sequence_feature,
    save_json,
)
from disentangleNet.data import FacialMotionSequenceDataset
from disentangleNet.training.data import build_specs

SIDE_LABEL_NAMES = {
    0: "Left",
    1: "Normal",
    2: "Right",
}


def combine_patient_observation_matrix(
    *,
    shared_component: np.ndarray,
    side_component: np.ndarray,
    private_component: np.ndarray | None = None,
) -> np.ndarray:
    composed = shared_component.astype(np.float32, copy=False) + side_component.astype(np.float32, copy=False)
    if private_component is not None:
        composed = composed + private_component.astype(np.float32, copy=False)
    return composed.astype(np.float32, copy=False)


def export_patient(
    *,
    checkpoint_path: str,
    subject: str,
    data_roots: str | None = None,
    output_dir: str | None = None,
    batch_size: int = 8,
):
    """
    Export a patient bundle from a disentangleNet checkpoint.

    Outputs:
    - ``patient_<subject>_x_sequence.npz``
    - ``patient_<subject>_side_predictions.csv``
    - ``patient_<subject>_summary.json``
    """
    from disentangleNet.bridge.matrix_vis import restore_physical_observation_scale
    from disentangleNet.data import DatasetSpec, FacialMotionSequenceDataset

    ckpt = Path(checkpoint_path).expanduser().resolve()
    contract = infer_checkpoint_contract(ckpt)

    data_roots_str = data_roots or contract.config.get("data_roots", "")
    dataset_name = Path(data_roots_str).name if data_roots_str else "unknown"

    roots = [r.strip() for r in data_roots_str.split(",") if r.strip()] if data_roots_str else []
    specs = build_specs(data_roots_str) if data_roots_str else []

    # Load model
    num_classes = max(len(specs), 1)
    model, config, contract = load_model_for_analysis(ckpt, num_dataset_classes=num_classes)
    if model is None:
        raise RuntimeError(
            "Full model loading is required for patient export. "
            "Ensure disentangleNet.models.families is fully restored."
        )

    mode = str(config.get("mode", "x"))
    region = str(config.get("region", "mouth"))
    matrix_size = int(config.get("basis_size", 119))
    levels = parse_levels(config.get("levels"), default=contract.levels)
    level_boundaries = compute_level_boundaries(levels)

    basis = get_shared_basis_bank(model)
    side_basis = get_side_basis_bank(model)

    # Build dataset for this patient
    import pandas as pd

    for root_str in roots:
        root = Path(root_str)
        meta_path = root / "metadata.csv"
        if not meta_path.exists():
            continue
        meta = pd.read_csv(meta_path)
        subj_col = meta["subj"].astype(str).str.lstrip("0")
        if subject not in subj_col.values:
            continue
        spec = DatasetSpec(root=root, dataset_label=0, dataset_name=root.name)
        dataset = FacialMotionSequenceDataset(
            spec=spec,
            subjects=[subject],
            mode=mode,
            region=region,
            use_difference=bool(config.get("use_difference", True)),
            signed_normalize=str(config.get("signed_normalize", "per_sample")),
            global_scale=None,
            group_size=int(config.get("group_size", 4)),
            apply_deleted_filter=bool(config.get("apply_deleted_filter", True)),
            static_side_input_enabled=bool(config.get("static_side_input_enabled", False)),
            ordered_indices_path=config.get("ordered_indices_path"),
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        device = next(model.parameters()).device

        window_indices = []
        composed_matrices = []
        side_preds_list = []
        side_true_list = []
        observation_scales = []

        with torch.no_grad():
            for batch in loader:
                x = batch["images"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                side_labels = batch.get("side_label")
                if side_labels is not None:
                    side_labels = side_labels.to(device)
                static_side_input = batch.get("static_side_input")
                if isinstance(static_side_input, torch.Tensor):
                    static_side_input = static_side_input.to(device)

                outputs = model(
                    x,
                    side_labels=side_labels,
                    dataset_labels=None,
                    valid_mask=valid_mask,
                    static_side_input=static_side_input,
                )

                # Per-window basis coefficients
                coeffs = outputs.get("free_path_coeff", outputs.get("action_usage", None))
                if coeffs is not None:
                    coeffs_np = coeffs.detach().cpu().numpy().astype(np.float32)
                else:
                    continue

                # Compose per-window observation matrix
                for w in range(coeffs_np.shape[0]):
                    level_weights = expand_level_coefficients_to_basis_weights(
                        coeffs_np[w], level_boundaries,
                    )
                    window_mat = compose_window_matrix(level_weights, basis)
                    composed_matrices.append(window_mat)

                # Side predictions
                side_logits = outputs.get("side_logits", None)
                if side_logits is not None:
                    side_preds = side_logits.argmax(dim=-1).detach().cpu().numpy()
                    side_preds_list.extend(side_preds.tolist())

                if side_labels is not None:
                    side_true_list.extend(side_labels.detach().cpu().tolist())

                if len(window_indices) == 0:
                    window_indices.extend(list(range(coeffs_np.shape[0])))
                else:
                    last_idx = window_indices[-1]
                    window_indices.extend(list(range(last_idx + 1, last_idx + 1 + coeffs_np.shape[0])))

        if not composed_matrices:
            continue

        composed = np.stack(composed_matrices, axis=0).astype(np.float32)
        side_pred_arr = np.array(side_preds_list, dtype=np.int64) if side_preds_list else None
        side_true_arr = np.array(side_true_list, dtype=np.int64) if side_true_list else None

        out_dir = (
            Path(output_dir).expanduser().resolve() if output_dir
            else ckpt.parent / "matrix_vis_exports" / "patients" / subject
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save NPZ bundle
        npz_payload: dict[str, Any] = {
            "window_indices": np.array(window_indices, dtype=np.int64),
            "composed_basis_matrices": composed,
        }
        if side_pred_arr is not None:
            npz_payload["side_pred"] = side_pred_arr
        if side_true_arr is not None:
            npz_payload["side_true"] = side_true_arr

        npz_path = out_dir / f"patient_{subject}_x_sequence.npz"
        np.savez(npz_path, **npz_payload)

        # Save side predictions CSV
        if side_preds_list:
            side_df = pd.DataFrame({
                "window_idx": window_indices[:len(side_preds_list)],
                "side_pred": side_preds_list,
                "side_pred_name": [SIDE_LABEL_NAMES.get(s, str(s)) for s in side_preds_list],
            })
            if side_true_list:
                side_df["side_true"] = side_true_list[:len(side_preds_list)]
                side_df["side_true_name"] = [
                    SIDE_LABEL_NAMES.get(s, str(s)) for s in side_true_list[:len(side_preds_list)]
                ]
            side_df.to_csv(out_dir / f"patient_{subject}_side_predictions.csv", index=False)

        # Save summary JSON
        bundle_contract = build_patient_bundle_contract(
            framework=contract.framework,
            mode=mode,
            region=region,
            matrix_size=matrix_size,
        )
        summary = build_patient_bundle_summary(
            framework=contract.framework,
            checkpoint_path=str(ckpt),
            dataset_name=dataset_name,
            subject=subject,
            mode=mode,
            region=region,
            matrix_size=matrix_size,
            num_valid_windows=len(window_indices),
            point_layout=resolve_bridge_point_layout(region=region),
            point_layout_region_names=resolve_bridge_point_layout_region_names(region=region) or [],
            bundle_contract=bundle_contract,
            bundle_path=str(npz_path),
            side_predictions_csv=str(out_dir / f"patient_{subject}_side_predictions.csv"),
            model_family=contract.model_family,
        )
        save_json(out_dir / f"patient_{subject}_summary.json", summary)

        print(json.dumps({
            "npz_path": str(npz_path),
            "num_windows": len(window_indices),
            "basis_shape": list(basis.shape),
        }, indent=2, ensure_ascii=False))
        return {
            "npz_path": str(npz_path),
            "summary_path": str(out_dir / f"patient_{subject}_summary.json"),
        }

    raise FileNotFoundError(f"No data found for subject {subject} in {data_roots_str}")
