#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet_trainprobe.analysis.common import (
    SIDE_LABEL_NAMES,
    build_specs,
    load_model_from_checkpoint,
)
from scripts.disentangleNet_trainprobe.data import FacialMotionSequenceDataset
from scripts.disentangleNet_trainprobe.data.io import infer_subject_width
from scripts.disentangleNet_trainprobe.regions import crop_region


def resolve_subject_for_spec(spec, subject: str) -> str | None:
    raw_subject = str(subject).strip()
    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        value = str(value).strip()
        if value and value not in candidates:
            candidates.append(value)

    add_candidate(raw_subject)
    digits_only = re.sub(r"\D", "", raw_subject)
    add_candidate(digits_only)

    subject_width = infer_subject_width(spec)
    if digits_only:
        normalized_digits = digits_only.lstrip("0") or "0"
        add_candidate(normalized_digits)
        add_candidate(normalized_digits.zfill(subject_width))
        add_candidate(digits_only.zfill(subject_width))

    for candidate in candidates:
        if (spec.root / candidate).exists():
            return candidate
    return None


def resolve_target_spec(*, data_roots: str, subject: str):
    for spec in build_specs(data_roots):
        resolved_subject = resolve_subject_for_spec(spec, subject)
        if resolved_subject is not None:
            return spec, resolved_subject
    raise FileNotFoundError(f"Subject {subject!r} was not found in any data root from {data_roots!r}")


def resolve_motion_normalization_scale(
    *,
    dataset: FacialMotionSequenceDataset,
    subject: str,
    window_idx: int,
) -> float:
    """
    返回某个真实窗口在 trainprobe 输入侧使用的 signed normalization scale。

    这让 matrix_vis 在患者级重建时能把模型输出的“归一化 ΔD”恢复到近似原始尺度。
    """
    if dataset.signed_normalize == "none":
        return 1.0
    if dataset.signed_normalize == "global":
        scale = dataset.global_scale
        if scale is None or scale <= 0:
            raise ValueError("global_scale must be provided for global normalization")
        return float(scale)
    if dataset.signed_normalize != "per_sample":
        raise ValueError(f"Unsupported signed_normalize mode: {dataset.signed_normalize!r}")

    current = dataset._load_matrix(subject, window_idx)
    if dataset.use_difference:
        previous = dataset._load_matrix(subject, window_idx - 1)
        motion = current - previous
    else:
        motion = current
    motion = crop_region(motion, dataset.region)
    scale = float(np.percentile(np.abs(motion), 98))
    return max(scale, 1e-6)


def export(
    checkpoint_path: str,
    subject: str,
    data_roots: str | None = None,
    output_dir: str | None = None,
    batch_size: int = 8,
):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    resolved_data_roots = data_roots or config.get("data_roots")
    if not resolved_data_roots:
        raise ValueError("data_roots must be provided either via argument or checkpoint config")

    spec, resolved_subject = resolve_target_spec(
        data_roots=str(resolved_data_roots),
        subject=str(subject),
    )
    dataset = FacialMotionSequenceDataset(
        spec,
        [resolved_subject],
        mode=str(config.get("mode", "x")),
        region=str(config.get("region", "full")),
        use_difference=bool(config.get("use_difference", True)),
        signed_normalize=str(config.get("signed_normalize", "per_sample")),
        group_size=int(config.get("group_size", 4)),
        apply_deleted_filter=bool(config.get("apply_deleted_filter", True)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mode = str(config.get("mode", "x"))

    model, _ = load_model_from_checkpoint(checkpoint, num_dataset_classes=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    patient_rows = []
    aggregated_basis = None
    aggregated_side_basis = None
    branch_basis_export = None
    branch_side_basis_export = None
    branch_names = list(model.BRANCH_NAMES)

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["images"].to(device), return_group_pooled=True)
            if aggregated_basis is None:
                aggregated_basis = outputs["basis"].detach().cpu().numpy().astype(np.float32)
                aggregated_side_basis = outputs["side_basis"].detach().cpu().numpy().astype(np.float32)
                branch_basis_export = np.stack(
                    [outputs["branch_action_basis"][name].detach().cpu().numpy() for name in branch_names],
                    axis=0,
                ).astype(np.float32)
                branch_side_basis_export = np.stack(
                    [outputs["branch_side_basis"][name].detach().cpu().numpy() for name in branch_names],
                    axis=0,
                ).astype(np.float32)

            valid_mask = batch["valid_mask"].cpu().numpy().astype(bool)
            window_indices = batch["window_indices"].cpu().numpy().astype(np.int64)
            group_ids = list(batch["group_id"])
            subjects = list(batch["subject"])
            dataset_names = list(batch["dataset_name"])
            side_true = batch["side_label"].cpu().numpy().astype(np.int64)
            group_side_logits = outputs["group_side_logits"].detach().cpu().numpy().astype(np.float32)
            group_side_pred = group_side_logits.argmax(axis=1).astype(np.int64)
            free_usage = outputs["free_path_usage"].detach().cpu().numpy().astype(np.float32)
            side_usage = outputs["side_path_usage"].detach().cpu().numpy().astype(np.float32)
            free_rep = outputs["free_path_representation"].detach().cpu().numpy().astype(np.float32)
            side_rep = outputs["side_path_representation"].detach().cpu().numpy().astype(np.float32)
            shared_reconstruction = (
                outputs["shared_reconstruction"].detach().cpu().numpy().astype(np.float32)
            )
            prev_window_groups = [
                [-1 if value is None else int(value) for value in per_subject]
                for per_subject in zip(*batch["prev_window_indices"])
            ]

            for batch_idx, batch_subject in enumerate(subjects):
                if str(batch_subject) != str(resolved_subject):
                    continue
                for frame_idx, keep in enumerate(valid_mask[batch_idx]):
                    if not keep:
                        continue
                    patient_rows.append(
                        {
                            "dataset_name": dataset_names[batch_idx],
                            "subject": str(batch_subject),
                            "group_id": group_ids[batch_idx],
                            "window_idx": int(window_indices[batch_idx, frame_idx]),
                            "prev_window_idx": int(prev_window_groups[batch_idx][frame_idx]),
                            "side_pred": int(group_side_pred[batch_idx]),
                            "side_true": int(side_true[batch_idx]),
                            "shared_basis_coeffs": free_rep[batch_idx, frame_idx],
                            "side_basis_coeffs": side_rep[batch_idx, frame_idx],
                            "free_path_usage": free_usage[batch_idx, frame_idx],
                            "side_path_usage": side_usage[batch_idx, frame_idx],
                            "composed_basis_matrix": shared_reconstruction[batch_idx, frame_idx, 0],
                            "observation_scale": resolve_motion_normalization_scale(
                                dataset=dataset,
                                subject=str(batch_subject),
                                window_idx=int(window_indices[batch_idx, frame_idx]),
                            ),
                            "signed_normalize": str(config.get("signed_normalize", "per_sample")),
                        }
                    )

    if not patient_rows:
        raise RuntimeError(f"No valid windows were exported for subject {subject}")

    patient_rows = sorted(patient_rows, key=lambda row: int(row["window_idx"]))
    dataset_name = str(patient_rows[0]["dataset_name"])
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / "matrix_vis_exports" / "patients" / f"{dataset_name}_{subject}"
    )
    destination.mkdir(parents=True, exist_ok=True)

    bundle_path = destination / f"patient_{subject}_{mode}_sequence.npz"
    np.savez(
        bundle_path,
        window_indices=np.asarray([row["window_idx"] for row in patient_rows], dtype=np.int64),
        prev_window_indices=np.asarray([row["prev_window_idx"] for row in patient_rows], dtype=np.int64),
        side_pred=np.asarray([row["side_pred"] for row in patient_rows], dtype=np.int64),
        side_true=np.asarray([row["side_true"] for row in patient_rows], dtype=np.int64),
        observation_scales=np.asarray(
            [row["observation_scale"] for row in patient_rows],
            dtype=np.float32,
        ),
        shared_basis_coeffs=np.stack([row["shared_basis_coeffs"] for row in patient_rows], axis=0).astype(np.float32),
        side_basis_coeffs=np.stack([row["side_basis_coeffs"] for row in patient_rows], axis=0).astype(np.float32),
        free_path_usage=np.stack([row["free_path_usage"] for row in patient_rows], axis=0).astype(np.float32),
        side_path_usage=np.stack([row["side_path_usage"] for row in patient_rows], axis=0).astype(np.float32),
        composed_basis_matrices=np.stack(
            [row["composed_basis_matrix"] for row in patient_rows],
            axis=0,
        ).astype(np.float32),
        shared_basis_bank=aggregated_basis,
        side_basis_bank=aggregated_side_basis,
        group_id=np.asarray([row["group_id"] for row in patient_rows], dtype=object),
        branch_shared_basis_banks=branch_basis_export,
        branch_side_basis_banks=branch_side_basis_export,
        branch_names=np.asarray(branch_names, dtype=object),
    )

    side_df = pd.DataFrame(
        [
            {
                "dataset_name": row["dataset_name"],
                "subject": row["subject"],
                "group_id": row["group_id"],
                "window_idx": int(row["window_idx"]),
                "prev_window_idx": int(row["prev_window_idx"]),
                "side_pred": int(row["side_pred"]),
                "side_pred_name": SIDE_LABEL_NAMES.get(int(row["side_pred"]), str(int(row["side_pred"]))),
                "side_true": int(row["side_true"]),
                "side_true_name": SIDE_LABEL_NAMES.get(int(row["side_true"]), str(int(row["side_true"]))),
                "observation_scale": float(row["observation_scale"]),
            }
            for row in patient_rows
        ]
    )
    side_csv_path = destination / f"patient_{subject}_side_predictions.csv"
    side_df.to_csv(side_csv_path, index=False)

    summary = {
        "checkpoint_path": str(checkpoint),
        "dataset_name": dataset_name,
        "subject": str(resolved_subject),
        "subject_input": str(subject),
        "mode": mode,
        "region": str(config.get("region", "full")),
        "point_layout": "face_regions_grouped",
        "point_layout_region_names": None,
        "matrix_size": int(aggregated_basis.shape[-1]),
        "num_valid_windows": int(len(patient_rows)),
        "signed_normalize": str(config.get("signed_normalize", "per_sample")),
        "composition_rule": "model_forward_shared_reconstruction",
        "bundle_path": str(bundle_path),
        "side_predictions_csv": str(side_csv_path),
    }
    summary_path = destination / f"patient_{subject}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    fire.Fire({"export": export})
