#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
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
    build_analysis_dataset,
    build_specs,
    load_metadata_manifest,
    load_model_from_checkpoint,
    parse_levels,
)


def build_free_basis_manifest(levels: tuple[int, ...]) -> list[dict]:
    rows = []
    offset = 0
    for level_index, level_size in enumerate(levels):
        for local_basis_index in range(level_size):
            global_basis_index = offset + local_basis_index
            rows.append(
                {
                    "basis_global_index": global_basis_index,
                    "basis_name": f"free_b{global_basis_index}",
                    "branch": "free",
                    "level_index": level_index,
                    "level_name": f"free_level{level_index}",
                    "local_basis_index": local_basis_index,
                    "raw_coeff_column": f"free_coeff_l{level_index}",
                }
            )
        offset += level_size
    return rows


def build_side_basis_manifest(*, side_basis_count: int, free_basis_count: int) -> list[dict]:
    return [
        {
            "basis_global_index": free_basis_count + local_basis_index,
            "basis_name": f"side_b{local_basis_index}",
            "branch": "side",
            "level_index": None,
            "level_name": "side",
            "local_basis_index": local_basis_index,
            "raw_coeff_column": "side_coeff",
        }
        for local_basis_index in range(side_basis_count)
    ]


def collate_prev_window_indices(prev_window_indices) -> np.ndarray:
    batch_size = len(prev_window_indices[0])
    seq_len = len(prev_window_indices)
    values = np.full((batch_size, seq_len), -1, dtype=np.int64)
    for frame_idx, per_frame_values in enumerate(prev_window_indices):
        for batch_idx, value in enumerate(per_frame_values):
            values[batch_idx, frame_idx] = -1 if value is None else int(value)
    return values


def extract_rows(
    model,
    loader,
    *,
    device: str,
    levels: tuple[int, ...],
    side_basis_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    free_basis_rows = build_free_basis_manifest(levels)
    free_basis_count = sum(levels)
    side_basis_rows = build_side_basis_manifest(
        side_basis_count=side_basis_count,
        free_basis_count=free_basis_count,
    )
    basis_manifest = pd.DataFrame(free_basis_rows + side_basis_rows)
    free_level_offsets = np.cumsum((0,) + levels[:-1]).astype(np.int64)

    wide_rows = []
    long_rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["images"].to(device), return_group_pooled=False)

            valid_mask = batch["valid_mask"].cpu().numpy().astype(bool)
            window_indices = batch["window_indices"].cpu().numpy().astype(np.int64)
            prev_window_indices = collate_prev_window_indices(batch["prev_window_indices"])
            free_usage = outputs["free_path_usage"].detach().cpu().numpy().astype(np.float32)
            free_activation = outputs["free_path_representation"].detach().cpu().numpy().astype(np.float32)
            free_coefficients = outputs["free_path_coefficients"].detach().cpu().numpy().astype(np.float32)
            side_usage = outputs["side_path_usage"].detach().cpu().numpy().astype(np.float32)
            side_activation = outputs["side_path_representation"].detach().cpu().numpy().astype(np.float32)
            side_coefficients = outputs["side_path_coefficients"].detach().cpu().numpy().astype(np.float32)
            decoded_indices = [
                level_indices.detach().cpu().numpy().astype(np.int64)
                for level_indices in outputs["decoded_indices"]
            ]

            group_ids = list(batch["group_id"])
            subjects = list(batch["subject"])
            dataset_names = list(batch["dataset_name"])
            side_labels = batch["side_label"].cpu().numpy().astype(np.int64)
            dataset_labels = batch["dataset_label"].cpu().numpy().astype(np.int64)

            for batch_idx in range(len(subjects)):
                for frame_idx, keep in enumerate(valid_mask[batch_idx]):
                    if not keep:
                        continue

                    row = {
                        "dataset_name": dataset_names[batch_idx],
                        "dataset_label": int(dataset_labels[batch_idx]),
                        "subject": str(subjects[batch_idx]),
                        "group_id": str(group_ids[batch_idx]),
                        "group_frame_idx": int(frame_idx),
                        "window_idx": int(window_indices[batch_idx, frame_idx]),
                        "prev_window_idx": int(prev_window_indices[batch_idx, frame_idx]),
                        "side_label": int(side_labels[batch_idx]),
                        "side_label_name": SIDE_LABEL_NAMES.get(
                            int(side_labels[batch_idx]),
                            str(int(side_labels[batch_idx])),
                        ),
                    }
                    for level_index in range(len(levels)):
                        row[f"free_coeff_l{level_index}"] = float(
                            free_coefficients[batch_idx, frame_idx, level_index]
                        )
                        row[f"free_decoded_index_l{level_index}"] = int(
                            decoded_indices[level_index][batch_idx, frame_idx]
                        )
                        level_start = int(free_level_offsets[level_index])
                        level_end = level_start + int(levels[level_index])
                        level_usage = free_usage[batch_idx, frame_idx, level_start:level_end]
                        level_argmax_local = int(np.argmax(level_usage))
                        row[f"free_argmax_usage_index_l{level_index}"] = level_argmax_local
                        row[f"free_argmax_usage_global_b_l{level_index}"] = level_start + level_argmax_local

                    row["side_coeff"] = float(side_coefficients[batch_idx, frame_idx, 0])
                    row["side_argmax_usage_index"] = int(np.argmax(side_usage[batch_idx, frame_idx]))

                    for basis_index in range(free_basis_count):
                        row[f"basis_usage_b{basis_index}"] = float(free_usage[batch_idx, frame_idx, basis_index])
                        row[f"basis_activation_b{basis_index}"] = float(
                            free_activation[batch_idx, frame_idx, basis_index]
                        )
                    for side_basis_index in range(side_basis_count):
                        global_basis_index = free_basis_count + side_basis_index
                        row[f"basis_usage_b{global_basis_index}"] = float(
                            side_usage[batch_idx, frame_idx, side_basis_index]
                        )
                        row[f"basis_activation_b{global_basis_index}"] = float(
                            side_activation[batch_idx, frame_idx, side_basis_index]
                        )
                    wide_rows.append(row)

                    base_long = {
                        "dataset_name": dataset_names[batch_idx],
                        "dataset_label": int(dataset_labels[batch_idx]),
                        "subject": str(subjects[batch_idx]),
                        "group_id": str(group_ids[batch_idx]),
                        "group_frame_idx": int(frame_idx),
                        "window_idx": int(window_indices[batch_idx, frame_idx]),
                        "prev_window_idx": int(prev_window_indices[batch_idx, frame_idx]),
                        "side_label": int(side_labels[batch_idx]),
                        "side_label_name": SIDE_LABEL_NAMES.get(
                            int(side_labels[batch_idx]),
                            str(int(side_labels[batch_idx])),
                        ),
                    }
                    for basis_row in free_basis_rows:
                        level_index = int(basis_row["level_index"])
                        local_basis_index = int(basis_row["local_basis_index"])
                        global_basis_index = int(basis_row["basis_global_index"])
                        decoded_local = int(decoded_indices[level_index][batch_idx, frame_idx])
                        argmax_local = int(row[f"free_argmax_usage_index_l{level_index}"])
                        long_rows.append(
                            {
                                **base_long,
                                **basis_row,
                                "usage": float(free_usage[batch_idx, frame_idx, global_basis_index]),
                                "raw_coeff": float(free_coefficients[batch_idx, frame_idx, level_index]),
                                "activation": float(free_activation[batch_idx, frame_idx, global_basis_index]),
                                "decoded_index_local": decoded_local,
                                "decoded_index_global": int(free_level_offsets[level_index] + decoded_local),
                                "argmax_usage_index_local": argmax_local,
                                "argmax_usage_index_global": int(free_level_offsets[level_index] + argmax_local),
                                "is_decoded_basis": int(local_basis_index == decoded_local),
                                "is_argmax_usage_basis": int(local_basis_index == argmax_local),
                            }
                        )
                    side_argmax_local = int(row["side_argmax_usage_index"])
                    for basis_row in side_basis_rows:
                        local_basis_index = int(basis_row["local_basis_index"])
                        long_rows.append(
                            {
                                **base_long,
                                **basis_row,
                                "usage": float(side_usage[batch_idx, frame_idx, local_basis_index]),
                                "raw_coeff": float(side_coefficients[batch_idx, frame_idx, 0]),
                                "activation": float(side_activation[batch_idx, frame_idx, local_basis_index]),
                                "decoded_index_local": None,
                                "decoded_index_global": None,
                                "argmax_usage_index_local": side_argmax_local,
                                "argmax_usage_index_global": int(free_basis_count + side_argmax_local),
                                "is_decoded_basis": None,
                                "is_argmax_usage_basis": int(local_basis_index == side_argmax_local),
                            }
                        )

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows), basis_manifest


def merge_metadata(df: pd.DataFrame, metadata_manifest: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        metadata_manifest,
        how="left",
        on=["dataset_name", "dataset_label", "subject", "window_idx"],
        validate="many_to_one",
    )
    if merged[["start_frame", "end_frame"]].isna().any(axis=None):
        raise RuntimeError("Failed to match metadata for some exported rows")
    return merged


def export(
    checkpoint_path: str,
    data_roots: str | None = None,
    split: str = "all",
    batch_size: int = 64,
    num_workers: int = 0,
    output_dir: str | None = None,
):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    resolved_data_roots = data_roots or config.get("data_roots")
    if not resolved_data_roots:
        raise ValueError("data_roots must be provided either via argument or checkpoint config")

    mode = str(config.get("mode", "x"))
    region = str(config.get("region", "full"))
    use_difference = bool(config.get("use_difference", True))
    signed_normalize = str(config.get("signed_normalize", "per_sample"))
    val_ratio = float(config.get("val_ratio", 0.2))
    seed = int(config.get("seed", 42))
    group_size = int(config.get("group_size", 4))
    apply_deleted_filter = bool(config.get("apply_deleted_filter", True))
    levels = parse_levels(config.get("levels", "2,6"))
    side_basis_count = int(config.get("side_basis_count", 3))

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent / f"window_basis_activations_{split}"
    )
    destination.mkdir(parents=True, exist_ok=True)

    specs = build_specs(str(resolved_data_roots))
    metadata_manifest = load_metadata_manifest(specs)
    dataset = build_analysis_dataset(
        specs,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
        group_size=group_size,
        apply_deleted_filter=apply_deleted_filter,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model, _ = load_model_from_checkpoint(checkpoint, num_dataset_classes=len(specs))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    wide_df, long_df, basis_manifest = extract_rows(
        model,
        loader,
        device=device,
        levels=levels,
        side_basis_count=side_basis_count,
    )
    wide_df = merge_metadata(wide_df, metadata_manifest)
    long_df = merge_metadata(long_df, metadata_manifest)
    wide_df = wide_df.sort_values(["dataset_name", "subject", "window_idx"]).reset_index(drop=True)
    long_df = long_df.sort_values(
        ["dataset_name", "subject", "window_idx", "basis_global_index"]
    ).reset_index(drop=True)
    basis_manifest = basis_manifest.sort_values(["basis_global_index"]).reset_index(drop=True)

    wide_path = destination / "window_basis_activations_wide.csv"
    long_path = destination / "window_basis_activations_long.csv"
    manifest_path = destination / "basis_manifest.csv"
    summary_path = destination / "summary.json"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    basis_manifest.to_csv(manifest_path, index=False)

    summary = {
        "checkpoint_path": str(checkpoint),
        "split": split,
        "num_window_rows": int(wide_df.shape[0]),
        "num_long_rows": int(long_df.shape[0]),
        "num_subjects": int(wide_df["subject"].nunique()),
        "num_groups": int(wide_df["group_id"].nunique()),
        "num_datasets": int(wide_df["dataset_name"].nunique()),
        "free_levels": [int(v) for v in levels],
        "free_basis_count": int(sum(levels)),
        "side_basis_count": int(side_basis_count),
        "total_basis_count": int(sum(levels) + side_basis_count),
        "data_roots": str(resolved_data_roots),
        "mode": mode,
        "region": region,
        "paths": {
            "wide_csv": str(wide_path),
            "long_csv": str(long_path),
            "basis_manifest_csv": str(manifest_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    fire.Fire({"export": export})
