#!/usr/bin/env python
"""
Export window-level basis usage, coefficients, and activations.

What this script does:
- Re-run one checkpoint on grouped windows.
- Keep only valid patient windows after disentangleNet grouping/padding logic.
- Export a wide table with one row per real window.
- Export a long table with one row per `(window, basis)` pair.
- Merge back metadata such as frames, side, score, and label_5class.

Typical usage:
1. Export all windows:
   `python scripts/disentangleNet/analysis/export_window_basis_activations.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt --split all`
2. Export validation windows only:
   `python scripts/disentangleNet/analysis/export_window_basis_activations.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt --split val`
3. Custom destination:
   `python scripts/disentangleNet/analysis/export_window_basis_activations.py export \\
      --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \\
      --output_dir outputs/disentangleNet/v31_current_verify/window_basis_activations_custom`

Main outputs:
- `<output_dir>/window_basis_activations_wide.csv`
- `<output_dir>/window_basis_activations_long.csv`
- `<output_dir>/basis_manifest.csv`
- `<output_dir>/summary.json`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.analyze_checkpoint import build_specs
from scripts.disentangleNet.analysis.analyze_side_interpretability import (
    SIDE_LABEL_NAMES,
    load_model_from_checkpoint,
    parse_levels,
)
from scripts.disentangleNet.data import FacialMotionSequenceDataset, subject_split
from scripts.disentangleNet.data.io import infer_subject_width


def resolve_subjects_for_split(
    spec,
    *,
    split: str,
    val_ratio: float,
    seed: int,
) -> list[str]:
    if split == "all":
        meta = pd.read_csv(spec.root / "metadata.csv")
        subject_width = infer_subject_width(spec)
        return sorted(meta["subj"].astype(str).str.zfill(subject_width).unique().tolist())

    train_subjects, val_subjects = subject_split(spec, val_ratio=val_ratio, seed=seed)
    if split == "train":
        return train_subjects
    if split == "val":
        return val_subjects
    raise ValueError(f"Unsupported split: {split!r}. Expected one of: all, train, val")


def build_analysis_dataset(
    specs,
    *,
    mode: str,
    region: str,
    use_difference: bool,
    signed_normalize: str,
    val_ratio: float,
    seed: int,
    group_size: int,
    apply_deleted_filter: bool,
    split: str,
):
    datasets = []

    for spec in specs:
        train_subjects, _ = subject_split(spec, val_ratio=val_ratio, seed=seed)
        subjects = resolve_subjects_for_split(
            spec,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
        )

        global_scale = None
        if signed_normalize == "global":
            global_scale = FacialMotionSequenceDataset.compute_global_scale(
                spec,
                train_subjects,
                mode=mode,
                region=region,
                use_difference=use_difference,
                seed=seed,
                apply_deleted_filter=apply_deleted_filter,
            )

        datasets.append(
            FacialMotionSequenceDataset(
                spec,
                subjects,
                mode=mode,
                region=region,
                use_difference=use_difference,
                signed_normalize=signed_normalize,
                global_scale=global_scale,
                group_size=group_size,
                apply_deleted_filter=apply_deleted_filter,
            )
        )

    return ConcatDataset(datasets)


def load_metadata_manifest(specs) -> pd.DataFrame:
    manifests = []
    for spec in specs:
        subject_width = infer_subject_width(spec)
        meta = pd.read_csv(spec.root / "metadata.csv").copy()
        meta["subject"] = meta["subj"].astype(str).str.zfill(subject_width)
        meta["dataset_name"] = spec.dataset_name
        meta["dataset_label"] = int(spec.dataset_label)
        manifests.append(
            meta[
                [
                    "dataset_name",
                    "dataset_label",
                    "subject",
                    "window_idx",
                    "start_frame",
                    "end_frame",
                    "side",
                    "score",
                    "label_5class",
                    "matrix_size",
                    "deleted_x",
                    "deleted_y",
                ]
            ].copy()
        )
    manifest = pd.concat(manifests, ignore_index=True)
    manifest["window_idx"] = manifest["window_idx"].astype(np.int64)
    return manifest


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
    rows = []
    for local_basis_index in range(side_basis_count):
        rows.append(
            {
                "basis_global_index": free_basis_count + local_basis_index,
                "basis_name": f"side_b{local_basis_index}",
                "branch": "side",
                "level_index": None,
                "level_name": "side",
                "local_basis_index": local_basis_index,
                "raw_coeff_column": "side_coeff",
            }
        )
    return rows


def collate_prev_window_indices(prev_window_indices) -> np.ndarray:
    if not prev_window_indices:
        return np.zeros((0, 0), dtype=np.int64)

    batch_size = len(prev_window_indices[0])
    seq_len = len(prev_window_indices)
    values = np.full((batch_size, seq_len), -1, dtype=np.int64)
    for frame_idx, per_frame_values in enumerate(prev_window_indices):
        for batch_idx, value in enumerate(per_frame_values):
            if value is None:
                values[batch_idx, frame_idx] = -1
            elif hasattr(value, "item"):
                values[batch_idx, frame_idx] = int(value.item())
            else:
                values[batch_idx, frame_idx] = int(value)
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
    wide_rows: list[dict] = []
    long_rows: list[dict] = []

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
                        row[f"basis_usage_b{basis_index}"] = float(
                            free_usage[batch_idx, frame_idx, basis_index]
                        )
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
                        argmax_local = int(
                            row[f"free_argmax_usage_index_l{level_index}"]
                        )
                        long_rows.append(
                            {
                                **base_long,
                                **basis_row,
                                "usage": float(free_usage[batch_idx, frame_idx, global_basis_index]),
                                "raw_coeff": float(
                                    free_coefficients[batch_idx, frame_idx, level_index]
                                ),
                                "activation": float(
                                    free_activation[batch_idx, frame_idx, global_basis_index]
                                ),
                                "decoded_index_local": decoded_local,
                                "decoded_index_global": int(
                                    free_level_offsets[level_index] + decoded_local
                                ),
                                "argmax_usage_index_local": argmax_local,
                                "argmax_usage_index_global": int(
                                    free_level_offsets[level_index] + argmax_local
                                ),
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
                                "activation": float(
                                    side_activation[batch_idx, frame_idx, local_basis_index]
                                ),
                                "decoded_index_local": None,
                                "decoded_index_global": None,
                                "argmax_usage_index_local": side_argmax_local,
                                "argmax_usage_index_global": int(
                                    free_basis_count + side_argmax_local
                                ),
                                "is_decoded_basis": None,
                                "is_argmax_usage_basis": int(local_basis_index == side_argmax_local),
                            }
                        )

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    return wide_df, long_df, basis_manifest


def merge_metadata(df: pd.DataFrame, metadata_manifest: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        metadata_manifest,
        how="left",
        on=["dataset_name", "dataset_label", "subject", "window_idx"],
        validate="many_to_one",
    )
    if merged[["start_frame", "end_frame"]].isna().any(axis=None):
        missing = merged[merged["start_frame"].isna()][
            ["dataset_name", "subject", "window_idx"]
        ].drop_duplicates()
        raise RuntimeError(
            "Failed to match metadata for some rows: "
            + missing.to_dict(orient="records").__repr__()
        )
    return merged


def summarize_outputs(
    *,
    checkpoint_path: Path,
    split: str,
    levels: tuple[int, ...],
    side_basis_count: int,
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> dict:
    return {
        "checkpoint_path": str(checkpoint_path),
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
    }


def export(
    checkpoint_path: str,
    data_roots: str | None = None,
    split: str = "all",
    batch_size: int = 64,
    num_workers: int = 0,
    output_dir: str | None = None,
):
    """
    Main CLI entry for window-level basis export.

    Parameters:
    - `checkpoint_path`: trained checkpoint
    - `data_roots`: optional dataset roots; defaults to checkpoint config
    - `split`: `all`, `train`, or `val`
    - `batch_size`, `num_workers`: loader controls
    - `output_dir`: destination for wide/long CSV exports
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    ckpt = torch.load(checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    resolved_data_roots = data_roots or config.get("data_roots")
    if not resolved_data_roots:
        raise ValueError("data_roots must be provided either via argument or checkpoint config")

    mode = str(config.get("mode", "x"))
    region = str(config.get("region", "mouth"))
    use_difference = bool(config.get("use_difference", True))
    signed_normalize = str(config.get("signed_normalize", "per_sample"))
    val_ratio = float(config.get("val_ratio", 0.2))
    seed = int(config.get("seed", 42))
    group_size = int(config.get("group_size", 4))
    apply_deleted_filter = bool(config.get("apply_deleted_filter", True))
    levels = parse_levels(config.get("levels", "2,3,6"))
    side_basis_count = int(config.get("side_basis_count", 0))

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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model, _ = load_model_from_checkpoint(
        checkpoint,
        num_dataset_classes=len(specs),
    )
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

    wide_df = wide_df.sort_values(
        ["dataset_name", "subject", "window_idx"]
    ).reset_index(drop=True)
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

    summary = summarize_outputs(
        checkpoint_path=checkpoint,
        split=split,
        levels=levels,
        side_basis_count=side_basis_count,
        wide_df=wide_df,
        long_df=long_df,
    )
    summary.update(
        {
            "data_roots": str(resolved_data_roots),
            "mode": mode,
            "region": region,
            "use_difference": use_difference,
            "signed_normalize": signed_normalize,
            "group_size": group_size,
            "apply_deleted_filter": apply_deleted_filter,
            "paths": {
                "wide_csv": str(wide_path),
                "long_csv": str(long_path),
                "basis_manifest_csv": str(manifest_path),
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved wide CSV: {wide_path}")
    print(f"Saved long CSV: {long_path}")
    print(f"Saved basis manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")
    return summary


if __name__ == "__main__":
    fire.Fire({"export": export})
