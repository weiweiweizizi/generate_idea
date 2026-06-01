from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from scripts.disentangleNet.analysis.analyze_checkpoint import plot_basis_grid
from scripts.disentangleNet_trainprobe.data import (
    DatasetSpec,
    FacialMotionSequenceDataset,
    subject_kfold_split,
    subject_split,
)
from scripts.disentangleNet_trainprobe.data.io import infer_subject_width
from scripts.disentangleNet_trainprobe.model.distnet import DistNet

SIDE_LABEL_NAMES = {
    0: "Left",
    1: "Normal",
    2: "Right",
}


def parse_levels(levels) -> tuple[int, ...]:
    if isinstance(levels, str):
        return tuple(int(v) for v in levels.split(",") if str(v).strip())
    if isinstance(levels, (tuple, list)):
        return tuple(int(v) for v in levels)
    raise TypeError(f"Unsupported levels value: {levels!r}")


def build_specs(data_roots: str) -> list[DatasetSpec]:
    roots = [Path(root).expanduser() for root in data_roots.split(",") if root.strip()]
    return [
        DatasetSpec(root=root, dataset_label=idx, dataset_name=root.name)
        for idx, root in enumerate(roots)
    ]


def resolve_subjects_for_split(
    spec: DatasetSpec,
    *,
    split: str,
    val_ratio: float,
    seed: int,
    num_folds: int = 1,
    fold_index: int | None = None,
    subjects_by_dataset: dict[str, list[str]] | None = None,
) -> list[str]:
    if subjects_by_dataset is not None:
        return sorted(subjects_by_dataset.get(spec.dataset_name, []))

    if split == "all":
        meta = pd.read_csv(spec.root / "metadata.csv")
        subject_width = infer_subject_width(spec)
        return sorted(meta["subj"].astype(str).str.zfill(subject_width).unique().tolist())

    if num_folds > 1:
        if fold_index is None:
            raise ValueError("fold_index must be provided when num_folds > 1")
        train_subjects, val_subjects = subject_kfold_split(
            spec,
            num_folds=num_folds,
            fold_index=fold_index,
            seed=seed,
        )
    else:
        train_subjects, val_subjects = subject_split(spec, val_ratio=val_ratio, seed=seed)
    if split == "train":
        return train_subjects
    if split == "val":
        return val_subjects
    raise ValueError(f"Unsupported split: {split!r}. Expected one of: all, train, val")


def build_analysis_dataset(
    specs: list[DatasetSpec],
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
    num_folds: int = 1,
    fold_index: int | None = None,
    subjects_by_dataset: dict[str, list[str]] | None = None,
):
    datasets = []
    for spec in specs:
        if num_folds > 1:
            if fold_index is None:
                raise ValueError("fold_index must be provided when num_folds > 1")
            train_subjects, _ = subject_kfold_split(
                spec,
                num_folds=num_folds,
                fold_index=fold_index,
                seed=seed,
            )
        else:
            train_subjects, _ = subject_split(spec, val_ratio=val_ratio, seed=seed)
        subjects = resolve_subjects_for_split(
            spec,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            num_folds=num_folds,
            fold_index=fold_index,
            subjects_by_dataset=subjects_by_dataset,
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


def load_metadata_manifest(specs: list[DatasetSpec]) -> pd.DataFrame:
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
    manifest["window_idx"] = manifest["window_idx"].astype(int)
    return manifest


def load_fold_manifest(path: str | Path) -> dict:
    manifest_path = Path(path).expanduser().resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    num_dataset_classes: int,
) -> tuple[DistNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = dict(ckpt.get("config", {}))

    model = DistNet(
        levels=parse_levels(config.get("levels", "2,6")),
        basis_size=int(config.get("basis_size", 341)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        pool_size=int(config.get("pool_size", 1)),
        shared_dim=config.get("shared_dim"),
        private_dim=int(config.get("private_dim", 32)),
        private_decoder_hidden_dim=config.get("private_decoder_hidden_dim"),
        num_side_classes=int(config.get("num_side_classes", 3)),
        num_severity_classes=3,
        num_dataset_classes=num_dataset_classes,
        target_label_mode=str(config.get("target_label_mode", "side")),
        private_residual_weight=float(config.get("private_residual_weight", 0.05)),
        private_residual_max_l1=config.get("private_residual_max_l1"),
        shared_basis_soft_mixing=bool(config.get("shared_basis_soft_mixing", True)),
        shared_basis_anchor_bias=float(config.get("shared_basis_anchor_bias", 2.0)),
        shared_basis_topk=config.get("shared_basis_topk"),
        grl_lambda=float(config.get("grl_lambda", 1.0)),
        use_dataset_aux=bool(config.get("use_dataset_aux", False)),
        side_semantic_enabled=bool(config.get("side_semantic_enabled", True)),
        side_basis_count=int(config.get("side_basis_count", 3)),
        side_pooling=str(config.get("side_pooling", "tri_region_contrast")),
        side_subspace_dim=config.get("side_subspace_dim"),
        side_free_frame_qr=bool(config.get("side_free_frame_qr", False)),
        free_side_grl_lambda=float(config.get("free_side_grl_lambda", 1.0)),
        early_branch_factorization=bool(config.get("early_branch_factorization", True)),
        free_pool_size=int(config.get("free_pool_size", 2)),
        side_pool_size=int(config.get("side_pool_size", 2)),
        private_pool_size=int(config.get("private_pool_size", 1)),
        free_z_dim=config.get("free_z_dim"),
        side_z_dim=config.get("side_z_dim"),
        private_adapter_enabled=bool(config.get("private_adapter_enabled", False)),
        action_basis_init_path=None,
        side_basis_init_path=None,
        lq_commitment_loss_weight=float(config.get("lq_commitment_loss_weight", 0.1)),
        lq_quantization_loss_weight=float(config.get("lq_quantization_loss_weight", 0.1)),
        lq_optimize_values=bool(config.get("lq_optimize_values", True)),
        quantizer_type=str(config.get("quantizer_type", "residual_fsq")),
        fsq_preserve_symmetry=bool(config.get("fsq_preserve_symmetry", True)),
        basis_orthogonalization=str(config.get("basis_orthogonalization", "joint_global_qr")),
        discrete_side_loss_enabled=bool(config.get("discrete_side_loss_enabled", False)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, config
