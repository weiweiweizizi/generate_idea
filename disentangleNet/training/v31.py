"""
V31 training entry.

Reconstructed from:
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 25-93: V31_FIXED_CONFIG)
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 95-100: build_v31_config)
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 103-158: build_v31_train_config)
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 162-194: load_v31_train_config)
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 197-391: train function)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from disentangleNet.config import (
    CheckpointSelectionConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    ValidationConfig,
)

DEFAULT_ACTION_BASIS_INIT_PATH = "disentangleNet/init_basis/basis_x_shared_2_6.npy"
DEFAULT_SIDE_BASIS_INIT_PATH = "disentangleNet/init_basis/basis_x_side_from_level2.npy"
DEFAULT_OUTPUT_DIR = "outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50"

V31_FIXED_CONFIG = {
    "action_basis_init_path": DEFAULT_ACTION_BASIS_INIT_PATH,
    "side_basis_init_path": DEFAULT_SIDE_BASIS_INIT_PATH,
    "mode": "x",
    "region": "mouth",
    "use_difference": True,
    "signed_normalize": "per_sample",
    "group_size": 4,
    "apply_deleted_filter": True,
    "basis_size": 119,
    "levels": "2,6",
    "hidden_dim": 32,
    "pool_size": 1,
    "early_branch_factorization": True,
    "free_pool_size": 2,
    "side_pool_size": 2,
    "private_pool_size": 1,
    "free_z_dim": None,
    "side_z_dim": 32,
    "private_adapter_enabled": False,
    "shared_dim": None,
    "private_dim": 32,
    "private_decoder_hidden_dim": None,
    "recon_weight": 1.0,
    "shared_recon_weight": 1.0,
    "lq_weight": 10.0,
    "orth_weight": 0.1,
    "basis_l1_weight": 1.0,
    "residual_weight": 0.02,
    "side_weight": 0.0,
    "side_cont_weight": 0.0,
    "side_disc_weight": 0.0,
    "target_label_mode": "side",
    "num_side_classes": 3,
    "side_semantic_enabled": True,
    "side_basis_count": 3,
    "side_pooling": "fixed_region2_contrast",
    "static_side_input_enabled": False,
    "static_side_fusion_mode": "add",
    "side_loss_weight": 0.3,
    "group_side_loss_weight": 0.3,
    "side_subspace_dim": None,
    "side_free_frame_qr": False,
    "subspace_orth_weight": 0.0,
    "free_side_adv_weight": 0.0,
    "free_side_grl_lambda": 1.0,
    "severity_loss_weight": 0.0,
    "dataset_private_weight": 0.0,
    "dataset_adv_weight": 0.0,
    "private_residual_weight": 0.05,
    "private_residual_max_l1": 0.5,
    "shared_basis_soft_mixing": True,
    "shared_basis_anchor_bias": 2.0,
    "shared_basis_topk": 2,
    "grl_lambda": 1.0,
    "use_dataset_aux": False,
    "lq_commitment_loss_weight": 0.1,
    "lq_quantization_loss_weight": 0.1,
    "lq_optimize_values": True,
    "quantizer_type": "residual_fsq",
    "fsq_preserve_symmetry": True,
    "basis_orthogonalization": "joint_global_qr",
    "discrete_side_loss_enabled": False,
    "require_basis_init": True,
}


def build_v31_config(runtime_config: dict) -> dict:
    config = {
        **V31_FIXED_CONFIG,
        **runtime_config,
    }
    from disentangleNet.training.config import prepare_train_config
    return prepare_train_config(config)


def build_v31_train_config(runtime_config: dict) -> TrainConfig:
    runtime_config = dict(runtime_config)
    output_dir = str(runtime_config.get("output_dir", DEFAULT_OUTPUT_DIR))
    action_basis_init_path = runtime_config.get("action_basis_init_path", DEFAULT_ACTION_BASIS_INIT_PATH)
    side_basis_init_path = runtime_config.get("side_basis_init_path", DEFAULT_SIDE_BASIS_INIT_PATH)
    optimizer = OptimizerConfig(
        lr=float(runtime_config.get("lr", 3e-4)),
        weight_decay=float(runtime_config.get("weight_decay", 1e-4)),
        basis_lr_mult=1.0,
    )
    validation = ValidationConfig(
        val_ratio=float(runtime_config.get("val_ratio", 0.2)),
        validate_batch_memory=bool(runtime_config.get("validate_batch_memory", True)),
        num_workers=int(runtime_config.get("num_workers", 0)),
        log_confusion_matrix=False,
    )
    excluded = {
        "data_roots", "output_dir", "seed", "epochs", "batch_size", "mode",
        "action_basis_init_path", "side_basis_init_path", "lr", "weight_decay",
        "val_ratio", "validate_batch_memory", "num_workers", "ordered_indices_path",
        "laplacian_topology_source", "laplacian_basis_weight",
        "laplacian_side_basis_weight", "laplacian_recon_weight",
    }
    extra = {key: value for key, value in runtime_config.items() if key not in excluded}
    return TrainConfig(
        data_roots=str(runtime_config.get("data_roots", "data/win20-step20/IMR,data/win20-step20/TT")),
        output_dir=output_dir,
        seed=int(runtime_config.get("seed", 42)),
        epochs=int(runtime_config.get("epochs", 50)),
        batch_size=int(runtime_config.get("batch_size", 64)),
        mode=str(runtime_config.get("mode", "x")),
        action_basis_init_path=action_basis_init_path,
        side_basis_init_path=side_basis_init_path,
        ordered_indices_path=str(runtime_config.get("ordered_indices_path", "data/win20-step20/IMR/ordered_indices.npy")),
        laplacian_topology_source=str(runtime_config.get("laplacian_topology_source", "scripts/matrix_vis/face_mesh_connections.py")),
        laplacian_basis_weight=float(runtime_config.get("laplacian_basis_weight", 0.0)),
        laplacian_side_basis_weight=float(runtime_config.get("laplacian_side_basis_weight", 0.0)),
        laplacian_recon_weight=float(runtime_config.get("laplacian_recon_weight", 0.0)),
        optimizer=optimizer,
        scheduler=SchedulerConfig(),
        selection=CheckpointSelectionConfig(),
        validation=validation,
        extra=extra,
    )


def load_v31_train_config(path: str | Path) -> TrainConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    optimizer_payload = payload.get("optimizer", {})
    validation_payload = payload.get("validation", {})
    return TrainConfig(
        data_roots=str(payload["data_roots"]),
        output_dir=str(payload["output_dir"]),
        seed=int(payload.get("seed", 42)),
        epochs=int(payload.get("epochs", 50)),
        batch_size=int(payload.get("batch_size", 64)),
        mode=str(payload.get("mode", "x")),
        action_basis_init_path=payload.get("action_basis_init_path"),
        side_basis_init_path=payload.get("side_basis_init_path"),
        ordered_indices_path=str(payload.get("ordered_indices_path", "data/win20-step20/IMR/ordered_indices.npy")),
        laplacian_topology_source=str(payload.get("laplacian_topology_source", "scripts/matrix_vis/face_mesh_connections.py")),
        laplacian_basis_weight=float(payload.get("laplacian_basis_weight", 0.0)),
        laplacian_side_basis_weight=float(payload.get("laplacian_side_basis_weight", 0.0)),
        laplacian_recon_weight=float(payload.get("laplacian_recon_weight", 0.0)),
        optimizer=OptimizerConfig(
            lr=float(optimizer_payload.get("lr", 3e-4)),
            weight_decay=float(optimizer_payload.get("weight_decay", 1e-4)),
            basis_lr_mult=float(optimizer_payload.get("basis_lr_mult", 1.0)),
        ),
        scheduler=SchedulerConfig(),
        selection=CheckpointSelectionConfig(),
        validation=ValidationConfig(
            val_ratio=float(validation_payload.get("val_ratio", 0.2)),
            validate_batch_memory=bool(validation_payload.get("validate_batch_memory", True)),
            num_workers=int(validation_payload.get("num_workers", 0)),
            log_confusion_matrix=False,
        ),
        extra=dict(payload.get("extra", {})),
    )


def train_v31(config_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    V31 training entry point.

    The core training loop is reconstructed from:
    - disentangle_modern_reconstructed/train_v31_entry.py  (lines 197-391)

    Note: requires a working model forward pass from models/families/.
    """
    raise NotImplementedError(
        "train_v31 requires the full model forward pass to be verified. "
        "Use the original train_v31_entry.py from disentangle_modern_reconstructed/ "
        "as a reference until models/families/ are fully restored."
    )


__all__ = [
    "V31_FIXED_CONFIG",
    "DEFAULT_ACTION_BASIS_INIT_PATH",
    "DEFAULT_SIDE_BASIS_INIT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "build_v31_config",
    "build_v31_train_config",
    "load_v31_train_config",
    "train_v31",
]
