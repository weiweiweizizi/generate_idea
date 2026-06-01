"""
Configuration schema for disentangleNet.

Reconstructed from:
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 21-38: imports)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 408-490: build_modular_model_config)
- disentangle_modern_reconstructed/train_reflex_entry.py  (lines 194-328: build_reflex_train_config)
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 103-158: build_v31_train_config)
- disentangle_modern_reconstructed/train_v31_entry.py    (lines 162-194: load_v31_train_config)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ModelFamily(str, Enum):
    MODULAR = "modular"
    LEGACY_V31 = "legacy_v31"


class BasisModeType(str, Enum):
    DIRECT = "direct"
    LOWRANK = "lowrank"
    LOWRANK_BASIS = "lowrank_basis"


class SideBranchType(str, Enum):
    STRUCTURED_REFLEX = "structured_reflex"
    RESIDUAL = "residual"
    PAIRED_COMPETITIVE = "paired_competitive_side"


class SidePredictionInput(str, Enum):
    FREE_PATH_COEFF = "free_path_coeff"
    ACTION_USAGE = "action_usage"
    SIDE_PAIR_CHOICE_COEFF = "side_pair_choice_coeff"


class SideVariant(str, Enum):
    THREE_WAY = "three_way"
    THREE_SIDE = "three_side"


@dataclass
class BasisModeConfig:
    mode_type: BasisModeType = BasisModeType.DIRECT
    basis_size: int = 119
    levels: tuple[int, ...] = (2, 6)
    init_path: str | None = None
    orthogonalization: str = "none"
    lowrank_level_ranks: tuple[int, ...] = (3, 5)
    self_reflex_all: bool = False
    postprocess: str = "none"
    svd_truncation_rank: int | None = None


@dataclass
class ReflexConfig:
    enabled: bool = False
    self_count: int = 0
    pair_count: int = 0
    mirror_perm_source: str | None = None


@dataclass
class SideBranchConfig:
    enabled: bool = False
    variant: SideVariant = SideVariant.THREE_WAY
    branch_type: SideBranchType = SideBranchType.RESIDUAL
    prediction_input: SidePredictionInput = SidePredictionInput.FREE_PATH_COEFF
    feature_mode: str = "none"
    residual_weight: float = 1.0
    coeff_l1_weight: float = 0.0
    private_orth_weight: float = 0.0
    private_adv_weight: float = 0.0
    private_grl_lambda: float = 1.0
    pair_count: int = 0
    pair_rank: int = 0


@dataclass
class ModelConfig:
    family: ModelFamily = ModelFamily.MODULAR
    mode: str = "x"
    region: str = "mouth"
    hidden_dim: int = 32
    pool_size: int = 1
    shared_dim: int | None = None
    private_dim: int = 32
    private_decoder_hidden_dim: int | None = None
    free_pool_size: int = 2
    private_pool_size: int = 1
    free_z_dim: int | None = None
    private_branch_enabled: bool = True
    private_adapter_enabled: bool = False
    shared_trunk_attention_enabled: bool = False
    shared_trunk_attention_layers: int = 2
    shared_trunk_attention_heads: int = 4
    shared_trunk_attention_ffn_dim: int = 64
    basis: BasisModeConfig = field(default_factory=BasisModeConfig)
    reflex: ReflexConfig = field(default_factory=ReflexConfig)
    side: SideBranchConfig = field(default_factory=SideBranchConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        for sub_key in ("basis", "reflex", "side"):
            sub = d[sub_key]
            if isinstance(sub, dict):
                for kk, vv in sub.items():
                    if isinstance(vv, Enum):
                        sub[kk] = vv.value
        return d


@dataclass
class OptimizerConfig:
    optimizer_type: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    basis_lr_mult: float = 1.0


@dataclass
class SchedulerConfig:
    scheduler_type: str = "none"
    min_lr: float = 1e-6


@dataclass
class CheckpointSelectionConfig:
    best_metric: str = "loss"
    best_start_epoch: int = 1


@dataclass
class ValidationConfig:
    val_ratio: float = 0.2
    val_data_roots: str | None = None
    validate_batch_memory: bool = True
    num_workers: int = 0
    log_confusion_matrix: bool = True


@dataclass
class TrainConfig:
    data_roots: str = "data/win20-step20/IMR,data/win20-step20/TT"
    output_dir: str = "outputs/disentangleNet/default"
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    mode: str = "x"
    action_basis_init_path: str | None = None
    side_basis_init_path: str | None = None
    init_checkpoint_path: str | None = None
    private_branch_enabled: bool = True
    side_residual_enabled: bool = False
    side_feature_mode: str = "none"
    action_side_input: str = "free_path_coeff"
    side_basis_count: int = 3
    action_side_weight: float = 10.0
    freq_weight: float = 0.01
    lowrank_orth_weight: float = 0.01
    residual_weight: float = 0.0
    private_residual_weight: float = 0.20
    side_residual_weight: float = 1.0
    side_coeff_l1_weight: float = 0.0
    side_private_orth_weight: float = 0.0
    private_side_adv_weight: float = 0.0
    private_side_grl_lambda: float = 1.0
    action_side_detach: bool = False
    freeze_non_side_for_action_probe: bool = False
    action_side_init_path: str | None = None
    reflex_self_count: int = 2
    reflex_pair_count: int = 3
    rank0: int = 3
    rank1: int = 5
    side_branch_type: str = "structured_reflex_side"
    side_pair_count: int = 4
    side_pair_rank: int = 9
    shared_self_reflex_all: bool = False
    shared_selection_mode: str = "mlp_coeff"
    ordered_indices_path: str = "data/win20-step20/IMR/ordered_indices.npy"
    laplacian_topology_source: str = "scripts/matrix_vis/face_mesh_connections.py"
    laplacian_basis_weight: float = 0.0
    laplacian_side_basis_weight: float = 0.0
    laplacian_recon_weight: float = 0.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    selection: CheckpointSelectionConfig = field(default_factory=CheckpointSelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        for sub_key in ("optimizer", "scheduler", "selection", "validation"):
            sub = d[sub_key]
            if isinstance(sub, dict):
                for kk, vv in sub.items():
                    if isinstance(vv, Enum):
                        sub[kk] = vv.value
        return d

    def save_json(self, path: str | Any) -> None:
        import pathlib
        p = pathlib.Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )


@dataclass
class PipelineStageConfig:
    name: str = ""
    epochs: int = 50
    freeze_keys: list[str] = field(default_factory=list)
    unfreeze_keys: list[str] = field(default_factory=list)
    lr: float | None = None
    resume_from: str | None = None
    loss_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    stages: list[PipelineStageConfig] = field(default_factory=list)
    base_output_dir: str = "outputs/disentangleNet/pipeline"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BasisModeConfig",
    "BasisModeType",
    "CheckpointSelectionConfig",
    "ModelConfig",
    "ModelFamily",
    "OptimizerConfig",
    "PipelineConfig",
    "PipelineStageConfig",
    "ReflexConfig",
    "SchedulerConfig",
    "SideBranchConfig",
    "SideBranchType",
    "SidePredictionInput",
    "SideVariant",
    "TrainConfig",
    "ValidationConfig",
]
