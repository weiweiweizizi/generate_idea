#!/usr/bin/env python
"""按 phase 统一导出并重建 disentangleNet 训练结果。

这个脚本把 `phaseA / phaseB / phaseC` 的后处理固化成一条稳定流水线：

1. 患者导出
2. 患者 x 轴重建（首窗 `D0` + 后续 delta 递推；y 使用患者自身静止坐标）
3. 患者 preview
4. basis 导出
5. basis x 轴重建（y 固定为标准脸模板静止坐标）
6. basis preview

目录规范固定为：

`<run_root>/<phase>/patient/<patient_id>/`
`<run_root>/<phase>/basis/`

其中 `phaseC` 的患者导出会把 private residual 一并加进最终观测矩阵，
但 basis 导出仍只包含 shared / side basis。

默认只运行患者新流程和 basis 旧流程。若需要保留旧的
`standardFace` 患者结果，可通过显式参数单独开启。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import fire
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from disentangleNet.analysis.exporters.basis import export_basis
from disentangleNet.analysis.exporters.patient import export_patient
from disentangleNet.io_utils import save_json
from scripts.matrix_vis.core.landmark_layout import resolve_subset_layout
from scripts.matrix_vis.io.load_patient_reference import load_label_first_line_normalized_points
from scripts.matrix_vis.pipelines.compose_patient_static_y import compose_patient_static_y
from scripts.matrix_vis.pipelines.patient_sequence import (
    DEFAULT_LANDMARK_CONFIG,
    build_default_projection,
    run_patient_sequence,
)
from scripts.matrix_vis.pipelines.preview_real_mouth_regions import (
    DEFAULT_REGION_NAMES,
    run_preview_real_mouth_regions,
)
from scripts.matrix_vis.scripts.run_disentanglenet_basis_batch import run_batch


DEFAULT_RUN_ROOT = "outputs/disentangleNet_frame/reflex_pair_side_imr_tt"
DEFAULT_PHASES = ("phaseA", "phaseB", "phaseC")
DEFAULT_PATIENT_ID = "TT_851519"
DEFAULT_PATIENT_IDS = ("TT_851519",)
DEFAULT_ANCHOR_POINT_IDS = (205, 425, 200)


def _normalize_export_subject(patient_id: str) -> str:
    patient_id = str(patient_id).strip()
    for delimiter in ("_", "-"):
        if delimiter in patient_id:
            _, suffix = patient_id.split(delimiter, 1)
            if suffix:
                return suffix
    return patient_id


def _normalize_patient_subdir(patient_id: str) -> str:
    return str(patient_id).strip().replace("/", "_")


def _resolve_patient_data_roots(patient_id: str) -> str | None:
    patient_id = str(patient_id).strip()
    if patient_id.startswith("IMR_") or patient_id.startswith("IMR-"):
        return "data/win20-step20/IMR"
    if patient_id.startswith("TTMORECF_") or patient_id.startswith("TTMORECF-"):
        return "data/win20-step20/TTMORECF"
    if patient_id.startswith("TTMOREC_") or patient_id.startswith("TTMOREC-"):
        return "data/win20-step20/TTMOREC"
    if patient_id.startswith("TT_") or patient_id.startswith("TT-"):
        return "data/win20-step20/TT"
    if patient_id.startswith("XW_") or patient_id.startswith("XW-"):
        return "data/win20-step20/XW"
    return None


def _patient_numeric_id(patient_id: str) -> str:
    patient_id = str(patient_id).strip()
    for delimiter in ("_", "-"):
        if delimiter in patient_id:
            _, suffix = patient_id.split(delimiter, 1)
            if suffix:
                return suffix
    return patient_id


def _resolve_patient_data_dir(patient_id: str) -> Path:
    numeric_id = _patient_numeric_id(patient_id)
    preferred_root = _resolve_patient_data_roots(patient_id)
    if preferred_root is not None:
        candidate = Path(preferred_root) / numeric_id
        if (candidate / "lmks_crop.label").exists() and (candidate / "win_000_x.npy").exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Could not find patient init dir for {patient_id} under {preferred_root}/{numeric_id}"
        )

    roots = sorted(Path("data/win20-step20").glob(f"*/{numeric_id}"))
    candidates = [
        path for path in roots
        if (path / "lmks_crop.label").exists()
        and (path / "win_000_x.npy").exists()
    ]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"Could not find unique patient init dir for {patient_id} under data/win20-step20/*/{numeric_id}"
        )
    raise ValueError(
        f"Multiple candidate patient init dirs found for {patient_id}: {[str(path) for path in candidates]}"
    )


def _prepare_patient_initialization_files(
    *,
    patient_id: str,
    patient_root: Path,
    matrix_size: int,
) -> tuple[Path, Path, Path]:
    patient_data_dir = _resolve_patient_data_dir(patient_id)
    init_dir = (patient_root / "patient_init").resolve()
    init_dir.mkdir(parents=True, exist_ok=True)

    landmark_points = load_label_first_line_normalized_points(patient_data_dir / "lmks_crop.label")
    landmark_output_path = init_dir / "initial_landmarks_first_line_normalized.npy"
    np.save(landmark_output_path, landmark_points.astype(np.float32, copy=False))

    full_d1 = np.load(patient_data_dir / "win_000_x.npy").astype(np.float32, copy=False)
    if full_d1.shape[0] == int(matrix_size):
        d1_subset = full_d1
    else:
        full_layout = resolve_subset_layout(
            subset_layout="face_regions_grouped",
            subset_layout_source=DEFAULT_LANDMARK_CONFIG,
            subset_layout_extractor_name="mediapipe",
            subset_layout_region_names=None,
        )
        subset_point_ids = np.asarray(
            build_default_projection(axis="x", matrix_size=matrix_size).subset_point_ids,
            dtype=np.int64,
        )
        index_by_point_id = {int(point_id): idx for idx, point_id in enumerate(full_layout)}
        subset_indices = [index_by_point_id[int(point_id)] for point_id in subset_point_ids.tolist()]
        d1_subset = full_d1[np.ix_(subset_indices, subset_indices)].astype(np.float32, copy=False)
    d1_output_path = init_dir / "initial_distance_matrix_win000_x.npy"
    np.save(d1_output_path, d1_subset)
    return patient_data_dir, landmark_output_path, d1_output_path


def _run_phase_basis(
    *,
    checkpoint_path: Path,
    phase: str,
    basis_root: Path,
    anchor_point_ids: list[int] | tuple[int, ...],
    lambda_laplacian: float,
    lambda_area_sign: float,
    area_barrier_margin: float,
) -> dict[str, Any]:
    summary_path = basis_root / "basis_reconstruction_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    basis_export_dir = basis_root / "basis_export"
    basis_generated_dir = basis_root / "generated_configs"
    basis_reconstruction_root = basis_root / "reconstructions"
    basis_preview_root = basis_root / "preview_x_no_motion_y"

    basis_export_summary = export_basis(
        checkpoint_path=str(checkpoint_path),
        output_dir=str(basis_export_dir),
    )
    basis_manifest_path = Path(basis_export_summary["manifest_path"]).resolve()
    basis_batch_summary = run_batch(
        manifest_path=str(basis_manifest_path),
        generated_dir=str(basis_generated_dir),
        reconstruction_root=str(basis_reconstruction_root),
        preview_root=str(basis_preview_root),
        anchor_point_ids=tuple(int(point_id) for point_id in anchor_point_ids),
        run_fixed_other_preview=False,
        run_no_motion_other_preview=True,
        lambda_laplacian=lambda_laplacian,
        lambda_area_sign=lambda_area_sign,
        area_barrier_margin=area_barrier_margin,
    )

    basis_summary = {
        "phase": str(phase),
        "checkpoint_path": str(checkpoint_path),
        "anchor_point_ids": [int(point_id) for point_id in anchor_point_ids],
        "lambda_laplacian": float(lambda_laplacian),
        "lambda_area_sign": float(lambda_area_sign),
        "area_barrier_margin": float(area_barrier_margin),
        "basis": {
            "export_dir": str(basis_export_dir),
            "manifest_path": str(basis_manifest_path),
            "generated_dir": str(basis_generated_dir),
            "reconstruction_root": str(basis_reconstruction_root),
            "preview_root": str(basis_preview_root),
            "export_summary": basis_export_summary,
            "batch_summary": basis_batch_summary,
        },
    }
    save_json(summary_path, basis_summary)
    return basis_summary


def _run_phase_patient(
    *,
    checkpoint_path: Path,
    phase: str,
    patient_root: Path,
    patient_id: str,
    anchor_point_ids: list[int] | tuple[int, ...],
    include_initial_reference_window: bool,
    lambda_acc: float,
    max_displacement: float | None,
    lambda_laplacian: float,
    lambda_area_sign: float,
    area_barrier_margin: float,
    lambda_trajectory_tether: float,
    include_standard_face_patient: bool = False,
) -> dict[str, Any]:
    summary_path = patient_root / "patient_reconstruction_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    patient_bundle_dir = patient_root / "patient_bundle"
    patient_sequence_dir = patient_root / "matrix_vis_sequence_x_no_motion_y"
    patient_preview_dir = patient_root / "preview_x_patient_static_y"
    patient_export_summary = export_patient(
        checkpoint_path=str(checkpoint_path),
        subject=_normalize_export_subject(patient_id),
        data_roots=_resolve_patient_data_roots(patient_id),
        output_dir=str(patient_bundle_dir),
    )
    patient_bundle_path = Path(patient_export_summary["bundle_path"]).resolve()
    patient_data_dir, initial_landmark_path, initial_distance_matrix_path = _prepare_patient_initialization_files(
        patient_id=patient_id,
        patient_root=patient_root,
        matrix_size=int(patient_export_summary["matrix_size"]),
    )

    patient_sequence_summary = run_patient_sequence(
        patient_bundle_path=str(patient_bundle_path),
        output_dir=str(patient_sequence_dir),
        initial_landmark_source=str(initial_landmark_path),
        initial_distance_matrix_source=str(initial_distance_matrix_path),
        include_initial_reference_window=include_initial_reference_window,
        lambda_acc=lambda_acc,
        max_displacement=max_displacement,
        carry_forward_initial_positions=True,
        lambda_laplacian=lambda_laplacian,
        lambda_area_sign=lambda_area_sign,
        area_barrier_margin=area_barrier_margin,
        lambda_trajectory_tether=lambda_trajectory_tether,
    )
    patient_preview_summary = compose_patient_static_y(
        sequence_dir=str(patient_sequence_dir),
        output_dir=str(patient_preview_dir),
        initial_landmark_source=str(initial_landmark_path),
    )

    patient_summary = {
        "phase": str(phase),
        "checkpoint_path": str(checkpoint_path),
        "patient_id": str(patient_id),
        "anchor_point_ids": [int(point_id) for point_id in anchor_point_ids],
        "include_standard_face_patient": bool(include_standard_face_patient),
        "include_initial_reference_window": bool(include_initial_reference_window),
        "lambda_acc": float(lambda_acc),
        "max_displacement": None if max_displacement is None else float(max_displacement),
        "lambda_laplacian": float(lambda_laplacian),
        "lambda_area_sign": float(lambda_area_sign),
        "area_barrier_margin": float(area_barrier_margin),
        "lambda_trajectory_tether": float(lambda_trajectory_tether),
        "patient": {
            "bundle_dir": str(patient_bundle_dir),
            "bundle_path": str(patient_bundle_path),
            "patient_data_dir": str(patient_data_dir),
            "patient_init_dir": str((patient_root / "patient_init").resolve()),
            "initial_landmark_source": str(initial_landmark_path),
            "initial_distance_matrix_source": str(initial_distance_matrix_path),
            "sequence_dir": str(patient_sequence_dir),
            "preview_dir": str(patient_preview_dir),
            "export_summary": patient_export_summary,
            "sequence_summary": patient_sequence_summary,
            "preview_summary": patient_preview_summary,
        },
    }
    if include_standard_face_patient:
        patient_sequence_standard_dir = patient_root / "matrix_vis_sequence_x_no_motion_y_standardFace"
        patient_preview_standard_dir = patient_root / "preview_x_no_motion_y_standardFace"
        patient_sequence_standard_summary = run_patient_sequence(
            patient_bundle_path=str(patient_bundle_path),
            output_dir=str(patient_sequence_standard_dir),
            lambda_acc=lambda_acc,
            max_displacement=max_displacement,
            carry_forward_initial_positions=True,
            lambda_laplacian=lambda_laplacian,
            lambda_area_sign=lambda_area_sign,
            area_barrier_margin=area_barrier_margin,
            lambda_trajectory_tether=lambda_trajectory_tether,
        )
        patient_preview_standard_summary = run_preview_real_mouth_regions(
            x_solution=str(patient_sequence_standard_dir / "solution.npz"),
            y_solution=None,
            output_dir=str(patient_preview_standard_dir),
            anchor_point_ids=tuple(int(point_id) for point_id in anchor_point_ids),
            subset_layout_region_names=DEFAULT_REGION_NAMES,
            title=f"{phase} patient preview x + no_motion_y_standardFace",
            static_y=True,
            align_to_anchor=False,
        )
        patient_summary["patient"]["standard_face"] = {
            "sequence_dir": str(patient_sequence_standard_dir),
            "preview_dir": str(patient_preview_standard_dir),
            "sequence_summary": patient_sequence_standard_summary,
            "preview_summary": patient_preview_standard_summary,
        }
    save_json(summary_path, patient_summary)
    return patient_summary


def run_phase(
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    phase: str,
    patient_id: str = DEFAULT_PATIENT_ID,
    anchor_point_ids: list[int] | tuple[int, ...] = DEFAULT_ANCHOR_POINT_IDS,
    include_initial_reference_window: bool = False,
    lambda_acc: float = 10.0,
    max_displacement: float | None = 0.2,
    lambda_laplacian: float = 1.0,
    lambda_area_sign: float = 1.0,
    area_barrier_margin: float = 0.05,
    lambda_trajectory_tether: float = 0.0,
    include_standard_face_patient: bool = False,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    phase_dir = root / str(phase)
    checkpoint_path = phase_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    patient_root = phase_dir / "patient" / _normalize_patient_subdir(patient_id)
    basis_root = phase_dir / "basis"

    basis_summary = _run_phase_basis(
        checkpoint_path=checkpoint_path,
        phase=str(phase),
        basis_root=basis_root,
        anchor_point_ids=anchor_point_ids,
        lambda_laplacian=lambda_laplacian,
        lambda_area_sign=lambda_area_sign,
        area_barrier_margin=area_barrier_margin,
    )
    patient_summary = _run_phase_patient(
        checkpoint_path=checkpoint_path,
        phase=str(phase),
        patient_root=patient_root,
        patient_id=str(patient_id),
        anchor_point_ids=anchor_point_ids,
        include_initial_reference_window=include_initial_reference_window,
        lambda_acc=lambda_acc,
        max_displacement=max_displacement,
        lambda_laplacian=lambda_laplacian,
        lambda_area_sign=lambda_area_sign,
        area_barrier_margin=area_barrier_margin,
        lambda_trajectory_tether=lambda_trajectory_tether,
        include_standard_face_patient=include_standard_face_patient,
    )

    phase_summary = {
        "phase": str(phase),
        "checkpoint_path": str(checkpoint_path),
        "patient_id": str(patient_id),
        "include_standard_face_patient": bool(include_standard_face_patient),
        "include_initial_reference_window": bool(include_initial_reference_window),
        "lambda_trajectory_tether": float(lambda_trajectory_tether),
        "patient_root": str(patient_root),
        "basis_root": str(basis_root),
        "patient": patient_summary["patient"],
        "basis": basis_summary["basis"],
    }
    save_json(patient_root / "phase_reconstruction_summary.json", phase_summary)
    print(json.dumps(phase_summary, indent=2, ensure_ascii=False))
    return phase_summary


def run_all(
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    phases: list[str] | tuple[str, ...] = DEFAULT_PHASES,
    patient_id: str = DEFAULT_PATIENT_ID,
    anchor_point_ids: list[int] | tuple[int, ...] = DEFAULT_ANCHOR_POINT_IDS,
    include_initial_reference_window: bool = False,
    lambda_acc: float = 10.0,
    max_displacement: float | None = 0.2,
    lambda_laplacian: float = 1.0,
    lambda_area_sign: float = 1.0,
    area_barrier_margin: float = 0.05,
    lambda_trajectory_tether: float = 0.0,
    include_standard_face_patient: bool = False,
) -> dict[str, Any]:
    summaries = []
    for phase in phases:
        summaries.append(
            run_phase(
                run_root=run_root,
                phase=str(phase),
                patient_id=patient_id,
                anchor_point_ids=anchor_point_ids,
                include_initial_reference_window=include_initial_reference_window,
                lambda_acc=lambda_acc,
                max_displacement=max_displacement,
                lambda_laplacian=lambda_laplacian,
                lambda_area_sign=lambda_area_sign,
                area_barrier_margin=area_barrier_margin,
                lambda_trajectory_tether=lambda_trajectory_tether,
                include_standard_face_patient=include_standard_face_patient,
            )
        )

    root = Path(run_root).expanduser().resolve()
    summary = {
        "run_root": str(root),
        "patient_id": str(patient_id),
        "include_standard_face_patient": bool(include_standard_face_patient),
        "include_initial_reference_window": bool(include_initial_reference_window),
        "phases": [str(phase) for phase in phases],
        "anchor_point_ids": [int(point_id) for point_id in anchor_point_ids],
        "lambda_acc": float(lambda_acc),
        "max_displacement": None if max_displacement is None else float(max_displacement),
        "lambda_laplacian": float(lambda_laplacian),
        "lambda_area_sign": float(lambda_area_sign),
        "area_barrier_margin": float(area_barrier_margin),
        "lambda_trajectory_tether": float(lambda_trajectory_tether),
        "phase_summaries": summaries,
    }
    save_json(root / "phase_comparison_reconstruction_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_patients(
    *,
    run_root: str = DEFAULT_RUN_ROOT,
    phases: list[str] | tuple[str, ...] = DEFAULT_PHASES,
    patient_ids: list[str] | tuple[str, ...] = DEFAULT_PATIENT_IDS,
    anchor_point_ids: list[int] | tuple[int, ...] = DEFAULT_ANCHOR_POINT_IDS,
    include_initial_reference_window: bool = False,
    lambda_acc: float = 10.0,
    max_displacement: float | None = 0.2,
    lambda_laplacian: float = 1.0,
    lambda_area_sign: float = 1.0,
    area_barrier_margin: float = 0.05,
    lambda_trajectory_tether: float = 0.0,
    include_standard_face_patient: bool = False,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    phase_summaries: list[dict[str, Any]] = []

    for phase in phases:
        phase_dir = root / str(phase)
        checkpoint_path = phase_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        basis_root = phase_dir / "basis"
        basis_summary = _run_phase_basis(
            checkpoint_path=checkpoint_path,
            phase=str(phase),
            basis_root=basis_root,
            anchor_point_ids=anchor_point_ids,
            lambda_laplacian=lambda_laplacian,
            lambda_area_sign=lambda_area_sign,
            area_barrier_margin=area_barrier_margin,
        )

        patient_summaries: list[dict[str, Any]] = []
        for patient_id in patient_ids:
            patient_root = phase_dir / "patient" / _normalize_patient_subdir(str(patient_id))
            patient_summary = _run_phase_patient(
                checkpoint_path=checkpoint_path,
                phase=str(phase),
                patient_root=patient_root,
                patient_id=str(patient_id),
                anchor_point_ids=anchor_point_ids,
                include_initial_reference_window=include_initial_reference_window,
                lambda_acc=lambda_acc,
                max_displacement=max_displacement,
                lambda_laplacian=lambda_laplacian,
                lambda_area_sign=lambda_area_sign,
                area_barrier_margin=area_barrier_margin,
                lambda_trajectory_tether=lambda_trajectory_tether,
                include_standard_face_patient=include_standard_face_patient,
            )
            phase_patient_summary = {
                "phase": str(phase),
                "checkpoint_path": str(checkpoint_path),
                "patient_id": str(patient_id),
                "patient_root": str(patient_root),
                "basis_root": str(basis_root),
                "patient": patient_summary["patient"],
                "basis": basis_summary["basis"],
            }
            save_json(patient_root / "phase_reconstruction_summary.json", phase_patient_summary)
            patient_summaries.append(phase_patient_summary)

        phase_summary = {
            "phase": str(phase),
            "checkpoint_path": str(checkpoint_path),
            "basis_root": str(basis_root),
            "basis": basis_summary["basis"],
            "patients": patient_summaries,
        }
        save_json(phase_dir / "phase_reconstruction_summary.json", phase_summary)
        phase_summaries.append(phase_summary)

    summary = {
        "run_root": str(root),
        "patient_ids": [str(patient_id) for patient_id in patient_ids],
        "include_standard_face_patient": bool(include_standard_face_patient),
        "include_initial_reference_window": bool(include_initial_reference_window),
        "phases": [str(phase) for phase in phases],
        "anchor_point_ids": [int(point_id) for point_id in anchor_point_ids],
        "lambda_acc": float(lambda_acc),
        "max_displacement": None if max_displacement is None else float(max_displacement),
        "lambda_laplacian": float(lambda_laplacian),
        "lambda_area_sign": float(lambda_area_sign),
        "area_barrier_margin": float(area_barrier_margin),
        "lambda_trajectory_tether": float(lambda_trajectory_tether),
        "phase_summaries": phase_summaries,
    }
    save_json(root / "multi_patient_phase_comparison_reconstruction_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire({"run_phase": run_phase, "run_all": run_all, "run_patients": run_patients})
