#!/usr/bin/env python
"""
Build action-basis initialization tensors for the LQ model.

Why this script exists:
- `DistNet.action_basis_bank` needs a deterministic, interpretable warm start.
- Existing grouped SVD results already give useful priors for source and side.
- Action-pattern priors are sparse, so we mix one known grouped SVD basis
  (`by_severity/mild`) with a small K-means basis bank built from single-patient
  PC1 results.

Output ordering must match `levels=(2, 3, 6)` in the current network:
1. by_source: 2 bases  -> IMR, TT
2. by_side:   3 bases  -> left_affected, bilateral_normal, right_affected
3. action:    6 bases  -> mild + 5 K-means cluster centers

The script is run separately for `mode=x` and `mode=y`, because the current
training setup uses independent branches for the two motion directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.lq.regions import REGION_BOUNDARIES, REGION_NAMES, crop_region
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from regions import REGION_BOUNDARIES, REGION_NAMES, crop_region

BY_SOURCE_ORDER = ["IMR", "TT"]
BY_SIDE_ORDER = ["left_affected", "bilateral_normal", "right_affected"]


def enforce_matrix_constraints(mat: np.ndarray) -> np.ndarray:
    """
    Project a basis matrix into the structural subspace expected by the model.

    Distance / difference-distance matrices are symmetric, and the diagonal
    should not carry useful motion information. We therefore force symmetry and
    zero the diagonal before using the basis for initialization.
    """

    mat = 0.5 * (mat + mat.T)
    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)
    return mat


def canonicalize_sign(mat: np.ndarray) -> np.ndarray:
    """
    Fix the arbitrary sign ambiguity of singular vectors.

    SVD bases can be flipped by ±1 without changing the decomposition. K-means
    would treat such sign-flipped copies as different samples unless we first
    choose a consistent orientation. We use the largest-magnitude entry as the
    sign anchor.
    """

    flat = mat.reshape(-1)
    pivot = int(np.argmax(np.abs(flat)))
    if flat[pivot] < 0:
        mat = -mat
    return mat


def normalize_basis(mat: np.ndarray) -> np.ndarray:
    """Normalize one basis to unit Frobenius norm for stable initialization."""

    norm = float(np.linalg.norm(mat))
    if norm <= 1e-8:
        return mat.astype(np.float32)
    return (mat / norm).astype(np.float32)


def preprocess_basis(mat: np.ndarray, region: str) -> np.ndarray:
    """
    Apply the full preprocessing pipeline shared by all initialization bases.

    The order matters:
    1. crop to the model region
    2. enforce matrix structure
    3. canonicalize sign
    4. normalize magnitude
    """

    mat = crop_region(mat, region)
    mat = enforce_matrix_constraints(mat)
    mat = canonicalize_sign(mat)
    return normalize_basis(mat)


def load_group_basis(grouped_dir: Path, group_mode: str, group_name: str, mode: str, region: str) -> np.ndarray:
    """Load one grouped SVD PC1 basis and convert it into model-ready format."""

    path = grouped_dir / group_mode / group_name / f"PC1_{mode}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return preprocess_basis(np.load(path), region)


def list_subject_pc1_paths(root: Path, mode: str) -> list[Path]:
    """Enumerate single-patient PC1 files for one motion direction."""

    return sorted(root.glob(f"*/PC1_{mode}.npy"))


def sample_patient_pc1_paths(
    tt_root: Path,
    imr_root: Path,
    mode: str,
    total_samples: int,
    tt_priority_count: int,
    seed: int,
) -> list[Path]:
    """
    Sample single-patient PC1 bases with TT preference.

    Research rationale:
    - TT is more heterogeneous in practice, so we bias the small action-basis
      K-means pool toward TT.
    - IMR still acts as a fallback / supplement when TT alone is insufficient.
    """

    rng = np.random.RandomState(seed)
    tt_paths = list_subject_pc1_paths(tt_root, mode)
    imr_paths = list_subject_pc1_paths(imr_root, mode)

    rng.shuffle(tt_paths)
    rng.shuffle(imr_paths)

    picked = tt_paths[: min(tt_priority_count, len(tt_paths))]
    remaining = total_samples - len(picked)

    if remaining > 0:
        extra_tt = tt_paths[len(picked) : len(picked) + remaining]
        picked.extend(extra_tt)
        remaining = total_samples - len(picked)

    if remaining > 0:
        picked.extend(imr_paths[:remaining])

    if len(picked) < total_samples:
        raise ValueError(
            f"Not enough single-patient PC1 files to sample {total_samples}; only found {len(picked)}"
        )

    return picked[:total_samples]


def run_kmeans(data: np.ndarray, k: int, seed: int, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal NumPy K-means used only for basis initialization.

    Keeping this local avoids bringing in an extra sklearn dependency just to
    build a tiny initialization bank.
    """

    rng = np.random.RandomState(seed)
    if len(data) < k:
        raise ValueError(f"k={k} cannot exceed sample count={len(data)}")

    center_indices = rng.choice(len(data), size=k, replace=False)
    centers = data[center_indices].copy()

    assignments = np.zeros(len(data), dtype=np.int64)
    for _ in range(max_iters):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assignments = distances.argmin(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        for idx in range(k):
            members = data[assignments == idx]
            if len(members) == 0:
                centers[idx] = data[rng.randint(len(data))]
            else:
                centers[idx] = members.mean(axis=0)

    return centers, assignments


def build_action_cluster_bases(
    tt_root: Path,
    imr_root: Path,
    mode: str,
    region: str,
    sample_count: int,
    cluster_count: int,
    tt_priority_count: int,
    seed: int,
) -> tuple[list[np.ndarray], dict]:
    """
    Build the 5 K-means action bases from single-patient PC1 tensors.

    Each sampled patient PC1 is preprocessed into the same matrix space used by
    the network. K-means is then run on flattened matrices, and cluster centers
    are reshaped back into square motion bases.
    """

    sampled_paths = sample_patient_pc1_paths(
        tt_root=tt_root,
        imr_root=imr_root,
        mode=mode,
        total_samples=sample_count,
        tt_priority_count=tt_priority_count,
        seed=seed,
    )

    matrices = [preprocess_basis(np.load(path), region) for path in sampled_paths]
    flat = np.stack([mat.reshape(-1) for mat in matrices], axis=0)
    centers, assignments = run_kmeans(flat, k=cluster_count, seed=seed)

    cluster_bases = []
    basis_size = matrices[0].shape[0]
    for center in centers:
        # Cluster centers live in flattened matrix space; after reshaping we
        # re-apply the structural cleanup to avoid tiny numerical asymmetries.
        mat = center.reshape(basis_size, basis_size)
        mat = enforce_matrix_constraints(mat)
        mat = canonicalize_sign(mat)
        cluster_bases.append(normalize_basis(mat))

    metadata = {
        "sampled_subjects": [path.parent.name for path in sampled_paths],
        "sampled_paths": [str(path) for path in sampled_paths],
        "assignments": assignments.tolist(),
        "cluster_count": cluster_count,
        "sample_count": sample_count,
        "tt_priority_count": tt_priority_count,
    }
    return cluster_bases, metadata


def build_basis_bank(
    grouped_dir: Path,
    tt_svd_dir: Path,
    imr_svd_dir: Path,
    mode: str,
    region: str,
    seed: int,
    sample_count: int,
    cluster_count: int,
    tt_priority_count: int,
) -> tuple[np.ndarray, dict]:
    """
    Assemble the full initialization bank in network order.

    Final layout:
    [ by_source(2) | by_side(3) | action(6) ]

    This ordering is not arbitrary: `DistNet.split_basis()` slices the bank by
    level boundaries, so the initialization tensor must already be arranged to
    match `levels=(2, 3, 6)`.
    """

    source_bases = [
        load_group_basis(grouped_dir, "by_source", group_name, mode, region)
        for group_name in BY_SOURCE_ORDER
    ]
    side_bases = [
        load_group_basis(grouped_dir, "by_side", group_name, mode, region)
        for group_name in BY_SIDE_ORDER
    ]

    mild_basis = load_group_basis(grouped_dir, "by_severity", "mild", mode, region)
    cluster_bases, cluster_meta = build_action_cluster_bases(
        tt_root=tt_svd_dir,
        imr_root=imr_svd_dir,
        mode=mode,
        region=region,
        sample_count=sample_count,
        cluster_count=cluster_count,
        tt_priority_count=tt_priority_count,
        seed=seed,
    )

    action_bases = [mild_basis, *cluster_bases]
    bank = np.stack([*source_bases, *side_bases, *action_bases], axis=0).astype(np.float32)

    metadata = {
        "mode": mode,
        "region": region,
        "bank_shape": list(bank.shape),
        "ordering": {
            "by_source": BY_SOURCE_ORDER,
            "by_side": BY_SIDE_ORDER,
            "action": ["mild", *[f"kmeans_{i+1}" for i in range(cluster_count)]],
        },
        "paths": {
            "grouped_dir": str(grouped_dir),
            "tt_svd_dir": str(tt_svd_dir),
            "imr_svd_dir": str(imr_svd_dir),
        },
        "cluster_metadata": cluster_meta,
    }
    return bank, metadata


def parse_args() -> argparse.Namespace:
    """CLI for generating one direction-specific initialization tensor."""

    parser = argparse.ArgumentParser(description="Build action basis initialization from grouped and single-patient SVD bases.")
    parser.add_argument("--mode", choices=["x", "y"], required=True)
    parser.add_argument("--region", choices=["full", "mouth"], default="mouth")
    parser.add_argument(
        "--grouped_dir",
        default="data/win20-step20/svd_multi_patient_grouped_results",
    )
    parser.add_argument("--tt_svd_dir", default="data/win20-step20/TT-SVD")
    parser.add_argument("--imr_svd_dir", default="data/win20-step20/IMR-SVD")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_count", type=int, default=10)
    parser.add_argument("--cluster_count", type=int, default=5)
    parser.add_argument("--tt_priority_count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bank, metadata = build_basis_bank(
        grouped_dir=Path(args.grouped_dir),
        tt_svd_dir=Path(args.tt_svd_dir),
        imr_svd_dir=Path(args.imr_svd_dir),
        mode=args.mode,
        region=args.region,
        seed=args.seed,
        sample_count=args.sample_count,
        cluster_count=args.cluster_count,
        tt_priority_count=args.tt_priority_count,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, bank)

    # The JSON sidecar is important for reproducibility: because the K-means
    # bank is built from a random 10-patient subset, we need to save which
    # patients and which ordering produced the final tensor.
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"Saved basis bank to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"bank shape: {bank.shape}")


if __name__ == "__main__":
    main()
