#!/usr/bin/env python

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

import fire
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.disentangleNet.analysis.analyze_checkpoint import (
    build_eval_dataset,
    build_specs,
    collect_group_representations,
)
from scripts.disentangleNet.analysis.analyze_side_interpretability import (
    SIDE_LABEL_NAMES,
    collect_group_side_semantics,
    load_model_from_checkpoint,
)


try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


def parse_subject(group_id: str) -> str:
    parts = str(group_id).split(":")
    if len(parts) < 2:
        raise ValueError(f"Unexpected group_id format: {group_id}")
    return parts[1]


def parse_dataset_name(group_id: str) -> str:
    parts = str(group_id).split(":")
    if len(parts) < 1:
        raise ValueError(f"Unexpected group_id format: {group_id}")
    return parts[0]


def build_full_eval_dataset(
    *,
    data_roots: str,
    mode: str,
    region: str,
    use_difference: bool,
    signed_normalize: str,
    val_ratio: float,
    seed: int,
    group_size: int,
    apply_deleted_filter: bool,
):
    specs = build_specs(data_roots)
    train_dataset = build_eval_dataset(
        specs,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
        group_size=group_size,
        apply_deleted_filter=apply_deleted_filter,
        split="train",
    )
    val_dataset = build_eval_dataset(
        specs,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
        group_size=group_size,
        apply_deleted_filter=apply_deleted_filter,
        split="val",
    )
    return specs, ConcatDataset([train_dataset, val_dataset])


def combine_group_representations(group_outputs: dict) -> pd.DataFrame:
    group_ids = group_outputs["group_representations"]["group_id"]
    side_labels = group_outputs["group_representations"]["side_label"]
    dataset_labels = group_outputs["group_representations"]["dataset_label"]
    return pd.DataFrame(
        {
            "group_id": group_ids.astype(str),
            "subject": [parse_subject(group_id) for group_id in group_ids],
            "dataset_name": [parse_dataset_name(group_id) for group_id in group_ids],
            "side_label": side_labels.astype(np.int64),
            "side_label_name": [SIDE_LABEL_NAMES.get(int(label), str(int(label))) for label in side_labels],
            "dataset_label": dataset_labels.astype(np.int64),
        }
    )


def align_side_semantics(
    branch_df: pd.DataFrame,
    side_df: pd.DataFrame,
) -> pd.DataFrame:
    side_df = side_df.copy()
    branch_df = branch_df.copy()
    merged = branch_df.merge(
        side_df,
        on=["group_id", "subject", "dataset_name", "side_label", "dataset_label", "side_label_name"],
        how="left",
        validate="one_to_one",
    )
    if merged.isna().any().any():
        missing = merged[merged.isna().any(axis=1)]["group_id"].tolist()
        raise RuntimeError(f"Failed to align side semantics for groups: {missing[:5]}")
    return merged


def resolve_stratify_labels(
    manifest_df: pd.DataFrame,
    *,
    requested_splits: int,
) -> tuple[str, pd.Series, int]:
    subject_df = (
        manifest_df.groupby("subject", as_index=False)
        .agg(
            side_label=("side_label", "first"),
            dataset_label=("dataset_label", "first"),
            num_groups=("group_id", "count"),
        )
        .copy()
    )
    subject_df["joint_label"] = (
        subject_df["side_label"].astype(str) + "|d" + subject_df["dataset_label"].astype(str)
    )

    joint_counts = subject_df["joint_label"].value_counts()
    if not joint_counts.empty and int(joint_counts.min()) >= requested_splits:
        return "joint_side_dataset", subject_df.set_index("subject")["joint_label"], requested_splits

    side_counts = subject_df["side_label"].value_counts()
    if side_counts.empty:
        raise RuntimeError("No subject labels found for k-fold stratification")
    resolved_splits = min(int(requested_splits), int(side_counts.min()))
    if resolved_splits < 2:
        raise RuntimeError(f"Need at least 2 subjects per class, got side counts {side_counts.to_dict()}")
    return "side_label_only", subject_df.set_index("subject")["side_label"].astype(str), resolved_splits


def greedy_subject_stratified_folds(
    manifest_df: pd.DataFrame,
    *,
    stratify_labels: pd.Series,
    num_splits: int,
    seed: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subject_counts = manifest_df.groupby("subject")["group_id"].count().to_dict()
    unique_labels = sorted(set(stratify_labels.astype(str).tolist()))
    subjects_by_label: dict[str, list[str]] = defaultdict(list)
    for subject, label in stratify_labels.astype(str).items():
        subjects_by_label[label].append(subject)

    fold_label_group_counts = {label: [0 for _ in range(num_splits)] for label in unique_labels}
    fold_total_group_counts = [0 for _ in range(num_splits)]
    fold_subject_counts = [0 for _ in range(num_splits)]
    assignments: dict[str, int] = {}

    label_order = sorted(
        unique_labels,
        key=lambda label: (len(subjects_by_label[label]), sum(subject_counts[s] for s in subjects_by_label[label])),
    )
    for label in label_order:
        subjects = subjects_by_label[label][:]
        rng.shuffle(subjects)
        subjects.sort(key=lambda subject: subject_counts[subject], reverse=True)
        for subject in subjects:
            best_fold = min(
                range(num_splits),
                key=lambda fold_idx: (
                    fold_label_group_counts[label][fold_idx],
                    fold_total_group_counts[fold_idx],
                    fold_subject_counts[fold_idx],
                    fold_idx,
                ),
            )
            assignments[subject] = int(best_fold)
            fold_label_group_counts[label][best_fold] += int(subject_counts[subject])
            fold_total_group_counts[best_fold] += int(subject_counts[subject])
            fold_subject_counts[best_fold] += 1

    subject_fold_rows = []
    for subject, fold_idx in sorted(assignments.items(), key=lambda item: (item[1], item[0])):
        subject_fold_rows.append(
            {
                "subject": subject,
                "fold": int(fold_idx),
                "num_groups": int(subject_counts[subject]),
                "stratify_label": str(stratify_labels.loc[subject]),
            }
        )
    return assignments, pd.DataFrame(subject_fold_rows)


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    unique_labels = sorted(int(label) for label in np.unique(labels).tolist())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    encoded = np.asarray([label_to_index[int(label)] for label in labels], dtype=np.int64)
    return encoded, label_to_index, index_to_label


def standardize_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def fit_predict_torch_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    epochs: int = 300,
) -> np.ndarray:
    torch.manual_seed(seed)

    x_train = torch.from_numpy(train_x.astype(np.float32))
    y_train = torch.from_numpy(train_y.astype(np.int64))
    x_test = torch.from_numpy(test_x.astype(np.float32))

    classifier = torch.nn.Linear(x_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = classifier(x_test).argmax(dim=1).cpu().numpy().astype(np.int64)
    return predictions


def confusion_matrix_from_predictions(
    true_encoded: np.ndarray,
    pred_encoded: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(true_encoded.tolist(), pred_encoded.tolist()):
        matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def compute_classification_metrics(confusion: np.ndarray) -> dict:
    total = int(confusion.sum())
    diag = np.diag(confusion).astype(np.float64)
    row_sums = confusion.sum(axis=1).astype(np.float64)
    col_sums = confusion.sum(axis=0).astype(np.float64)

    accuracy = float(diag.sum() / total) if total > 0 else 0.0
    recalls = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums > 0)
    precisions = np.divide(diag, col_sums, out=np.zeros_like(diag), where=col_sums > 0)
    f1 = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(diag),
        where=(precisions + recalls) > 0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recalls.mean()) if recalls.size else 0.0,
        "macro_precision": float(precisions.mean()) if precisions.size else 0.0,
        "macro_recall": float(recalls.mean()) if recalls.size else 0.0,
        "macro_f1": float(f1.mean()) if f1.size else 0.0,
        "support": total,
        "per_class_precision": precisions.tolist(),
        "per_class_recall": recalls.tolist(),
        "per_class_f1": f1.tolist(),
    }


def render_confusion_matrix(
    confusion: np.ndarray,
    *,
    label_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    cell_w = 120
    cell_h = 72
    left_margin = 160
    top_margin = 120
    width = left_margin + cell_w * confusion.shape[1] + 24
    height = top_margin + cell_h * confusion.shape[0] + 24
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    vmax = max(int(confusion.max()), 1)
    draw.text((24, 24), title, fill=(0, 0, 0))
    draw.text((24, 56), "Rows=True, Cols=Pred", fill=(80, 80, 80))

    for col_idx, label in enumerate(label_names):
        x = left_margin + col_idx * cell_w + 8
        draw.text((x, 88), label, fill=(0, 0, 0))
    for row_idx, label in enumerate(label_names):
        y = top_margin + row_idx * cell_h + 24
        draw.text((24, y), label, fill=(0, 0, 0))

    for row_idx in range(confusion.shape[0]):
        for col_idx in range(confusion.shape[1]):
            value = int(confusion[row_idx, col_idx])
            intensity = float(value / vmax)
            color = (
                int(255 - 35 * intensity),
                int(255 - 115 * intensity),
                int(255 - 185 * intensity),
            )
            x0 = left_margin + col_idx * cell_w
            y0 = top_margin + row_idx * cell_h
            x1 = x0 + cell_w - 4
            y1 = y0 + cell_h - 4
            draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=color, outline=(180, 180, 180))
            draw.text((x0 + 12, y0 + 26), str(value), fill=(0, 0, 0))

    canvas.save(output_path)


def evaluate_probe_task(
    *,
    task_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    label_names_map: dict[int, str],
    manifest_df: pd.DataFrame,
    subject_to_fold: dict[str, int],
    num_splits: int,
    seed: int,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError(f"{task_name}: expected 2D features, got {features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"{task_name}: feature/label mismatch {features.shape[0]} vs {labels.shape[0]}")

    encoded_labels, _, index_to_label = encode_labels(labels)
    predictions = np.full_like(encoded_labels, fill_value=-1)
    fold_rows = []

    for fold_idx in range(num_splits):
        test_mask = manifest_df["subject"].map(subject_to_fold).to_numpy(dtype=np.int64) == fold_idx
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            raise RuntimeError(f"{task_name}: fold {fold_idx} has empty train/test split")

        train_x, test_x = standardize_features(features[train_mask], features[test_mask])
        train_y = encoded_labels[train_mask]
        test_y = encoded_labels[test_mask]
        fold_pred = fit_predict_torch_linear(
            train_x,
            train_y,
            test_x,
            num_classes=len(index_to_label),
            seed=seed + fold_idx,
        )
        predictions[test_mask] = fold_pred
        fold_confusion = confusion_matrix_from_predictions(
            test_y,
            fold_pred,
            num_classes=len(index_to_label),
        )
        fold_metrics = compute_classification_metrics(fold_confusion)
        fold_rows.append(
            {
                "task_name": task_name,
                "fold": int(fold_idx),
                "num_train_groups": int(train_mask.sum()),
                "num_test_groups": int(test_mask.sum()),
                "accuracy": fold_metrics["accuracy"],
                "balanced_accuracy": fold_metrics["balanced_accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
            }
        )

    if (predictions < 0).any():
        raise RuntimeError(f"{task_name}: some out-of-fold predictions were not assigned")

    confusion = confusion_matrix_from_predictions(
        encoded_labels,
        predictions,
        num_classes=len(index_to_label),
    )
    metrics = compute_classification_metrics(confusion)
    label_names = [label_names_map[index_to_label[idx]] for idx in range(len(index_to_label))]

    confusion_df = pd.DataFrame(confusion, index=label_names, columns=label_names)
    normalized_confusion = confusion.astype(np.float64) / np.clip(
        confusion.sum(axis=1, keepdims=True).astype(np.float64),
        1.0,
        None,
    )
    normalized_confusion_df = pd.DataFrame(normalized_confusion, index=label_names, columns=label_names)

    confusion_path = output_dir / f"{task_name}_confusion.csv"
    confusion_norm_path = output_dir / f"{task_name}_confusion_normalized.csv"
    confusion_png_path = output_dir / f"{task_name}_confusion.png"
    confusion_df.to_csv(confusion_path)
    normalized_confusion_df.to_csv(confusion_norm_path)
    render_confusion_matrix(
        confusion,
        label_names=label_names,
        title=task_name,
        output_path=confusion_png_path,
    )

    prediction_df = manifest_df.copy()
    prediction_df["task_name"] = task_name
    prediction_df["true_label"] = labels.astype(np.int64)
    prediction_df["true_label_name"] = [label_names_map[int(label)] for label in labels.tolist()]
    prediction_df["pred_label"] = np.asarray([index_to_label[int(pred)] for pred in predictions.tolist()], dtype=np.int64)
    prediction_df["pred_label_name"] = [
        label_names_map[int(index_to_label[int(pred)])] for pred in predictions.tolist()
    ]
    prediction_df["fold"] = manifest_df["subject"].map(subject_to_fold).astype(np.int64)

    summary = {
        "task_name": task_name,
        "num_groups": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_classes": int(len(index_to_label)),
        "label_names": label_names,
        "metrics": metrics,
        "fold_accuracies": [row["accuracy"] for row in fold_rows],
        "fold_balanced_accuracies": [row["balanced_accuracy"] for row in fold_rows],
        "fold_macro_f1": [row["macro_f1"] for row in fold_rows],
        "confusion_matrix": confusion.tolist(),
        "normalized_confusion_matrix": normalized_confusion.tolist(),
        "artifacts": {
            "confusion_csv": str(confusion_path),
            "confusion_normalized_csv": str(confusion_norm_path),
            "confusion_png": str(confusion_png_path),
        },
    }
    return summary, prediction_df, pd.DataFrame(fold_rows)


def build_report_markdown(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    fold_manifest: pd.DataFrame,
    probe_summary_df: pd.DataFrame,
    metadata_summary: dict,
    task_summaries: dict,
) -> str:
    def df_to_markdown_table(df: pd.DataFrame) -> list[str]:
        headers = [str(col) for col in df.columns.tolist()]
        align = ["---" for _ in headers]
        rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
        for _, row in df.iterrows():
            values = [str(row[col]) for col in df.columns.tolist()]
            rows.append("| " + " | ".join(values) + " |")
        return rows

    lines = [
        "# V31 Full Post-hoc K-Fold Report",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- output_dir: `{output_dir}`",
        f"- total_groups: `{metadata_summary['total_groups']}`",
        f"- total_subjects: `{metadata_summary['total_subjects']}`",
        f"- requested_splits: `{metadata_summary['requested_splits']}`",
        f"- resolved_splits: `{metadata_summary['resolved_splits']}`",
        f"- subject_stratification: `{metadata_summary['stratification_mode']}`",
        "",
        "## Fold Summary",
        "",
        *df_to_markdown_table(fold_manifest),
        "",
        "## Probe Metrics",
        "",
        *df_to_markdown_table(probe_summary_df),
        "",
        "## Task Artifacts",
        "",
    ]
    for task_name, task_summary in task_summaries.items():
        lines.extend(
            [
                f"### {task_name}",
                "",
                f"- accuracy: `{task_summary['metrics']['accuracy']:.4f}`",
                f"- balanced_accuracy: `{task_summary['metrics']['balanced_accuracy']:.4f}`",
                f"- macro_f1: `{task_summary['metrics']['macro_f1']:.4f}`",
                f"- confusion_csv: `{task_summary['artifacts']['confusion_csv']}`",
                f"- confusion_png: `{task_summary['artifacts']['confusion_png']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def analyze(
    checkpoint_path: str,
    data_roots: str | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    requested_splits: int = 5,
    output_dir: str | None = None,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})

    data_roots = data_roots or config.get("data_roots")
    if not data_roots:
        raise ValueError("data_roots must be provided or present in checkpoint config")

    seed = int(config.get("seed", 42))
    mode = str(config.get("mode", "x"))
    region = str(config.get("region", "mouth"))
    use_difference = bool(config.get("use_difference", True))
    signed_normalize = str(config.get("signed_normalize", "per_sample"))
    val_ratio = float(config.get("val_ratio", 0.2))
    group_size = int(config.get("group_size", 4))
    apply_deleted_filter = bool(config.get("apply_deleted_filter", True))
    early_branch_factorization = bool(config.get("early_branch_factorization", False))

    output_dir = Path(output_dir or checkpoint_path.parent / "kfold_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs, full_dataset = build_full_eval_dataset(
        data_roots=data_roots,
        mode=mode,
        region=region,
        use_difference=use_difference,
        signed_normalize=signed_normalize,
        val_ratio=val_ratio,
        seed=seed,
        group_size=group_size,
        apply_deleted_filter=apply_deleted_filter,
    )
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model, model_config = load_model_from_checkpoint(
        checkpoint_path,
        num_dataset_classes=len(specs),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    group_outputs = collect_group_representations(
        model,
        loader,
        device,
        max_batches=None,
        require_canonical_group_reps=early_branch_factorization,
    )
    branch_df = combine_group_representations(group_outputs)
    side_semantics_df = collect_group_side_semantics(model, loader, device)
    manifest_df = align_side_semantics(branch_df, side_semantics_df)

    manifest_df = manifest_df.sort_values(["dataset_name", "subject", "group_id"]).reset_index(drop=True)
    group_representations = group_outputs["group_representations"]
    sorted_order = manifest_df["group_id"].tolist()
    branch_position = {
        str(group_id): idx for idx, group_id in enumerate(group_representations["group_id"].astype(str).tolist())
    }
    row_indices = np.asarray([branch_position[group_id] for group_id in sorted_order], dtype=np.int64)

    full_rep_path = output_dir / "full_group_level_representations.npz"
    np.savez(
        full_rep_path,
        group_pooled_side_rep=group_representations["group_pooled_side_rep"][row_indices],
        group_pooled_free_rep=group_representations["group_pooled_free_rep"][row_indices],
        group_pooled_side_latent=group_representations["group_pooled_side_latent"][row_indices],
        group_pooled_free_latent=group_representations["group_pooled_free_latent"][row_indices],
        group_pooled_private_rep=group_representations["group_pooled_private_rep"][row_indices],
        side_label=manifest_df["side_label"].to_numpy(dtype=np.int64),
        dataset_label=manifest_df["dataset_label"].to_numpy(dtype=np.int64),
        group_id=manifest_df["group_id"].to_numpy(dtype=object),
        subject=manifest_df["subject"].to_numpy(dtype=object),
        dataset_name=manifest_df["dataset_name"].to_numpy(dtype=object),
    )
    full_side_semantics_path = output_dir / "full_group_side_semantics.csv"
    manifest_df.to_csv(full_side_semantics_path, index=False)

    stratification_mode, stratify_labels, resolved_splits = resolve_stratify_labels(
        manifest_df,
        requested_splits=requested_splits,
    )
    subject_to_fold, subject_fold_df = greedy_subject_stratified_folds(
        manifest_df,
        stratify_labels=stratify_labels,
        num_splits=resolved_splits,
        seed=seed,
    )
    subject_fold_path = output_dir / "subject_fold_assignments.csv"
    subject_fold_df.to_csv(subject_fold_path, index=False)

    fold_summary_rows = []
    for fold_idx in range(resolved_splits):
        fold_subjects = {subject for subject, assigned_fold in subject_to_fold.items() if assigned_fold == fold_idx}
        fold_manifest = manifest_df[manifest_df["subject"].isin(fold_subjects)].copy()
        fold_summary_rows.append(
            {
                "fold": int(fold_idx),
                "num_subjects": int(len(fold_subjects)),
                "num_groups": int(len(fold_manifest)),
                "side_counts": json.dumps(
                    {
                        SIDE_LABEL_NAMES.get(int(label), str(int(label))): int(count)
                        for label, count in fold_manifest["side_label"].value_counts().sort_index().items()
                    },
                    ensure_ascii=False,
                ),
                "dataset_counts": json.dumps(
                    {
                        str(specs[int(label)].dataset_name): int(count)
                        for label, count in fold_manifest["dataset_label"].value_counts().sort_index().items()
                    },
                    ensure_ascii=False,
                ),
            }
        )
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    fold_summary_path = output_dir / "fold_summary.csv"
    fold_summary_df.to_csv(fold_summary_path, index=False)

    task_specs = [
        ("side_from_side_rep", group_representations["group_pooled_side_rep"][row_indices], manifest_df["side_label"].to_numpy(dtype=np.int64), SIDE_LABEL_NAMES),
        ("side_from_free_rep", group_representations["group_pooled_free_rep"][row_indices], manifest_df["side_label"].to_numpy(dtype=np.int64), SIDE_LABEL_NAMES),
        (
            "dataset_from_side_rep",
            group_representations["group_pooled_side_rep"][row_indices],
            manifest_df["dataset_label"].to_numpy(dtype=np.int64),
            {idx: spec.dataset_name for idx, spec in enumerate(specs)},
        ),
        (
            "dataset_from_free_rep",
            group_representations["group_pooled_free_rep"][row_indices],
            manifest_df["dataset_label"].to_numpy(dtype=np.int64),
            {idx: spec.dataset_name for idx, spec in enumerate(specs)},
        ),
        (
            "dataset_from_private_rep",
            group_representations["group_pooled_private_rep"][row_indices],
            manifest_df["dataset_label"].to_numpy(dtype=np.int64),
            {idx: spec.dataset_name for idx, spec in enumerate(specs)},
        ),
        (
            "side_from_usage",
            manifest_df[[col for col in manifest_df.columns if col.startswith("usage_b")]].to_numpy(dtype=np.float32),
            manifest_df["side_label"].to_numpy(dtype=np.int64),
            SIDE_LABEL_NAMES,
        ),
        (
            "side_from_coeff",
            manifest_df[["side_coeff_mean"]].to_numpy(dtype=np.float32),
            manifest_df["side_label"].to_numpy(dtype=np.int64),
            SIDE_LABEL_NAMES,
        ),
        (
            "side_from_usage_coeff",
            manifest_df[
                [col for col in manifest_df.columns if col.startswith("usage_b")] + ["side_coeff_mean"]
            ].to_numpy(dtype=np.float32),
            manifest_df["side_label"].to_numpy(dtype=np.int64),
            SIDE_LABEL_NAMES,
        ),
        (
            "dataset_from_usage",
            manifest_df[[col for col in manifest_df.columns if col.startswith("usage_b")]].to_numpy(dtype=np.float32),
            manifest_df["dataset_label"].to_numpy(dtype=np.int64),
            {idx: spec.dataset_name for idx, spec in enumerate(specs)},
        ),
        (
            "dataset_from_coeff",
            manifest_df[["side_coeff_mean"]].to_numpy(dtype=np.float32),
            manifest_df["dataset_label"].to_numpy(dtype=np.int64),
            {idx: spec.dataset_name for idx, spec in enumerate(specs)},
        ),
        (
            "dataset_from_usage_coeff",
            manifest_df[
                [col for col in manifest_df.columns if col.startswith("usage_b")] + ["side_coeff_mean"]
            ].to_numpy(dtype=np.float32),
            manifest_df["dataset_label"].to_numpy(dtype=np.int64),
            {idx: spec.dataset_name for idx, spec in enumerate(specs)},
        ),
    ]

    task_summaries = {}
    prediction_frames = []
    fold_metric_frames = []
    summary_rows = []
    for task_name, features, labels, label_name_map in task_specs:
        summary, prediction_df, fold_metrics_df = evaluate_probe_task(
            task_name=task_name,
            features=features,
            labels=labels,
            label_names_map=label_name_map,
            manifest_df=manifest_df[
                ["group_id", "subject", "dataset_name", "side_label", "side_label_name", "dataset_label"]
            ].copy(),
            subject_to_fold=subject_to_fold,
            num_splits=resolved_splits,
            seed=seed,
            output_dir=output_dir,
        )
        task_summaries[task_name] = summary
        prediction_frames.append(prediction_df)
        fold_metric_frames.append(fold_metrics_df)
        summary_rows.append(
            {
                "task_name": task_name,
                "num_groups": summary["num_groups"],
                "num_features": summary["num_features"],
                "num_classes": summary["num_classes"],
                "accuracy": summary["metrics"]["accuracy"],
                "balanced_accuracy": summary["metrics"]["balanced_accuracy"],
                "macro_f1": summary["metrics"]["macro_f1"],
            }
        )

    probe_summary_df = pd.DataFrame(summary_rows).sort_values("task_name").reset_index(drop=True)
    probe_summary_path = output_dir / "probe_summary.csv"
    probe_summary_df.to_csv(probe_summary_path, index=False)

    predictions_df = pd.concat(prediction_frames, axis=0, ignore_index=True)
    predictions_path = output_dir / "probe_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    fold_metrics_df = pd.concat(fold_metric_frames, axis=0, ignore_index=True)
    fold_metrics_path = output_dir / "probe_fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    metadata_summary = {
        "checkpoint_path": str(checkpoint_path),
        "data_roots": data_roots,
        "mode": model_config.get("mode", mode),
        "region": model_config.get("region", region),
        "total_groups": int(len(manifest_df)),
        "total_subjects": int(manifest_df["subject"].nunique()),
        "requested_splits": int(requested_splits),
        "resolved_splits": int(resolved_splits),
        "stratification_mode": stratification_mode,
        "side_counts": {
            SIDE_LABEL_NAMES.get(int(label), str(int(label))): int(count)
            for label, count in manifest_df["side_label"].value_counts().sort_index().items()
        },
        "dataset_counts": {
            specs[int(label)].dataset_name: int(count)
            for label, count in manifest_df["dataset_label"].value_counts().sort_index().items()
        },
        "artifacts": {
            "full_group_level_representations": str(full_rep_path),
            "full_group_side_semantics": str(full_side_semantics_path),
            "subject_fold_assignments": str(subject_fold_path),
            "fold_summary": str(fold_summary_path),
            "probe_summary": str(probe_summary_path),
            "probe_predictions": str(predictions_path),
            "probe_fold_metrics": str(fold_metrics_path),
        },
    }

    summary = {
        "metadata": metadata_summary,
        "tasks": task_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    report_text = build_report_markdown(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        fold_manifest=fold_summary_df,
        probe_summary_df=probe_summary_df,
        metadata_summary=metadata_summary,
        task_summaries=task_summaries,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_text)

    print(f"Saved combined group representations: {full_rep_path}")
    print(f"Saved combined side semantics: {full_side_semantics_path}")
    print(f"Saved subject fold assignments: {subject_fold_path}")
    print(f"Saved fold summary: {fold_summary_path}")
    print(f"Saved probe summary: {probe_summary_path}")
    print(f"Saved probe predictions: {predictions_path}")
    print(f"Saved probe fold metrics: {fold_metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    fire.Fire(analyze)
