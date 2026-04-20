#!/usr/bin/env python

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import warnings

import fire
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.lq.data import DatasetSpec, FacialMotionSequenceDataset, subject_split
    from scripts.lq.model.network import DistNet
else:
    from .data import DatasetSpec, FacialMotionSequenceDataset, subject_split
    from .model.network import DistNet

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST

SKLEARN_LOGISTIC_REGRESSION = None
SKLEARN_IMPORT_ERROR = None
SKLEARN_IMPORT_ATTEMPTED = False


def build_specs(data_roots: str) -> list[DatasetSpec]:
    roots = [Path(root).expanduser() for root in data_roots.split(",") if root.strip()]
    return [
        DatasetSpec(root=root, dataset_label=idx, dataset_name=root.name)
        for idx, root in enumerate(roots)
    ]


def build_eval_dataset(
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
):
    datasets = []

    for spec in specs:
        train_subjects, val_subjects = subject_split(spec, val_ratio=val_ratio, seed=seed)
        subjects = train_subjects if split == "train" else val_subjects

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


def basis_to_rgb_image(basis: np.ndarray, vmax: float) -> Image.Image:
    """Render one basis matrix into a simple RdBu-style RGB image."""

    clipped = np.clip(basis / vmax, -1.0, 1.0)
    positive = np.clip(clipped, 0.0, 1.0)
    negative = np.clip(-clipped, 0.0, 1.0)

    rgb = np.zeros((*basis.shape, 3), dtype=np.uint8)
    # Positive values -> red, negative values -> blue, near zero -> white.
    rgb[..., 0] = (255 * (1.0 - negative)).astype(np.uint8)
    rgb[..., 1] = (255 * (1.0 - np.maximum(positive, negative))).astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - positive)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def plot_basis_grid(basis: np.ndarray, levels: tuple[int, ...], output_path: Path) -> None:
    """Save all action bases as a compact heatmap grid without matplotlib."""

    total = basis.shape[0]
    cols = min(4, total)
    rows = int(np.ceil(total / cols))
    cell_size = 220
    pad = 16
    title_h = 28
    canvas = Image.new(
        "RGB",
        (cols * (cell_size + pad) + pad, rows * (cell_size + title_h + pad) + pad),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    vmax = float(np.max(np.abs(basis)))
    vmax = max(vmax, 1e-6)

    labels = []
    for level_idx, level_size in enumerate(levels):
        for local_idx in range(level_size):
            labels.append(f"L{level_idx + 1}-{local_idx}")

    for idx in range(total):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (cell_size + pad)
        y0 = pad + row * (cell_size + title_h + pad)

        basis_img = basis_to_rgb_image(basis[idx], vmax=vmax).resize(
            (cell_size, cell_size), RESAMPLE_NEAREST
        )
        canvas.paste(basis_img, (x0, y0 + title_h))
        draw.text((x0, y0), labels[idx], fill=(0, 0, 0))

    canvas.save(output_path)


def masked_mean_per_sequence(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool each grouped sequence with a boolean valid-frame mask."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)

    denom = mask.sum(dim=1).clamp_min(1.0)
    while denom.ndim < values.ndim - 1:
        denom = denom.unsqueeze(-1)

    return (values * expanded_mask).sum(dim=1) / denom


def get_logistic_regression_cls():
    """Try to import sklearn LogisticRegression once and cache the result."""

    global SKLEARN_LOGISTIC_REGRESSION, SKLEARN_IMPORT_ERROR, SKLEARN_IMPORT_ATTEMPTED

    if SKLEARN_IMPORT_ATTEMPTED:
        return SKLEARN_LOGISTIC_REGRESSION, SKLEARN_IMPORT_ERROR

    SKLEARN_IMPORT_ATTEMPTED = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                from sklearn.linear_model import LogisticRegression

        SKLEARN_LOGISTIC_REGRESSION = LogisticRegression
    except Exception as exc:  # pragma: no cover - depends on local env
        SKLEARN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    return SKLEARN_LOGISTIC_REGRESSION, SKLEARN_IMPORT_ERROR


def fit_torch_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
) -> float:
    """Fallback linear probe used when sklearn is unavailable."""

    torch.manual_seed(seed)

    x_train = torch.from_numpy(train_x).float()
    y_train = torch.from_numpy(train_y).long()
    x_test = torch.from_numpy(test_x).float()
    y_test = torch.from_numpy(test_y).long()

    classifier = torch.nn.Linear(x_train.shape[1], int(train_y.max()) + 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = classifier(x_test).argmax(dim=1)
    return float((predictions == y_test).float().mean().item())


def fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    test_ratio: float = 0.2,
) -> dict:
    """Fit a lightweight linear probe on held-out groups."""

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    result = {
        "accuracy": None,
        "backend": None,
        "num_samples": int(labels.shape[0]),
        "num_features": int(features.shape[1]) if features.ndim == 2 else 0,
        "num_classes": int(np.unique(labels).size) if labels.size else 0,
        "train_size": 0,
        "test_size": 0,
        "error": None,
    }

    if features.ndim != 2:
        result["error"] = f"expected 2D features, got shape {tuple(features.shape)}"
        return result
    if features.shape[0] != labels.shape[0]:
        result["error"] = "feature/label sample count mismatch"
        return result
    if features.shape[0] < 2:
        result["error"] = "need at least 2 samples"
        return result
    if features.shape[1] == 0:
        result["error"] = "probe features are empty"
        return result

    classes, encoded = np.unique(labels, return_inverse=True)
    if classes.size < 2:
        result["error"] = "need at least 2 classes"
        return result

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for class_idx in range(classes.size):
        class_members = np.flatnonzero(encoded == class_idx)
        if class_members.size < 2:
            result["error"] = f"class {int(classes[class_idx])} has fewer than 2 samples"
            return result
        shuffled = class_members.copy()
        rng.shuffle(shuffled)
        class_test = max(1, int(round(class_members.size * test_ratio)))
        class_test = min(class_test, class_members.size - 1)
        test_indices.extend(shuffled[:class_test].tolist())
        train_indices.extend(shuffled[class_test:].tolist())

    if not train_indices or not test_indices:
        result["error"] = "failed to create non-empty train/test split"
        return result

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    train_x = features[train_indices]
    test_x = features[test_indices]
    train_y = encoded[train_indices]
    test_y = encoded[test_indices]

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    result["train_size"] = int(train_x.shape[0])
    result["test_size"] = int(test_x.shape[0])

    logistic_regression_cls, import_error = get_logistic_regression_cls()
    if logistic_regression_cls is not None:
        try:
            classifier = logistic_regression_cls(
                max_iter=1000,
                random_state=seed,
            )
            classifier.fit(train_x, train_y)
            accuracy = float((classifier.predict(test_x) == test_y).mean())
            result["accuracy"] = accuracy
            result["backend"] = "sklearn_logistic_regression"
            return result
        except Exception as exc:  # pragma: no cover - depends on local env
            import_error = f"{type(exc).__name__}: {exc}"

    result["accuracy"] = fit_torch_linear_probe(train_x, train_y, test_x, test_y, seed=seed)
    result["backend"] = "torch_linear_probe_fallback"
    result["error"] = import_error
    return result


def collect_group_representations(
    model: DistNet,
    loader: DataLoader,
    device: str,
    max_batches: int | None,
) -> dict:
    """Collect round-1 analysis artifacts and code usage in one pass."""

    level_counts = [torch.zeros(level, dtype=torch.long) for level in model.levels]
    total_valid_frames = 0
    side_usage_sum = None
    free_usage_sum = None
    side_recon_l1_sum = 0.0
    free_recon_l1_sum = 0.0
    side_reps = []
    free_reps = []
    private_reps = []
    side_labels = []
    dataset_labels = []
    group_ids = []
    seen_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["images"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            outputs = model(x)
            decoded_indices = outputs["decoded_indices"]

            for level_idx, frame_indices in enumerate(decoded_indices):
                flat_indices = frame_indices[valid_mask].reshape(-1).cpu()
                if flat_indices.numel() == 0:
                    continue
                bincount = torch.bincount(flat_indices, minlength=model.levels[level_idx])
                level_counts[level_idx] += bincount

            valid_frames = int(valid_mask.sum().item())
            total_valid_frames += valid_frames

            side_recon_l1 = outputs["shared_side_reconstruction"].abs().mean(dim=(2, 3, 4))
            free_recon_l1 = outputs["shared_free_reconstruction"].abs().mean(dim=(2, 3, 4))
            side_recon_l1_sum += float(side_recon_l1[valid_mask].sum().item())
            free_recon_l1_sum += float(free_recon_l1[valid_mask].sum().item())

            side_path_usage = outputs["side_path_usage"]
            free_path_usage = outputs["free_path_usage"]
            if side_path_usage.ndim == 3:
                masked_side_usage = (
                    side_path_usage * valid_mask.unsqueeze(-1).to(side_path_usage.dtype)
                ).sum(dim=(0, 1))
                if side_usage_sum is None:
                    side_usage_sum = torch.zeros_like(masked_side_usage)
                side_usage_sum += masked_side_usage.cpu()
            if free_path_usage.ndim == 3:
                masked_free_usage = (
                    free_path_usage * valid_mask.unsqueeze(-1).to(free_path_usage.dtype)
                ).sum(dim=(0, 1))
                if free_usage_sum is None:
                    free_usage_sum = torch.zeros_like(masked_free_usage)
                free_usage_sum += masked_free_usage.cpu()

            group_valid_mask = valid_mask.any(dim=1)
            if group_valid_mask.any():
                group_side_rep = masked_mean_per_sequence(
                    outputs["side_path_representation"],
                    valid_mask,
                )
                group_free_rep = masked_mean_per_sequence(
                    outputs["free_path_representation"],
                    valid_mask,
                )
                group_private_rep = masked_mean_per_sequence(outputs["private_z"], valid_mask)

                selected = group_valid_mask.cpu().numpy().astype(bool)
                side_reps.append(group_side_rep[group_valid_mask].cpu().numpy().astype(np.float32))
                free_reps.append(group_free_rep[group_valid_mask].cpu().numpy().astype(np.float32))
                private_reps.append(
                    group_private_rep[group_valid_mask].cpu().numpy().astype(np.float32)
                )
                side_labels.append(
                    batch["side_label"][group_valid_mask.cpu()].cpu().numpy().astype(np.int64)
                )
                dataset_labels.append(
                    batch["dataset_label"][group_valid_mask.cpu()].cpu().numpy().astype(np.int64)
                )
                group_ids.extend(
                    [group_id for group_id, keep in zip(batch["group_id"], selected) if keep]
                )

            seen_batches += 1
            if max_batches is not None and seen_batches >= max_batches:
                break

    usage_summary = {"total_valid_frames": total_valid_frames, "levels": []}
    for level_idx, counts in enumerate(level_counts):
        counts_list = counts.tolist()
        total = max(int(sum(counts_list)), 1)
        usage_summary["levels"].append(
            {
                "level_index": level_idx,
                "size": model.levels[level_idx],
                "counts": counts_list,
                "fractions": [count / total for count in counts_list],
            }
        )

    def concat_feature_list(feature_list: list[np.ndarray]) -> np.ndarray:
        if not feature_list:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(feature_list, axis=0)

    def concat_label_list(label_list: list[np.ndarray]) -> np.ndarray:
        if not label_list:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(label_list, axis=0)

    side_usage_mean = (
        (side_usage_sum / max(total_valid_frames, 1)).numpy().astype(np.float32)
        if side_usage_sum is not None
        else np.zeros((0,), dtype=np.float32)
    )
    free_usage_mean = (
        (free_usage_sum / max(total_valid_frames, 1)).numpy().astype(np.float32)
        if free_usage_sum is not None
        else np.zeros((0,), dtype=np.float32)
    )

    return {
        "code_usage": usage_summary,
        "mean_side_path_usage": float(side_usage_mean.mean()) if side_usage_mean.size else 0.0,
        "mean_free_path_usage": float(free_usage_mean.mean()) if free_usage_mean.size else 0.0,
        "mean_side_recon_l1": float(side_recon_l1_sum / max(total_valid_frames, 1)),
        "mean_free_recon_l1": float(free_recon_l1_sum / max(total_valid_frames, 1)),
        "mean_side_path_usage_per_basis": side_usage_mean.tolist(),
        "mean_free_path_usage_per_basis": free_usage_mean.tolist(),
        "group_representations": {
            "group_pooled_side_rep": concat_feature_list(side_reps),
            "group_pooled_free_rep": concat_feature_list(free_reps),
            "group_pooled_private_rep": concat_feature_list(private_reps),
            "side_label": concat_label_list(side_labels),
            "dataset_label": concat_label_list(dataset_labels),
            "group_id": np.asarray(group_ids, dtype=object),
        },
    }


def build_probe_summary(group_representations: dict, seed: int) -> tuple[dict, dict]:
    """Fit post-hoc side and dataset probes from pooled group representations."""

    side_rep = group_representations["group_pooled_side_rep"]
    free_rep = group_representations["group_pooled_free_rep"]
    private_rep = group_representations["group_pooled_private_rep"]
    side_labels = group_representations["side_label"]
    dataset_labels = group_representations["dataset_label"]

    side_from_side = fit_linear_probe(side_rep, side_labels, seed=seed)
    side_from_free = fit_linear_probe(free_rep, side_labels, seed=seed)
    dataset_from_side = fit_linear_probe(side_rep, dataset_labels, seed=seed)
    dataset_from_free = fit_linear_probe(free_rep, dataset_labels, seed=seed)
    dataset_from_private = fit_linear_probe(private_rep, dataset_labels, seed=seed)

    side_probe = {
        "side_from_side_rep_acc": side_from_side["accuracy"],
        "side_from_free_rep_acc": side_from_free["accuracy"],
        "side_from_side_rep_backend": side_from_side["backend"],
        "side_from_free_rep_backend": side_from_free["backend"],
        "side_from_side_rep_error": side_from_side["error"],
        "side_from_free_rep_error": side_from_free["error"],
    }
    dataset_probe = {
        "dataset_from_side_rep_acc": dataset_from_side["accuracy"],
        "dataset_from_free_rep_acc": dataset_from_free["accuracy"],
        "dataset_from_private_rep_acc": dataset_from_private["accuracy"],
        "dataset_from_side_rep_backend": dataset_from_side["backend"],
        "dataset_from_free_rep_backend": dataset_from_free["backend"],
        "dataset_from_private_rep_backend": dataset_from_private["backend"],
        "dataset_from_side_rep_error": dataset_from_side["error"],
        "dataset_from_free_rep_error": dataset_from_free["error"],
        "dataset_from_private_rep_error": dataset_from_private["error"],
    }
    return side_probe, dataset_probe


def analyze(
    checkpoint_path: str,
    data_roots: str | None = None,
    split: str = "val",
    batch_size: int = 64,
    max_batches: int | None = None,
    num_workers: int = 0,
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

    mode = config.get("mode", "x")
    region = config.get("region", "mouth")
    use_difference = config.get("use_difference", True)
    signed_normalize = config.get("signed_normalize", "per_sample")
    val_ratio = float(config.get("val_ratio", 0.2))
    seed = int(config.get("seed", 42))
    group_size = int(config.get("group_size", 4))
    apply_deleted_filter = bool(config.get("apply_deleted_filter", True))
    basis_size = int(config.get("basis_size", 119))
    hidden_dim = int(config.get("hidden_dim", 32))
    pool_size = int(config.get("pool_size", 1))
    shared_dim = config.get("shared_dim")
    if shared_dim is not None:
        shared_dim = int(shared_dim)
    private_dim = int(config.get("private_dim", 32))
    private_decoder_hidden_dim = config.get("private_decoder_hidden_dim")
    if private_decoder_hidden_dim is not None:
        private_decoder_hidden_dim = int(private_decoder_hidden_dim)
    levels = tuple(int(v) for v in str(config.get("levels", "2,3,6")).split(","))
    use_dataset_aux = bool(config.get("use_dataset_aux", False))
    side_semantic_enabled = bool(config.get("side_semantic_enabled", False))
    side_basis_count = int(config.get("side_basis_count", 0))

    output_dir = Path(output_dir or checkpoint_path.parent / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(data_roots)
    dataset = build_eval_dataset(
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistNet(
        levels=levels,
        basis_size=basis_size,
        hidden_dim=hidden_dim,
        pool_size=pool_size,
        shared_dim=shared_dim,
        private_dim=private_dim,
        private_decoder_hidden_dim=private_decoder_hidden_dim,
        num_side_classes=3,
        num_dataset_classes=len(specs),
        private_residual_weight=float(config.get("private_residual_weight", 0.25)),
        private_residual_max_l1=config.get("private_residual_max_l1"),
        shared_basis_soft_mixing=bool(config.get("shared_basis_soft_mixing", False)),
        shared_basis_anchor_bias=float(config.get("shared_basis_anchor_bias", 1.0)),
        shared_basis_topk=config.get("shared_basis_topk"),
        grl_lambda=float(config.get("grl_lambda", 1.0)),
        use_dataset_aux=use_dataset_aux,
        side_semantic_enabled=side_semantic_enabled,
        side_basis_count=side_basis_count,
        action_basis_init_path=None,
        lq_commitment_loss_weight=float(config.get("lq_commitment_loss_weight", 0.1)),
        lq_quantization_loss_weight=float(config.get("lq_quantization_loss_weight", 0.1)),
        lq_optimize_values=bool(config.get("lq_optimize_values", True)),
        quantizer_type=config.get("quantizer_type", "latent_quantize"),
        fsq_preserve_symmetry=bool(config.get("fsq_preserve_symmetry", True)),
        basis_orthogonalization=config.get("basis_orthogonalization", "normalize"),
    ).to(device)
    load_result = model.load_state_dict(ckpt["model"], strict=False)

    basis = model.get_structured_basis().detach().cpu().numpy()
    basis_path = output_dir / "basis_bank.npy"
    np.save(basis_path, basis)

    basis_plot_path = output_dir / "basis_bank_heatmap.png"
    plot_basis_grid(basis, levels, basis_plot_path)

    side_basis = model.get_side_basis().detach().cpu().numpy()
    side_basis_path = output_dir / "side_basis_bank.npy"
    np.save(side_basis_path, side_basis)
    side_basis_plot_path = None
    if side_basis.shape[0] > 0:
        side_basis_plot_path = output_dir / "side_basis_bank_heatmap.png"
        plot_basis_grid(side_basis, (side_basis.shape[0],), side_basis_plot_path)

    analysis_outputs = collect_group_representations(
        model,
        loader,
        device,
        max_batches=max_batches,
    )
    group_representations = analysis_outputs.pop("group_representations")
    group_artifact_path = output_dir / "group_level_representations.npz"
    np.savez(
        group_artifact_path,
        group_pooled_side_rep=group_representations["group_pooled_side_rep"],
        group_pooled_free_rep=group_representations["group_pooled_free_rep"],
        group_pooled_private_rep=group_representations["group_pooled_private_rep"],
        side_label=group_representations["side_label"],
        dataset_label=group_representations["dataset_label"],
        group_id=group_representations["group_id"],
    )
    side_probe, dataset_probe = build_probe_summary(group_representations, seed=seed)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "analysis_split": split,
        "basis_shape": list(basis.shape),
        "levels": list(levels),
        "side_basis_shape": list(side_basis.shape),
        "train_metrics": ckpt.get("train_metrics"),
        "val_metrics": ckpt.get("val_metrics"),
        "model_load": {
            "strict": False,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "code_usage": analysis_outputs["code_usage"],
        "mean_side_path_usage": analysis_outputs["mean_side_path_usage"],
        "mean_free_path_usage": analysis_outputs["mean_free_path_usage"],
        "mean_side_recon_l1": analysis_outputs["mean_side_recon_l1"],
        "mean_free_recon_l1": analysis_outputs["mean_free_recon_l1"],
        "mean_side_path_usage_per_basis": analysis_outputs["mean_side_path_usage_per_basis"],
        "mean_free_path_usage_per_basis": analysis_outputs["mean_free_path_usage_per_basis"],
        "side_probe": side_probe,
        "dataset_probe": dataset_probe,
        "artifacts": {
            "basis_bank": str(basis_path),
            "basis_bank_heatmap": str(basis_plot_path),
            "side_basis_bank": str(side_basis_path),
            "side_basis_bank_heatmap": str(side_basis_plot_path)
            if side_basis_plot_path is not None
            else None,
            "group_level_representations": str(group_artifact_path),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved basis bank: {basis_path}")
    print(f"Saved basis heatmap: {basis_plot_path}")
    if side_basis_plot_path is not None:
        print(f"Saved side basis heatmap: {side_basis_plot_path}")
    print(f"Saved group representations: {group_artifact_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    fire.Fire(analyze)
