#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
import sys

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


def summarize_code_usage(
    model: DistNet,
    loader: DataLoader,
    device: str,
    max_batches: int | None,
) -> dict:
    """Collect code usage counts on valid frames only."""

    level_counts = [torch.zeros(level, dtype=torch.long) for level in model.levels]
    total_valid_frames = 0
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

            total_valid_frames += int(valid_mask.sum().item())
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
    return usage_summary


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
        action_basis_init_path=None,
        lq_commitment_loss_weight=float(config.get("lq_commitment_loss_weight", 0.1)),
        lq_quantization_loss_weight=float(config.get("lq_quantization_loss_weight", 0.1)),
        lq_optimize_values=bool(config.get("lq_optimize_values", True)),
        quantizer_type=config.get("quantizer_type", "latent_quantize"),
        fsq_preserve_symmetry=bool(config.get("fsq_preserve_symmetry", True)),
        basis_orthogonalization=config.get("basis_orthogonalization", "normalize"),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    basis = model.get_structured_basis().detach().cpu().numpy()
    basis_path = output_dir / "basis_bank.npy"
    np.save(basis_path, basis)

    basis_plot_path = output_dir / "basis_bank_heatmap.png"
    plot_basis_grid(basis, levels, basis_plot_path)

    usage_summary = summarize_code_usage(model, loader, device, max_batches=max_batches)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "analysis_split": split,
        "basis_shape": list(basis.shape),
        "levels": list(levels),
        "train_metrics": ckpt.get("train_metrics"),
        "val_metrics": ckpt.get("val_metrics"),
        "code_usage": usage_summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved basis bank: {basis_path}")
    print(f"Saved basis heatmap: {basis_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    fire.Fire(analyze)
