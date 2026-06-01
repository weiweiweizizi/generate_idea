from __future__ import annotations

import torch
from tqdm.auto import tqdm

from .losses import step_model
from .probe import compute_probe_stats, summarize_probe_stats


def tensor_memory_mib(tensor: torch.Tensor) -> float:
    """Estimate tensor memory footprint in MiB."""

    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def run_batch_memory_validation(model, loader, device, optimizer, loss_weights):
    """Run one forward/backward smoke pass and print batch memory info."""

    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Training loader is empty; cannot validate batch memory") from exc

    x = batch["images"]
    valid_mask = batch["valid_mask"]
    padding_mask = batch["padding_mask"]
    print(
        "[memory-check] "
        f"images.shape={tuple(x.shape)} "
        f"valid_mask.shape={tuple(valid_mask.shape)} "
        f"padding_mask.shape={tuple(padding_mask.shape)} "
        f"dtype={x.dtype} "
        f"input_mib={tensor_memory_mib(x):.2f}"
    )

    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        loss, loss_metrics, probe_outputs = step_model(model, batch, device, loss_weights)
        probe_metrics = summarize_probe_stats(compute_probe_stats(probe_outputs, batch, device))
        loss.backward()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "Batch memory validation failed with OOM. "
                "Try region=mouth, smaller group_size, or smaller batch_size."
            ) from exc
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[memory-check] smoke_loss_metrics={loss_metrics}")
    print(f"[memory-check] smoke_probe_metrics={probe_metrics}")
    if device == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[memory-check] peak_cuda_mib={peak_mib:.2f}")


def run_epoch(model, loader, device, optimizer, loss_weights, train: bool):
    model.train(train)
    total_loss_metrics = {}
    total_probe_stats = {
        "side_correct": 0.0,
        "side_total": 0.0,
        "group_side_correct": 0.0,
        "group_side_total": 0.0,
    }

    for batch in tqdm(loader, leave=False):
        if train:
            optimizer.zero_grad()

        loss, loss_metrics, probe_outputs = step_model(model, batch, device, loss_weights)
        probe_stats = compute_probe_stats(probe_outputs, batch, device)

        if train:
            loss.backward()
            optimizer.step()
            if hasattr(model, "project_basis_constraints_"):
                model.project_basis_constraints_()

        for key, value in loss_metrics.items():
            total_loss_metrics[key] = total_loss_metrics.get(key, 0.0) + value
        for key, value in probe_stats.items():
            total_probe_stats[key] = total_probe_stats.get(key, 0.0) + value

    denom = max(len(loader), 1)
    mean_loss_metrics = {key: value / denom for key, value in total_loss_metrics.items()}
    probe_metrics = summarize_probe_stats(total_probe_stats)
    return mean_loss_metrics, probe_metrics
