from __future__ import annotations

import torch


def _sequence_labels(labels: torch.Tensor, batch_size: int) -> torch.Tensor:
    if labels.ndim == 1:
        if labels.shape[0] != batch_size:
            raise ValueError(
                f"Expected sequence labels with batch size {batch_size}, got {tuple(labels.shape)}"
            )
        return labels
    if labels.ndim == 2:
        if labels.shape[0] != batch_size:
            raise ValueError(
                f"Expected sequence labels with batch size {batch_size}, got {tuple(labels.shape)}"
            )
        return labels[:, 0]
    raise ValueError(f"Unsupported label shape for probe metrics: {tuple(labels.shape)}")


def compute_probe_stats(outputs: dict, batch: dict, device: str) -> dict[str, float]:
    side_logits = outputs["side_logits"]
    group_side_logits = outputs["group_side_logits"]
    valid_mask = batch["valid_mask"].to(device)
    side_labels = batch["side_label"].to(device)

    stats = {
        "side_correct": 0.0,
        "side_total": 0.0,
        "group_side_correct": 0.0,
        "group_side_total": 0.0,
    }

    if side_logits is not None:
        side_logits = side_logits.detach()
        if side_logits.ndim == 3:
            expanded_labels = side_labels.unsqueeze(1).expand(-1, side_logits.shape[1])
            valid_positions = valid_mask.bool()
            if valid_positions.any():
                predictions = side_logits.argmax(dim=-1)
                correct = (predictions[valid_positions] == expanded_labels[valid_positions]).sum()
                stats["side_correct"] = float(correct.cpu())
                stats["side_total"] = float(valid_positions.sum().cpu())
        elif side_logits.ndim == 2:
            predictions = side_logits.argmax(dim=-1)
            correct = (predictions == side_labels).sum()
            stats["side_correct"] = float(correct.cpu())
            stats["side_total"] = float(side_labels.shape[0])
        else:
            raise ValueError(f"Unsupported side_logits shape for probe metrics: {tuple(side_logits.shape)}")

    if group_side_logits is not None:
        group_side_logits = group_side_logits.detach()
        group_valid_mask = valid_mask.any(dim=1)
        if group_valid_mask.any():
            group_labels = _sequence_labels(side_labels, group_side_logits.shape[0])
            predictions = group_side_logits.argmax(dim=-1)
            correct = (predictions[group_valid_mask] == group_labels[group_valid_mask]).sum()
            stats["group_side_correct"] = float(correct.cpu())
            stats["group_side_total"] = float(group_valid_mask.sum().cpu())

    return stats


def summarize_probe_stats(stats: dict[str, float]) -> dict[str, float]:
    side_total = stats.get("side_total", 0.0)
    group_side_total = stats.get("group_side_total", 0.0)
    metrics = {
        "side_total": side_total,
        "group_side_total": group_side_total,
    }
    if side_total > 0:
        metrics["side_acc"] = stats["side_correct"] / side_total
    if group_side_total > 0:
        metrics["group_side_acc"] = stats["group_side_correct"] / group_side_total
    return metrics
