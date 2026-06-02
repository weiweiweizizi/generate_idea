from __future__ import annotations

import torch


def flatten_sequence_input(x):
    if x.ndim == 5:
        batch_size, time_steps = x.shape[:2]
        x = x.reshape(batch_size * time_steps, *x.shape[2:])
        return x, (batch_size, time_steps)
    return x, None


def flatten_sequence_labels(labels, sequence_shape):
    if labels is None or sequence_shape is None:
        return labels
    batch_size, time_steps = sequence_shape
    return labels.reshape(batch_size * time_steps) if labels.ndim > 1 else labels


def reshape_sequence_tensor(tensor, sequence_shape):
    if tensor is None or sequence_shape is None:
        return tensor
    batch_size, time_steps = sequence_shape
    if tensor.ndim == 1:
        return tensor.reshape(batch_size, time_steps)
    return tensor.reshape(batch_size, time_steps, *tensor.shape[1:])


def mean_pool_sequence_tensor(tensor, sequence_shape, mask=None):
    if tensor is None or sequence_shape is None:
        return None
    batch_size, time_steps = sequence_shape
    if mask is not None:
        mask = mask.to(device=tensor.device, dtype=tensor.dtype)
        if tensor.ndim == 1:
            tensor = tensor.reshape(batch_size, time_steps)
            return (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        tensor = tensor.reshape(batch_size, time_steps, -1)
        expanded_mask = mask.unsqueeze(-1)
        return (tensor * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp_min(1.0)
    if tensor.ndim == 1:
        return tensor.reshape(batch_size, time_steps).mean(dim=1)
    return tensor.reshape(batch_size, time_steps, -1).mean(dim=1)


__all__ = [
    "flatten_sequence_input",
    "flatten_sequence_labels",
    "mean_pool_sequence_tensor",
    "reshape_sequence_tensor",
]
