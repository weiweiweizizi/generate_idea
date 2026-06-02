from __future__ import annotations

import torch
from tqdm.auto import tqdm

from disentangleNet.losses import step_model


def tensor_memory_mib(tensor) -> float:
    """Estimate tensor memory usage in MiB."""

    if tensor is None:
        return 0.0
    return float(tensor.numel() * tensor.element_size()) / float(1024 ** 2)


def run_batch_memory_validation(
    model,
    loader,
    device,
    optimizer,
    loss_weights,
):
    """
    Run a single-batch memory validation pass.

    TODO(recovery): extend this to report the original historical diagnostics
    once the complete training scaffold is back. For now it only verifies that
    one real batch can complete forward/backward/step on the recovered path.
    """

    was_training = model.training
    model.train()
    batch = next(iter(loader))

    optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loss, metrics = step_model(model, batch, device, loss_weights)
    loss.backward()
    optimizer.step()

    x = batch["images"].to(device)
    input_mib = tensor_memory_mib(x)
    peak_mib = (
        float(torch.cuda.max_memory_allocated()) / float(1024 ** 2)
        if device.startswith("cuda") and torch.cuda.is_available()
        else input_mib
    )
    print(
        "[memory-check] "
        f"loss={metrics['loss']:.5f} "
        f"input_mib={input_mib:.2f} "
        f"peak_mib={peak_mib:.2f}"
    )
    model.train(was_training)
    return {
        "loss": metrics["loss"],
        "input_mib": input_mib,
        "peak_mib": peak_mib,
    }


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    loss_weights,
    train,
):
    """
    Run one training or validation epoch.

    TODO(recovery): if future PhaseAB follow-up requires richer probes or
    per-step artifact dumps, add them here instead of overloading `step_model`.
    """

    totals: dict[str, float] = {}
    model.train(train)

    for batch in tqdm(loader, leave=False):
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = step_model(model, batch, device, loss_weights)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _, metrics = step_model(model, batch, device, loss_weights)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)

    denom = max(len(loader), 1)
    return {key: value / denom for key, value in totals.items()}


__all__ = ["tensor_memory_mib", "run_batch_memory_validation", "run_epoch"]
