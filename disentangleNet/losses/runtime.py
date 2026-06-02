from __future__ import annotations

import torch
import torch.nn.functional as F

from .laplacian import matrix_laplacian_loss


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked positions while keeping gradients well-defined."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_mean_per_sequence(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool each grouped sample independently using a valid-frame mask."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)

    denom = mask.sum(dim=1).clamp_min(1.0)
    while denom.ndim < values.ndim - 1:
        denom = denom.unsqueeze(-1)

    return (values * expanded_mask).sum(dim=1) / denom


def side_private_orthogonality_loss(
    side_residual: torch.Tensor,
    private_residual: torch.Tensor,
) -> torch.Tensor:
    """Penalize alignment between side residual and private residual."""

    side_flat = F.normalize(side_residual.reshape(side_residual.shape[0], -1), dim=1)
    private_flat = F.normalize(private_residual.reshape(private_residual.shape[0], -1), dim=1)
    return (side_flat * private_flat).sum(dim=1).abs().mean()


def _get_optional_metric(outputs: dict, key: str) -> torch.Tensor | None:
    value = outputs.get(key)
    if isinstance(value, torch.Tensor):
        return value
    return None


def _mean_basis_activation_count(
    free_path_usage: torch.Tensor | None,
    *,
    threshold: float,
) -> torch.Tensor | None:
    if not isinstance(free_path_usage, torch.Tensor):
        return None
    activations = (free_path_usage > float(threshold)).to(dtype=free_path_usage.dtype)
    return activations.sum(dim=1).mean()


def _laplacian_term(
    mats: torch.Tensor | None,
    laplacian: object,
) -> torch.Tensor | None:
    if not isinstance(mats, torch.Tensor) or laplacian is None:
        return None
    return matrix_laplacian_loss(mats, laplacian)


def _reconstruction_terms(
    outputs: dict,
    x: torch.Tensor,
    recon_mask: torch.Tensor,
    private_residual_weight: float,
) -> dict[str, torch.Tensor]:
    recon_per = (outputs["reconstructed"] - x).abs().mean(dim=(2, 3, 4))
    shared_per = (outputs["action_reconstruction"] - x).abs().mean(dim=(2, 3, 4))
    recon_loss = masked_mean(recon_per, recon_mask)
    shared_recon_loss = masked_mean(shared_per, recon_mask)
    lq_loss = masked_mean(outputs["lq_loss_per_sample"], recon_mask)
    residual_l1 = masked_mean(outputs["residual_l1_per_sample"], recon_mask)

    private_residual = outputs.get("private_residual")
    if isinstance(private_residual, torch.Tensor):
        scaled_private_residual_l1 = masked_mean(
            private_residual.abs().mean(dim=(2, 3, 4)) * float(private_residual_weight),
            recon_mask,
        )
    else:
        scaled_private_residual_l1 = x.new_zeros(())

    return {
        "recon_loss": recon_loss,
        "shared_recon_loss": shared_recon_loss,
        "lq_loss": lq_loss,
        "residual_l1": residual_l1,
        "scaled_private_residual_l1": scaled_private_residual_l1,
    }


def _action_side_terms(
    outputs: dict,
    side_labels: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    action_side_loss_value = None
    action_side_acc_value = None
    group_action_logits = outputs.get("group_action_logits")
    group_valid_mask = supervision_mask.any(dim=1)
    if isinstance(group_action_logits, torch.Tensor) and group_valid_mask.any():
        group_labels = side_labels[group_valid_mask]
        action_side_loss_value = F.cross_entropy(
            group_action_logits[group_valid_mask],
            group_labels,
        )
        preds = group_action_logits[group_valid_mask].argmax(dim=1)
        action_side_acc_value = (preds == group_labels).to(dtype=torch.float32).mean()
    return {
        "action_side_loss_value": action_side_loss_value,
        "action_side_acc_value": action_side_acc_value,
    }


def _private_side_adv_terms(
    outputs: dict,
    side_labels: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    private_side_adv_loss_value = None
    private_side_acc_value = None
    private_side_logits = outputs.get("private_side_logits")
    if isinstance(private_side_logits, torch.Tensor):
        if private_side_logits.ndim == 3:
            flat_logits = private_side_logits[supervision_mask]
            flat_labels = side_labels.unsqueeze(1).expand_as(supervision_mask)[supervision_mask]
        else:
            flat_logits = private_side_logits
            flat_labels = side_labels
        if flat_logits.numel() > 0:
            private_side_adv_loss_value = F.cross_entropy(flat_logits, flat_labels)
            preds = flat_logits.argmax(dim=1)
            private_side_acc_value = (preds == flat_labels).to(dtype=torch.float32).mean()
    return {
        "private_side_adv_loss_value": private_side_adv_loss_value,
        "private_side_acc_value": private_side_acc_value,
    }


def _optional_regularizer_terms(outputs: dict, model) -> dict[str, torch.Tensor | None]:
    return {
        "v9_freq": _get_optional_metric(outputs, "v9_freq_loss"),
        "lowrank_orth": _get_optional_metric(outputs, "lowrank_orth_loss"),
        "reflex_orth": _get_optional_metric(outputs, "reflex_orth_loss"),
        "shared_coeff_l1": _get_optional_metric(outputs, "shared_coeff_l1"),
        "side_coeff_l1": _get_optional_metric(outputs, "side_coeff_l1"),
        "side_private_orth": _get_optional_metric(outputs, "side_private_orth_loss"),
        "lap_basis": _laplacian_term(
            outputs.get("action_basis"),
            getattr(model, "region_laplacian", None),
        ),
        "lap_side_basis": _laplacian_term(
            outputs.get("side_basis"),
            getattr(model, "region_laplacian", None),
        ),
        "lap_recon": _laplacian_term(
            outputs.get("reconstructed"),
            getattr(model, "region_laplacian", None),
        ),
    }


def _accumulate_reflex_total_loss(
    *,
    outputs: dict,
    loss_weights: dict,
    reconstruction_terms: dict[str, torch.Tensor],
    action_side_terms: dict[str, torch.Tensor | None],
    private_side_terms: dict[str, torch.Tensor | None],
    regularizer_terms: dict[str, torch.Tensor | None],
) -> torch.Tensor:
    total_loss = loss_weights.get("recon", 1.0) * reconstruction_terms["recon_loss"]
    total_loss = total_loss + loss_weights.get("shared_recon", 1.0) * reconstruction_terms["shared_recon_loss"]
    total_loss = total_loss + loss_weights.get("lq", 0.0) * reconstruction_terms["lq_loss"]
    total_loss = total_loss + loss_weights.get("orth", 0.0) * outputs["orth_loss"]
    total_loss = total_loss + loss_weights.get("basis_l1", 0.0) * outputs["basis_l1"]
    total_loss = total_loss + loss_weights.get("residual", 0.0) * reconstruction_terms["residual_l1"]

    action_side_loss_value = action_side_terms["action_side_loss_value"]
    if action_side_loss_value is not None:
        total_loss = total_loss + loss_weights.get("action_side", 0.0) * action_side_loss_value

    private_side_adv_loss_value = private_side_terms["private_side_adv_loss_value"]
    if private_side_adv_loss_value is not None:
        total_loss = total_loss + loss_weights.get("private_side_adv", 0.0) * private_side_adv_loss_value

    for key, weight_key in (
        ("v9_freq", "v9_freq"),
        ("lowrank_orth", "lowrank_orth"),
        ("reflex_orth", "reflex_orth"),
        ("shared_coeff_l1", "shared_coeff_l1"),
        ("side_coeff_l1", "side_coeff_l1"),
        ("side_private_orth", "side_private_orth"),
        ("lap_basis", "lap_basis"),
        ("lap_side_basis", "lap_side_basis"),
        ("lap_recon", "lap_recon"),
    ):
        value = regularizer_terms[key]
        if value is not None:
            total_loss = total_loss + loss_weights.get(weight_key, 0.0) * value

    return total_loss


def _build_step_metrics(
    *,
    total_loss: torch.Tensor,
    outputs: dict,
    reconstruction_terms: dict[str, torch.Tensor],
    action_side_terms: dict[str, torch.Tensor | None],
    private_side_terms: dict[str, torch.Tensor | None],
    regularizer_terms: dict[str, torch.Tensor | None],
    recon_mask: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> dict[str, float]:
    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "recon": float(reconstruction_terms["recon_loss"].detach().cpu()),
        "shared_recon": float(reconstruction_terms["shared_recon_loss"].detach().cpu()),
        "lq": float(reconstruction_terms["lq_loss"].detach().cpu()),
        "orth": float(outputs["orth_loss"].detach().cpu()),
        "shared_basis_l1": float(outputs["shared_basis_l1"].detach().cpu()),
        "side_basis_l1": float(outputs["side_basis_l1"].detach().cpu()),
        "basis_l1": float(outputs["basis_l1"].detach().cpu()),
        "residual": float(reconstruction_terms["residual_l1"].detach().cpu()),
        "scaled_residual": float(reconstruction_terms["scaled_private_residual_l1"].detach().cpu()),
        "recon_frames": float(recon_mask.sum().detach().cpu()),
        "supervision_frames": float(supervision_mask.sum().detach().cpu()),
    }

    action_side_loss_value = action_side_terms["action_side_loss_value"]
    action_side_acc_value = action_side_terms["action_side_acc_value"]
    private_side_adv_loss_value = private_side_terms["private_side_adv_loss_value"]
    private_side_acc_value = private_side_terms["private_side_acc_value"]

    if action_side_loss_value is not None:
        metrics["action_side"] = float(action_side_loss_value.detach().cpu())
    if action_side_acc_value is not None:
        metrics["action_side_acc"] = float(action_side_acc_value.detach().cpu())
    if private_side_adv_loss_value is not None:
        metrics["private_side_adv"] = float(private_side_adv_loss_value.detach().cpu())
    if private_side_acc_value is not None:
        metrics["private_side_acc"] = float(private_side_acc_value.detach().cpu())

    for key in (
        "v9_freq",
        "lowrank_orth",
        "reflex_orth",
        "shared_coeff_l1",
        "side_coeff_l1",
        "side_private_orth",
        "lap_basis",
        "lap_side_basis",
        "lap_recon",
    ):
        value = regularizer_terms[key]
        if value is not None:
            metrics[key] = float(value.detach().cpu())

    return metrics


def forward_reflex_batch(
    model,
    batch: dict,
    device: str,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["images"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    recon_mask = ~padding_mask
    side_labels = batch["side_label"].to(device)
    static_side_input = batch.get("static_side_input")
    if isinstance(static_side_input, torch.Tensor):
        static_side_input = static_side_input.to(device)

    # TODO(recovery): re-introduce dataset supervision once the recovered modular
    # PhaseAB path is stable. The current short-run target only exercises the
    # side/action supervision branch used by the supplied historical config.
    outputs = model(
        x,
        side_labels=side_labels,
        dataset_labels=None,
        valid_mask=valid_mask,
        static_side_input=static_side_input,
    )
    return outputs, x, valid_mask, recon_mask, side_labels


def build_no_side_eval_metrics(
    *,
    outputs: dict,
    x: torch.Tensor,
    recon_mask: torch.Tensor,
    loss_weights: dict,
    model,
) -> dict[str, float]:
    reconstruction_terms = _reconstruction_terms(
        outputs,
        x,
        recon_mask,
        float(getattr(model, "private_residual_weight", 0.0)),
    )
    regularizer_terms = _optional_regularizer_terms(outputs, model)

    total_loss = loss_weights.get("recon", 1.0) * reconstruction_terms["recon_loss"]
    total_loss = total_loss + loss_weights.get("shared_recon", 1.0) * reconstruction_terms["shared_recon_loss"]

    for key, weight_key in (
        ("v9_freq", "v9_freq"),
        ("lowrank_orth", "lowrank_orth"),
        ("reflex_orth", "reflex_orth"),
        ("shared_coeff_l1", "shared_coeff_l1"),
        ("side_coeff_l1", "side_coeff_l1"),
        ("side_private_orth", "side_private_orth"),
        ("lap_basis", "lap_basis"),
        ("lap_side_basis", "lap_side_basis"),
        ("lap_recon", "lap_recon"),
    ):
        value = regularizer_terms[key]
        if value is not None:
            total_loss = total_loss + loss_weights.get(weight_key, 0.0) * value

    metrics: dict[str, float] = {
        "loss": float(total_loss.detach().cpu()),
        "recon": float(reconstruction_terms["recon_loss"].detach().cpu()),
        "shared_recon": float(reconstruction_terms["shared_recon_loss"].detach().cpu()),
    }
    for key in (
        "v9_freq",
        "lowrank_orth",
        "reflex_orth",
        "shared_coeff_l1",
        "side_coeff_l1",
        "side_private_orth",
        "lap_basis",
        "lap_side_basis",
        "lap_recon",
    ):
        value = regularizer_terms[key]
        if value is not None:
            metrics[key] = float(value.detach().cpu())

    basis_activation_count = _mean_basis_activation_count(outputs.get("free_path_usage"), threshold=0.05)
    if basis_activation_count is not None:
        metrics["basis_activation_count"] = float(basis_activation_count.detach().cpu())
    return metrics


def step_model(model, batch, device, loss_weights):
    outputs, x, valid_mask, recon_mask, side_labels = forward_reflex_batch(model, batch, device)
    supervision_mask = valid_mask

    reconstruction_terms = _reconstruction_terms(
        outputs,
        x,
        recon_mask,
        float(getattr(model, "private_residual_weight", 0.0)),
    )

    # TODO(recovery): `group_side_loss_weight` is intentionally ignored here.
    # The original PhaseAB run used `action_side_weight` as the effective entry,
    # while `group_side_loss_weight` survived as a stale compatibility field.
    action_side_terms = _action_side_terms(outputs, side_labels, supervision_mask)
    private_side_terms = _private_side_adv_terms(outputs, side_labels, supervision_mask)
    regularizer_terms = _optional_regularizer_terms(outputs, model)

    total_loss = _accumulate_reflex_total_loss(
        outputs=outputs,
        loss_weights=loss_weights,
        reconstruction_terms=reconstruction_terms,
        action_side_terms=action_side_terms,
        private_side_terms=private_side_terms,
        regularizer_terms=regularizer_terms,
    )

    metrics = _build_step_metrics(
        total_loss=total_loss,
        outputs=outputs,
        reconstruction_terms=reconstruction_terms,
        action_side_terms=action_side_terms,
        private_side_terms=private_side_terms,
        regularizer_terms=regularizer_terms,
        recon_mask=recon_mask,
        supervision_mask=supervision_mask,
    )

    basis_activation_count = _mean_basis_activation_count(outputs.get("free_path_usage"), threshold=0.05)
    if basis_activation_count is not None:
        metrics["basis_activation_count"] = float(basis_activation_count.detach().cpu())

    return total_loss, metrics


__all__ = [
    "build_no_side_eval_metrics",
    "forward_reflex_batch",
    "_get_optional_metric",
    "_mean_basis_activation_count",
    "masked_mean",
    "masked_mean_per_sequence",
    "side_private_orthogonality_loss",
    "step_model",
]
