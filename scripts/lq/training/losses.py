from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked positions while keeping gradients well-defined."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def step_model(model, batch, device, loss_weights):
    x = batch["images"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    recon_mask = ~padding_mask
    supervision_mask = valid_mask
    use_side_supervision = (
        loss_weights["side_cont"] > 0.0 or loss_weights["side_disc"] > 0.0
    )
    side_labels = batch["side_label"].to(device) if use_side_supervision else None
    dataset_labels = batch["dataset_label"].to(device) if model.use_dataset_aux else None

    outputs = model(x, side_labels=side_labels, dataset_labels=dataset_labels)

    recon_loss_per_frame = (outputs["reconstructed"] - x).abs().mean(dim=(2, 3, 4))
    shared_recon_loss_per_frame = (
        outputs["action_reconstruction"] - x
    ).abs().mean(dim=(2, 3, 4))
    recon_loss = masked_mean(recon_loss_per_frame, recon_mask)
    shared_recon_loss = masked_mean(shared_recon_loss_per_frame, recon_mask)
    lq_loss = masked_mean(outputs["lq_loss_per_sample"], recon_mask)
    residual_l1 = masked_mean(outputs["residual_l1_per_sample"], recon_mask)
    scaled_private_residual_l1 = masked_mean(
        outputs["private_residual"].abs().mean(dim=(2, 3, 4)) * model.private_residual_weight,
        recon_mask,
    )

    total_loss = loss_weights["recon"] * recon_loss
    total_loss = total_loss + loss_weights["shared_recon"] * shared_recon_loss
    total_loss = total_loss + loss_weights["lq"] * lq_loss
    total_loss = total_loss + loss_weights["orth"] * outputs["orth_loss"]
    total_loss = total_loss + loss_weights["basis_l1"] * outputs["basis_l1"]
    total_loss = total_loss + loss_weights["residual"] * residual_l1

    side_loss = outputs["side_loss"]["side_loss_per_sample"]
    side_loss_cont = outputs["side_loss"]["side_loss_cont_per_sample"]
    side_loss_disc = outputs["side_loss"]["side_loss_disc_per_sample"]
    side_loss_value = None
    side_loss_cont_value = None
    side_loss_disc_value = None
    if side_loss is not None:
        side_loss_value = masked_mean(side_loss, supervision_mask)
    if side_loss_cont is not None:
        side_loss_cont_value = masked_mean(side_loss_cont, supervision_mask)
        total_loss = total_loss + loss_weights["side_cont"] * side_loss_cont_value
    if side_loss_disc is not None:
        side_loss_disc_value = masked_mean(side_loss_disc, supervision_mask)
        total_loss = total_loss + loss_weights["side_disc"] * side_loss_disc_value

    dataset_private_loss = outputs["dataset_loss"]["private_dataset_loss_per_sample"]
    dataset_private_loss_value = None
    if dataset_private_loss is not None:
        dataset_private_loss_value = masked_mean(dataset_private_loss, supervision_mask)
        total_loss = total_loss + loss_weights["dataset_private"] * dataset_private_loss_value

    dataset_adv_loss = outputs["dataset_loss"]["shared_dataset_adv_loss_per_sample"]
    dataset_adv_loss_value = None
    if dataset_adv_loss is not None:
        dataset_adv_loss_value = masked_mean(dataset_adv_loss, supervision_mask)
        total_loss = total_loss + loss_weights["dataset_adv"] * dataset_adv_loss_value

    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "recon": float(recon_loss.detach().cpu()),
        "shared_recon": float(shared_recon_loss.detach().cpu()),
        "lq": float(lq_loss.detach().cpu()),
        "orth": float(outputs["orth_loss"].detach().cpu()),
        "basis_l1": float(outputs["basis_l1"].detach().cpu()),
        "residual": float(residual_l1.detach().cpu()),
        "scaled_residual": float(scaled_private_residual_l1.detach().cpu()),
        "recon_frames": float(recon_mask.sum().detach().cpu()),
        "supervision_frames": float(supervision_mask.sum().detach().cpu()),
    }
    if side_loss_value is not None:
        metrics["side"] = float(side_loss_value.detach().cpu())
    if side_loss_cont_value is not None:
        metrics["side_cont"] = float(side_loss_cont_value.detach().cpu())
    if side_loss_disc_value is not None:
        metrics["side_disc"] = float(side_loss_disc_value.detach().cpu())
    if dataset_private_loss_value is not None:
        metrics["dataset_private"] = float(dataset_private_loss_value.detach().cpu())
    if dataset_adv_loss_value is not None:
        metrics["dataset_adv"] = float(dataset_adv_loss_value.detach().cpu())

    return total_loss, metrics
