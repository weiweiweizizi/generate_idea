from __future__ import annotations

import torch
import torch.nn.functional as F


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
    if not getattr(model, "side_semantic_enabled", False):
        raise ValueError("disentangleNet v31 requires side_semantic_enabled=True")
    side_labels = batch["side_label"].to(device)

    outputs = model(
        x,
        side_labels=side_labels,
        dataset_labels=None,
    )

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
    subspace_orth_loss = outputs["reconstructed"].new_zeros(())

    total_loss = loss_weights["recon"] * recon_loss
    total_loss = total_loss + loss_weights["shared_recon"] * shared_recon_loss
    total_loss = total_loss + loss_weights["lq"] * lq_loss
    total_loss = total_loss + loss_weights["orth"] * outputs["orth_loss"]
    total_loss = total_loss + loss_weights["basis_l1"] * outputs["basis_l1"]
    total_loss = total_loss + loss_weights["residual"] * residual_l1
    total_loss = total_loss + loss_weights["subspace_orth"] * subspace_orth_loss

    side_loss_cont = outputs["side_loss"]["side_loss_cont_per_sample"]
    side_loss_cont_value = None
    side_group_loss_value = None
    if side_loss_cont is not None:
        side_loss_cont_value = masked_mean(side_loss_cont, supervision_mask)

    side_group_rep = outputs["side_path_representation"]
    if side_group_rep is None:
        raise RuntimeError("side_path_representation is required for group supervision")
    if side_group_rep.ndim != 3:
        raise ValueError(
            "group supervision expects grouped side_path_representation with shape B x T x D"
        )
    group_valid_mask = supervision_mask.any(dim=1)
    if group_valid_mask.any():
        group_side_logits = outputs["group_side_logits"]
        group_side_labels = batch["side_label"].to(device)[group_valid_mask]
        if group_side_logits is not None:
            side_group_loss_value = F.cross_entropy(
                group_side_logits[group_valid_mask],
                group_side_labels,
            )

    if side_loss_cont_value is not None:
        total_loss = total_loss + loss_weights["side_cont"] * side_loss_cont_value
    if side_group_loss_value is not None:
        total_loss = total_loss + loss_weights["side_group"] * side_group_loss_value

    loss_metrics = {
        "loss": float(total_loss.detach().cpu()),
        "recon": float(recon_loss.detach().cpu()),
        "shared_recon": float(shared_recon_loss.detach().cpu()),
        "lq": float(lq_loss.detach().cpu()),
        "orth": float(outputs["orth_loss"].detach().cpu()),
        "shared_basis_l1": float(outputs["shared_basis_l1"].detach().cpu()),
        "side_basis_l1": float(outputs["side_basis_l1"].detach().cpu()),
        "basis_l1": float(outputs["basis_l1"].detach().cpu()),
        "residual": float(residual_l1.detach().cpu()),
        "scaled_residual": float(scaled_private_residual_l1.detach().cpu()),
        "subspace_orth": float(subspace_orth_loss.detach().cpu()),
        "recon_frames": float(recon_mask.sum().detach().cpu()),
        "supervision_frames": float(supervision_mask.sum().detach().cpu()),
    }
    if side_loss_cont_value is not None:
        loss_metrics["side_ce"] = float(side_loss_cont_value.detach().cpu())
    if side_group_loss_value is not None:
        loss_metrics["side_group_ce"] = float(side_group_loss_value.detach().cpu())
    loss_metrics["free_side_adv"] = 0.0

    probe_outputs = {
        "side_logits": outputs["side_logits"],
        "group_side_logits": outputs["group_side_logits"],
    }

    return total_loss, loss_metrics, probe_outputs
