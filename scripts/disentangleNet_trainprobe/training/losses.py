from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked positions while keeping gradients well-defined."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def weighted_branch_group_side_loss(
    branch_group_side_logits: dict[str, torch.Tensor | None],
    *,
    side_labels: torch.Tensor,
    group_valid_mask: torch.Tensor,
    loss_weights: dict,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Compute the requested weighted CE over the three masked side branches."""

    branch_weights = {
        "mouth_self": float(loss_weights["mouth_side_group"]),
        "mouth_cross_other": float(loss_weights["mouth_cross_side_group"]),
        "other_self": float(loss_weights["other_side_group"]),
    }
    total = sum(branch_weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Branch side-group weights must sum to 1, got {total}")

    losses = []
    metrics = {}
    for branch_name, branch_weight in branch_weights.items():
        logits = branch_group_side_logits.get(branch_name)
        if logits is None or branch_weight == 0.0:
            continue
        branch_loss = F.cross_entropy(logits[group_valid_mask], side_labels)
        losses.append(branch_weight * branch_loss)
        metrics[f"{branch_name}_side_group_ce"] = float(branch_loss.detach().cpu())

    if not losses:
        return None, metrics
    return torch.stack(losses).sum(), metrics


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

    group_valid_mask = supervision_mask.any(dim=1)
    if group_valid_mask.any():
        group_side_labels = batch["side_label"].to(device)[group_valid_mask]
        branch_group_side_logits = outputs["branch_group_side_logits"]
        if branch_group_side_logits:
            side_group_loss_value, branch_side_metrics = weighted_branch_group_side_loss(
                branch_group_side_logits,
                side_labels=group_side_labels,
                group_valid_mask=group_valid_mask,
                loss_weights=loss_weights,
            )
        else:
            branch_side_metrics = {}
    else:
        branch_side_metrics = {}

    if side_loss_cont_value is not None:
        total_loss = total_loss + loss_weights["side_cont"] * side_loss_cont_value
    if side_group_loss_value is not None:
        total_loss = total_loss + side_group_loss_value

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
    loss_metrics.update(branch_side_metrics)
    loss_metrics["free_side_adv"] = 0.0

    probe_outputs = {
        "side_logits": outputs["side_logits"],
        "group_side_logits": outputs["group_side_logits"],
        "branch_group_side_logits": outputs["branch_group_side_logits"],
    }

    return total_loss, loss_metrics, probe_outputs
