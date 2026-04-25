from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked positions while keeping gradients well-defined."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_mean_per_sequence(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool each sequence independently using a boolean valid-frame mask."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)

    denom = mask.sum(dim=1).clamp_min(1.0)
    while denom.ndim < values.ndim - 1:
        denom = denom.unsqueeze(-1)

    return (values * expanded_mask).sum(dim=1) / denom


def masked_subspace_orthogonality_loss(
    side_latent: torch.Tensor | None,
    free_latent: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize linear leakage between side/free latent subspaces on valid frames."""

    if side_latent is None or free_latent is None:
        return mask.new_zeros((), dtype=torch.float32)
    if side_latent.numel() == 0 or free_latent.numel() == 0:
        return side_latent.new_zeros(())

    valid = mask.to(device=side_latent.device, dtype=torch.bool)
    side_valid = side_latent[valid]
    free_valid = free_latent[valid]
    if side_valid.numel() == 0 or free_valid.numel() == 0:
        return side_latent.new_zeros(())

    side_valid = side_valid - side_valid.mean(dim=0, keepdim=True)
    free_valid = free_valid - free_valid.mean(dim=0, keepdim=True)
    side_valid = side_valid / side_valid.std(dim=0, keepdim=True).clamp_min(1e-6)
    free_valid = free_valid / free_valid.std(dim=0, keepdim=True).clamp_min(1e-6)

    cross_corr = side_valid.transpose(0, 1) @ free_valid
    cross_corr = cross_corr / max(int(side_valid.shape[0]), 1)
    return cross_corr.square().mean()


def step_model(model, batch, device, loss_weights):
    x = batch["images"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    recon_mask = ~padding_mask
    supervision_mask = valid_mask
    use_side_group_supervision = getattr(model, "side_semantic_enabled", False)
    if loss_weights["side_group"] > 0.0 and not use_side_group_supervision:
        raise ValueError("side_group loss requires side_semantic_enabled=True on the model")
    use_side_supervision = (
        use_side_group_supervision
        or loss_weights["side_group"] > 0.0
        or loss_weights["side_cont"] > 0.0
        or loss_weights["side_disc"] > 0.0
        or loss_weights["free_side_adv"] > 0.0
    )
    side_labels = batch["side_label"].to(device) if use_side_supervision else None
    dataset_labels = batch["dataset_label"].to(device) if model.use_dataset_aux else None

    outputs = model(x, side_labels=side_labels, dataset_labels=dataset_labels)
    early_branch_factorization = bool(getattr(model, "early_branch_factorization", False))

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
    if early_branch_factorization:
        subspace_orth_loss = outputs["reconstructed"].new_zeros(())
    else:
        subspace_orth_loss = masked_subspace_orthogonality_loss(
            outputs.get("side_latent_raw", outputs.get("side_latent")),
            outputs.get("free_latent_raw", outputs.get("free_latent")),
            supervision_mask,
        )

    total_loss = loss_weights["recon"] * recon_loss
    total_loss = total_loss + loss_weights["shared_recon"] * shared_recon_loss
    total_loss = total_loss + loss_weights["lq"] * lq_loss
    total_loss = total_loss + loss_weights["orth"] * outputs["orth_loss"]
    total_loss = total_loss + loss_weights["basis_l1"] * outputs["basis_l1"]
    total_loss = total_loss + loss_weights["residual"] * residual_l1
    total_loss = total_loss + loss_weights["subspace_orth"] * subspace_orth_loss

    side_loss = outputs["side_loss"]["side_loss_per_sample"]
    side_loss_cont = outputs["side_loss"]["side_loss_cont_per_sample"]
    side_loss_disc = outputs["side_loss"]["side_loss_disc_per_sample"]
    side_loss_value = None
    side_loss_cont_value = None
    side_loss_disc_value = None
    side_group_loss_value = None
    severity_group_loss_value = None
    free_side_adv_loss_value = None
    if side_loss is not None:
        side_loss_value = masked_mean(side_loss, supervision_mask)
    if side_loss_cont is not None:
        side_loss_cont_value = masked_mean(side_loss_cont, supervision_mask)
        if not use_side_group_supervision:
            total_loss = total_loss + loss_weights["side_cont"] * side_loss_cont_value
    if side_loss_disc is not None:
        side_loss_disc_value = masked_mean(side_loss_disc, supervision_mask)
        if not use_side_group_supervision:
            total_loss = total_loss + loss_weights["side_disc"] * side_loss_disc_value
    if use_side_group_supervision:
        side_group_rep = outputs["side_path_representation"]
        if side_group_rep is None:
            raise RuntimeError("side_path_representation is required for side_group supervision")
        if side_group_rep.ndim != 3:
            raise ValueError(
                "side_group supervision expects grouped side_path_representation with shape B x T x D"
            )
        group_side_rep = masked_mean_per_sequence(side_group_rep, supervision_mask)
        group_valid_mask = supervision_mask.any(dim=1)
        if group_valid_mask.any():
            group_side_logits = model.classify_side_group(group_side_rep[group_valid_mask])
            group_side_labels = batch["side_label"].to(device)[group_valid_mask]
            side_group_loss_value = F.cross_entropy(group_side_logits, group_side_labels)
            total_loss = total_loss + loss_weights["side_group"] * side_group_loss_value
    if loss_weights["severity_group"] > 0.0:
        free_level2_usage = outputs["free_level2_usage"]
        if free_level2_usage is None:
            raise RuntimeError("free_level2_usage is required for severity supervision")
        if free_level2_usage.ndim != 3:
            raise ValueError(
                "severity supervision expects grouped free_level2_usage with shape B x T x D"
            )
        group_free_level2_usage = masked_mean_per_sequence(free_level2_usage, supervision_mask)
        group_valid_mask = supervision_mask.any(dim=1)
        if group_valid_mask.any():
            group_severity_logits = model.classify_severity_group(
                group_free_level2_usage[group_valid_mask]
            )
            group_severity_labels = batch["severity_label"].to(device)[group_valid_mask]
            severity_group_loss_value = F.cross_entropy(
                group_severity_logits,
                group_severity_labels,
            )
            total_loss = total_loss + loss_weights["severity_group"] * severity_group_loss_value
    free_side_adv_loss = outputs["side_loss"]["free_side_adv_loss_per_sample"]
    if early_branch_factorization:
        free_side_adv_loss_value = outputs["reconstructed"].new_zeros(())
    elif free_side_adv_loss is not None:
        free_side_adv_loss_value = masked_mean(free_side_adv_loss, supervision_mask)
        total_loss = total_loss + loss_weights["free_side_adv"] * free_side_adv_loss_value

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
        "shared_basis_l1": float(outputs["shared_basis_l1"].detach().cpu()),
        "side_basis_l1": float(outputs["side_basis_l1"].detach().cpu()),
        "basis_l1": float(outputs["basis_l1"].detach().cpu()),
        "residual": float(residual_l1.detach().cpu()),
        "scaled_residual": float(scaled_private_residual_l1.detach().cpu()),
        "subspace_orth": float(subspace_orth_loss.detach().cpu()),
        "recon_frames": float(recon_mask.sum().detach().cpu()),
        "supervision_frames": float(supervision_mask.sum().detach().cpu()),
    }
    if side_loss_value is not None:
        metrics["side"] = float(side_loss_value.detach().cpu())
    if side_loss_cont_value is not None:
        metrics["side_cont"] = float(side_loss_cont_value.detach().cpu())
    if side_loss_disc_value is not None:
        metrics["side_disc"] = float(side_loss_disc_value.detach().cpu())
    if side_group_loss_value is not None:
        metrics["side_group"] = float(side_group_loss_value.detach().cpu())
    if severity_group_loss_value is not None:
        metrics["severity_group"] = float(severity_group_loss_value.detach().cpu())
    if free_side_adv_loss_value is not None:
        metrics["free_side_adv"] = float(free_side_adv_loss_value.detach().cpu())
    if dataset_private_loss_value is not None:
        metrics["dataset_private"] = float(dataset_private_loss_value.detach().cpu())
    if dataset_adv_loss_value is not None:
        metrics["dataset_adv"] = float(dataset_adv_loss_value.detach().cpu())

    return total_loss, metrics
