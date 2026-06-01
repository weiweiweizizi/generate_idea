from __future__ import annotations


def build_reflex_loss_weights(
    *,
    action_side_weight: float = 10.0,
    basis_l1_weight: float = 0.0,
    freq_weight: float = 0.01,
    lowrank_orth_weight: float = 0.01,
    residual_weight: float = 0.0,
    shared_coeff_l1_weight: float = 0.0001,
    side_coeff_l1_weight: float = 0.0,
    side_private_orth_weight: float = 0.0,
    private_side_adv_weight: float = 0.0,
    laplacian_basis_weight: float = 0.0,
    laplacian_side_basis_weight: float = 0.0,
    laplacian_recon_weight: float = 0.0,
) -> dict[str, float]:
    return {
        "recon": 1.0,
        "shared_recon": 1.0,
        "lq": 0.0,
        "orth": 0.0,
        "basis_l1": float(basis_l1_weight),
        "residual": float(residual_weight),
        "side_cont": 0.0,
        "side_disc": 0.0,
        "action_side": float(action_side_weight),
        "severity_group": 0.0,
        "subspace_orth": 0.0,
        "free_side_adv": 0.0,
        "dataset_private": 0.0,
        "dataset_adv": 0.0,
        "v9_freq": float(freq_weight),
        "v9_weights_l1": 0.0,
        "v9_svd_tail": 0.0,
        "lowrank_orth": float(lowrank_orth_weight),
        "reflex_orth": 0.0,
        "shared_coeff_l1": float(shared_coeff_l1_weight),
        "side_coeff_l1": float(side_coeff_l1_weight),
        "side_private_orth": float(side_private_orth_weight),
        "private_side_adv": float(private_side_adv_weight),
        "lap_basis": float(laplacian_basis_weight),
        "lap_side_basis": float(laplacian_side_basis_weight),
        "lap_recon": float(laplacian_recon_weight),
    }


def build_lowrank_loss_weights(**kwargs) -> dict[str, float]:
    return build_reflex_loss_weights(**kwargs)


def build_v31_loss_weights(config: dict) -> dict[str, float]:
    return {
        "recon": config["recon_weight"],
        "shared_recon": config["shared_recon_weight"],
        "lq": config["lq_weight"],
        "orth": config["orth_weight"],
        "basis_l1": config["basis_l1_weight"],
        "residual": config["residual_weight"],
        "side_cont": config["side_cont_weight"],
        "side_disc": config["side_disc_weight"],
        "side_group": config["group_side_loss_weight"],
        "severity_group": 0.0,
        "subspace_orth": 0.0,
        "free_side_adv": 0.0,
        "dataset_private": 0.0,
        "dataset_adv": 0.0,
        "lap_basis": float(config.get("laplacian_basis_weight", 0.0)),
        "lap_side_basis": float(config.get("laplacian_side_basis_weight", 0.0)),
        "lap_recon": float(config.get("laplacian_recon_weight", 0.0)),
    }


__all__ = [
    "build_lowrank_loss_weights",
    "build_reflex_loss_weights",
    "build_v31_loss_weights",
]
