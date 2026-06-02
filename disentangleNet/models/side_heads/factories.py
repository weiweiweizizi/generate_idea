from __future__ import annotations


def build_side_branch_config_overrides(config) -> dict:
    """
    Flatten side-branch config fields into model-constructor kwargs.

    TODO(recovery): restore the historical side-head factory abstraction if we
    need multiple side branch runtimes beyond the current PhaseAB path. Right
    now this helper only exposes the kwargs consumed by the recovered V6-style
    family implementation.
    """

    action_side_input = (
        config.side.prediction_input.value
        if hasattr(config.side.prediction_input, "value")
        else str(config.side.prediction_input)
    )
    if action_side_input == "side_pair_choice_coeff":
        # TODO(recovery): restore the dedicated paired-side runtime naming once
        # the original side-head implementation is recovered. The current
        # recovered V6 family uses `shared_side_coeff` for this path.
        action_side_input = "shared_side_coeff"

    return {
        "side_residual_enabled": bool(config.side.enabled),
        "side_feature_mode": str(config.side.feature_mode),
        "side_residual_weight": float(config.side.residual_weight),
        "side_coeff_l1_weight": float(config.side.coeff_l1_weight),
        "side_private_orth_weight": float(config.side.private_orth_weight),
        "private_side_adv_weight": float(config.side.private_adv_weight),
        "private_side_grl_lambda": float(config.side.private_grl_lambda),
        "action_side_input": action_side_input,
    }


__all__ = ["build_side_branch_config_overrides"]
