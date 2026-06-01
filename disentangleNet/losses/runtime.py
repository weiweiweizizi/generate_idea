"""
Recovered module placeholder for disentangleNet/losses/runtime.py.

PYC-confirmed top-level symbols:
- runtime.cpython-311.pyc
- runtime.cpython-38.pyc

Observed names:
- masked_mean
- masked_mean_per_sequence
- step_model
- matrix_laplacian_loss
"""


def masked_mean(values, mask):
    """Compute a masked mean over batch or sequence dimensions."""


def masked_mean_per_sequence(values, mask):
    """Compute per-sequence masked means for grouped samples."""


def _get_optional_metric(outputs, key):
    """Read an optional scalar metric from a model output dictionary."""


def _mean_basis_activation_count(free_path_usage, *, threshold):
    """Estimate average basis activation count above a threshold."""


def step_model(model, batch, device, loss_weights):
    """
    Run one forward/loss/backward step and return loss plus metrics.

    Recovered locals show direct handling of:
    - `x`
    - `valid_mask`
    - `padding_mask`
    - `recon_mask`
    - `supervision_mask`
    - `side_labels`
    - `static_side_input`
    - `outputs`
    """

__all__ = [
    "_get_optional_metric",
    "_mean_basis_activation_count",
    "masked_mean",
    "masked_mean_per_sequence",
    "step_model",
]
