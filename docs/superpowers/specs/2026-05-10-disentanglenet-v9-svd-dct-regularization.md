# DisentangleNet V9: SVD Truncation + DCT Frequency Regularization

## Goal

Introduce structured low-rank and frequency-domain regularization to the learned action basis bank in the V6DistNet architecture. The goal is to:

1. Force each learned basis to be low-rank (smooth, interpretable structure) via **SVD truncation**
2. Penalize high-frequency noise in each basis via **DCT frequency loss**
3. Encourage sparse basis selection via **L1 on level weights**
4. Add mild **L1 on basis** to suppress weak activations

All regularization operates on the basis bank only — the reconstruction path remains unchanged.

## Background

V8A uses V6DistNet (no side branch) with `shared_basis_soft_mixing=True`. The basis bank `[8, 119, 119]` is split into level0 `[2, 119, 119]` and level1 `[6, 119, 119]`. Each level selects bases via softmax weights and reconstructs via weighted sum + scalar coefficient.

The basis is currently unregularized beyond the initial QR orthogonalization. In practice, learned bases can develop high-frequency noise and non-interpretable structure. V9 aims to fix this.

## Architecture

```
Base: V6DistNet (v8A architecture)
Checkpoint init: v2_recon_x/best.pt (strict=False, only shared params)

V9 additions:
  ┌─ get_structured_basis() ─────────────────────────────────────┐
  │  basis_bank [8,119,119]                                       │
  │     │                                                         │
  │     ├─ enforce_matrix_constraints (sym + zero-diag)           │
  │     ├─ SVD per-basis truncation:                              │
  │     │    level0 (2 bases): keep top k=2 singular components   │
  │     │    level1 (6 bases): keep top k=5 singular components   │
  │     └─ return truncated basis                                 │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

## SVD Truncation

### Per-basis SVD

For each basis matrix `B_i ∈ R^{119×119}`:

```
U, S, Vh = svd(B_i)                    # S ∈ R^{119}
S[k:] = 0                              # zero out beyond rank k
B_i' = U @ diag(S) @ Vh                # low-rank reconstruction
```

- **level0**: `k=2` (rank-2 approximation, retains ~1.7% of singular directions)
- **level1**: `k=5` (rank-5 approximation, retains ~4.2% of singular directions)

### Why per-basis, not per-level

Each basis is independently truncated so each remains an interpretable low-rank pattern. Per-level SVD would mix bases and destroy individual interpretability.

### Gradient flow

`torch.linalg.svd` is differentiable. Gradient flows through U, S, Vh back to the basis bank parameters, so the encoder learns to produce bases that are naturally low-rank even before truncation.

## New Loss Terms

### 1. Basis L1 penalty

```
L_basis_l1 = mean(|B_i|) for all i
```

- **Weight**: `1e-5` (very small — mild sparsity regularizer)
- **Rationale**: Suppresses weak activations without destroying structure

### 2. Level weights L1 penalty

```
L_weights_l1 = mean(|w_level0|) + mean(|w_level1|)
```

where `w_level` is the softmax-normalized basis selection weight per frame.

- **Weight**: `0.01`
- **Rationale**: Since softmax outputs sum to 1, L1 is minimized when one weight is 1 and the rest are 0. This encourages **sparse routing** — each frame uses as few bases as possible.

### 3. DCT frequency penalty

```
B̂_k = DCT2D(B_i)                       # 2D DCT of basis i
W_uv = u² + v²                          # frequency weight matrix
L_freq = sum_{i,u,v} W_uv * |B̂_k(u,v)|²
```

- **Weight**: `0.01`
- **Rationale**: The squared frequency weight `u²+v²` heavily penalizes high-frequency components (e.g., checkerboard noise) while leaving low-frequency (smooth) structure largely untouched. This forces each basis to be smooth and interpretable.

### DCT Implementation

2D DCT-II via matrix multiplication:

```
C = dct_matrix(N)     # [N, N] orthonormal DCT-II transform
B̂ = C @ B @ C^T       # 2D DCT
```

The DCT matrix is precomputed once and cached. This is fully differentiable.

## Training Configuration

```
checkpoint:        outputs/disentangleNet/v2_recon_x/best.pt
model:             V9DistNet (extends V6DistNet)
action_side_input: free_path_coeff (dim=2, Linear classifier, LR init)
output_dir:        outputs/disentangleNet/v9_svd_dct

Loss weights:
  recon:            1.0
  shared_recon:     1.0
  action_side:      10.0
  basis_l1:         1e-5
  level_weights_l1: 0.01
  freq:             0.01

SVD truncation:
  level0 k: 2
  level1 k: 5

Training:
  epochs:           200
  batch_size:       64
  lr:               3e-4
  shared_lr_mult:   3.0
  private_residual_weight: 0.20 (constant)
```

## File Changes

| File | Action | Purpose |
|------|--------|---------|
| `model/v9_utils.py` | **NEW** | `svd_truncate_basis()`, `dct2d()`, `dct_frequency_loss()` |
| `model/v9_distnet.py` | **NEW** | `V9DistNet(V6DistNet)` — overrides `get_structured_basis()` and `forward()` |
| `v9_loss.py` | **NEW** | `build_loss_weights(action_side_weight, basis_l1_weight, weights_l1_weight, freq_weight)` |
| `training/losses.py` | **MODIFY** | Add `step_model_v9()` or extend `step_model()` to handle v9-specific loss keys |
| `run_train_v9.py` | **NEW** | Training CLI (based on v8A structure) |

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| SVD k=2 too aggressive → reconstruction collapse | Medium | Monitor `shared_recon`; fallback to k=5/10 for level0 |
| DCT loss dominates total loss | Low | Start with weight 0.01; scale down if freq_loss >> recon_loss |
| SVD gradient instability (degenerate singular values) | Low | 119×119 matrices rarely degenerate; add eps to diag if needed |
| v2 checkpoint incompatibility (DistNet vs V6DistNet) | Known | Use `strict=False`; missing keys = side branch params |
| L1 on softmax weights conflicts with anchor_bias/topk | Low | L1 just encourages sparsity; softmax + topk still dominate selection |

## Validation

Smoke test:
- [ ] Model loads v2 checkpoint with strict=False
- [ ] Forward pass produces valid shapes
- [ ] SVD truncation reduces basis rank (verify: num nonzero singular values == k)
- [ ] DCT loss is computed and nonzero
- [ ] Backward pass succeeds (no NaN gradients)
- [ ] Batch memory fits in GPU

Full training:
- [ ] `shared_recon` loss decreases over epochs
- [ ] `freq_loss` decreases over epochs (bases become smoother)
- [ ] `level_weights_l1` approaches theoretical minimum (~1/k per level)
- [ ] TensorBoard shows basis norms, freq_loss, weights_l1 over time
