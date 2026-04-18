"""
NIHSS Heatmap Dataset — loads precomputed 3-channel correlation matrices.

Each sample is a ``(3, N, N)`` tensor (xy / x / y distance matrices) with
its 5-class label.  Matrices are read from ``.npy`` files produced by
``scripts/precompute_matrices.py``.

Note on small matrix values (e.g. max ≈ 0.01):
  Global normalization maps [vmin, vmax] → [0, 1], so the network always
  receives inputs in [0, 1] when vmax is estimated from the same data.
  However: (1) if vmax is 0 or ≤ vmin, outputs are forced to zeros; (2) if
  vmax is overestimated (e.g. from another dataset), normalized values stay
  small and train-time noise (std=0.01) can dominate; (3) 98th percentile from
  200 samples can be unstable.  We guard vmax > vmin and recommend checking
  logged vmin/vmax when values are very small.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Scale lower bound to avoid div-by-zero in tanh/log
_TRANSFORM_SCALE_EPS = 1e-6
import pandas as pd
import torch
from torch.utils.data import Dataset


class NIHSSHeatmapDataset(Dataset):
    """PyTorch Dataset for precomputed correlation-matrix heatmaps.

    After normalization, an optional transform is applied: power, tanh, or log.
    All transforms are zero-preserving (0→0) and sign-preserving (out = sign(x)*f(|x|)).
    Each supports a scale parameter (heatmap_transform_scale).
    """

    def __init__(
        self,
        subjs: list[str],
        precomputed_dir: str,
        *,
        is_train: bool = True,
        resize: Optional[Tuple[int, int]] = (128, 128),
        normalize: str = "global",
        global_stats: Optional[dict] = None,
        use_symmetric_normalize: bool = False,
        exponent: float = 1.0,
        transform: str = "power",
        transform_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        subjs : list[str]
            Subject IDs to include.
        precomputed_dir : str
            Root directory containing per-subject folders and ``metadata.csv``.
        is_train : bool
            Whether to apply training augmentation (random noise).
        resize : (H, W) or None
            Resize matrices to this size. ``None`` keeps original size.
        normalize : str
            ``"global"`` — use provided global_stats or [0, global_max].
            ``"per_sample"`` — normalize each channel to [0, 1] independently.
        global_stats : dict or None
            ``{"vmin": float, "vmax": float}`` for global normalization.
        exponent : float
            幂变换指数（仅当 transform=="power" 时作 scale 的兼容别名），归一化后应用。
        transform : str
            归一化后变换类型：power | tanh | log；保 0 且保号 out=sign(x)*f(|x|)。
        transform_scale : float
            缩放参数：power 即 x^scale；tanh 为 tanh(scale*x)/tanh(scale)；log 为 ln(1+scale*x)/ln(1+scale)。
        """
        self.precomputed_dir = Path(precomputed_dir)
        self.is_train = is_train
        self.resize = resize
        self.normalize = normalize
        self.global_stats = global_stats or {}
        self.use_symmetric_normalize = use_symmetric_normalize
        self.exponent = exponent
        self.transform = transform
        self.transform_scale = transform_scale

        meta_path = self.precomputed_dir / "metadata.csv"
        meta_df = pd.read_csv(str(meta_path))
        meta_df["subj"] = meta_df["subj"].astype(str).str.zfill(5)

        self.samples = meta_df[meta_df["subj"].isin(subjs)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]
        subj = str(row["subj"]).zfill(5)
        win_idx = int(row["window_idx"])
        label = int(row["label_5class"])

        subj_dir = self.precomputed_dir / subj

        mats = []
        for mode in ("xy", "x", "y"):
            npy_path = subj_dir / f"win_{win_idx:03d}_{mode}.npy"
            mat = np.load(str(npy_path)).astype(np.float64)
            mats.append(mat)

        image = np.stack(mats, axis=0)  # (3, N, N)

        if self.use_symmetric_normalize:
            scale = self.global_stats.get("scale")
            if scale is None or scale <= 0:
                scale = max(float(np.percentile(np.abs(image), 98)), 1e-6)
            x_signed = np.clip(image.astype(np.float64) / scale, -1.0, 1.0).astype(np.float64)
            image = self._apply_transform(x_signed, clip_to_01=False)
            # image = ((image + 1.0) * 0.5).astype(np.float64) 不压回[0,1]
        else:
            image = self._normalize(image)
            image = self._apply_transform(image)

        if self.resize is not None:
            image = self._resize(image, self.resize)

        if self.is_train:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float64)
            image = image + noise

        tensor_image = torch.from_numpy(image).float()
        tensor_label = torch.tensor(label, dtype=torch.int64)

        return {
            "image": tensor_image,
            "label": tensor_label,
            "image_name": f"{subj}_win{win_idx:03d}",
        }

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalize == "global":
            vmin = self.global_stats.get("vmin", 0.0)
            vmax = self.global_stats.get("vmax", None)
            if vmax is None:
                vmax = float(np.percentile(image, 99))
            if vmax <= vmin:
                return np.zeros_like(image)
            image = (image - vmin) / (vmax - vmin)
            image = np.clip(image, 0, 1)
        elif self.normalize == "per_sample":
            for c in range(image.shape[0]):
                ch = image[c]
                cmin, cmax = ch.min(), ch.max()
                if cmax > cmin:
                    image[c] = (ch - cmin) / (cmax - cmin)
                else:
                    image[c] = 0.0
        return image

    def _apply_transform(self, image: np.ndarray, clip_to_01: bool = True) -> np.ndarray:
        """Apply zero- and sign-preserving transform: out = sign(x) * f(|x|).

        When clip_to_01 is True (default), output is clipped to [0, 1].
        When False (symmetric path), output is clipped to [-1, 1].
        """
        scale = self.transform_scale
        if self.transform == "power" and self.exponent != 1.0:
            scale = self.exponent
        s = np.sign(image)
        a = np.clip(np.abs(image), 0.0, 1.0).astype(np.float64)
        if self.transform == "power":
            scale_safe = max(scale, _TRANSFORM_SCALE_EPS) if scale <= 0 else scale
            f = np.power(a, scale_safe)
        elif self.transform == "tanh":
            scale_safe = max(scale, _TRANSFORM_SCALE_EPS)
            f = np.tanh(scale_safe * a) / np.tanh(scale_safe)
        elif self.transform == "log":
            scale_safe = max(scale, _TRANSFORM_SCALE_EPS)
            f = np.log1p(scale_safe * a) / np.log1p(scale_safe)
        else:
            f = a
        out = s * f
        if clip_to_01:
            out = np.clip(out, 0.0, 1.0).astype(np.float64)
        else:
            out = np.clip(out, -1.0, 1.0).astype(np.float64)
        return out

    @staticmethod
    def _resize(image: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
        """Bilinear resize of (C, H, W) ndarray."""
        C, H, W = image.shape
        tH, tW = target
        if H == tH and W == tW:
            return image
        t = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)
        t = torch.nn.functional.interpolate(t, size=target, mode="bilinear", align_corners=False)
        return t.squeeze(0).numpy()

    @staticmethod
    def compute_global_stats(
        precomputed_dir: str,
        subjs: list[str],
        rng: Optional[np.random.RandomState] = None,
        use_symmetric_normalize: bool = False,
    ) -> dict:
        """Scan a subset of data to compute global normalization stats.

        Uses a local RNG (default RandomState(42)) so sampling is deterministic
        and independent of global numpy random state.

        When use_symmetric_normalize is True, computes scale from percentile of
        |raw| and returns scale, vmin=-scale, vmax=scale for zero-preserving
        map to [-1, 1].
        """
        if rng is None:
            rng = np.random.RandomState(42)
        precomputed_dir = Path(precomputed_dir)
        meta_df = pd.read_csv(str(precomputed_dir / "metadata.csv"))
        meta_df["subj"] = meta_df["subj"].astype(str).str.zfill(5)
        rows = meta_df[meta_df["subj"].isin(subjs)]

        vals = []
        n_sample = min(200, len(rows))
        sample_indices = rng.choice(len(rows), size=n_sample, replace=False)
        for i in sample_indices:
            row = rows.iloc[i]
            subj = str(row["subj"]).zfill(5)
            win_idx = int(row["window_idx"])
            for mode in ("xy", "x", "y"):
                npy_path = precomputed_dir / subj / f"win_{win_idx:03d}_{mode}.npy"
                if npy_path.exists():
                    mat = np.load(str(npy_path))
                    vals.append(mat.ravel())

        all_vals = np.concatenate(vals)
        if use_symmetric_normalize:
            scale = float(np.percentile(np.abs(all_vals), 98))
            scale = max(scale, 1e-6)
            if scale < 0.05:
                warnings.warn(
                    f"Global stats scale={scale:.4f} is very small (symmetric mode); "
                    "inputs may be sensitive to noise or sampling.",
                    UserWarning,
                    stacklevel=2,
                )
            return {"scale": scale, "vmin": -scale, "vmax": scale}
        vmin = 0.0
        vmax = float(np.percentile(all_vals, 98))
        if vmax <= vmin:
            vmax = max(vmin + 1e-6, float(np.max(all_vals)) + 1e-6) if len(all_vals) else 1.0
        if vmax < 0.05:
            warnings.warn(
                f"Global stats vmax={vmax:.4f} is very small; normalized inputs may be "
                "sensitive to noise (std=0.01) or sampling. Consider per_sample normalize or "
                "check precomputed matrix scale.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "vmin": vmin,
            "vmax": vmax,
        }