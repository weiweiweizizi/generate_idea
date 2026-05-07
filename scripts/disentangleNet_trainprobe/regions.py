"""
Shared region definitions for the tri-branch disentangleNet trainprobe model.

The full signed distance-difference matrix is 341 x 341. We keep that full
layout and carve it into three masked branches:

1. `mouth_self`
   - the square mouth super-block [188:307, 188:307]
2. `mouth_cross_other`
   - the off-diagonal mouth-vs-non-mouth interactions and their transpose
3. `other_self`
   - the non-mouth square self / cross block with mouth rows and cols removed

Side pooling must operate on the encoder feature map blocks that correspond to
these original matrix regions, rather than arbitrary adaptive pooling windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

FULL_BASIS_SIZE = 341
MOUTH_START = 188
MOUTH_END = 307

REGION_BOUNDARIES = [18, 38, 100, 130, 140, 188, 233, 307, 331, 341]
REGION_NAMES = [
    "forehead",
    "eyebrow",
    "eyehole",
    "eye_contour",
    "eye_iris",
    "nose",
    "around_mouth",
    "mouth",
    "cheek",
    "jaw",
]

UPPER_FACE_REGIONS = (
    "forehead",
    "eyebrow",
    "eyehole",
    "eye_contour",
    "eye_iris",
)
CHEEK_REGIONS = ("cheek",)
OTHERS_REGIONS = ("nose", "jaw")
MOUTH_REGION_GROUPS = {
    "upper_face": UPPER_FACE_REGIONS,
    "cheek": CHEEK_REGIONS,
    "others": OTHERS_REGIONS,
}
OTHER_SELF_REGION_GROUPS = {
    "upper_face": UPPER_FACE_REGIONS,
    "cheek": CHEEK_REGIONS,
    "others": OTHERS_REGIONS,
}


@dataclass(frozen=True)
class RegionSpec:
    """Immutable descriptor for one contiguous landmark span."""

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def midpoint(self) -> int:
        return self.start + self.size // 2

    @property
    def region_slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def left_slice(self) -> slice:
        return slice(self.start, self.midpoint)

    @property
    def right_slice(self) -> slice:
        return slice(self.midpoint, self.end)


REGION_SPECS = {
    name: RegionSpec(
        name=name,
        start=(0 if idx == 0 else REGION_BOUNDARIES[idx - 1]),
        end=REGION_BOUNDARIES[idx],
    )
    for idx, name in enumerate(REGION_NAMES)
}
REGION_SPECS["full"] = RegionSpec(name="full", start=0, end=FULL_BASIS_SIZE)


def get_region_spec(region: str) -> RegionSpec:
    if region not in REGION_SPECS:
        raise ValueError(f"Unsupported region: {region}")
    return REGION_SPECS[region]


def crop_region(mat: np.ndarray, region: str) -> np.ndarray:
    spec = get_region_spec(region)
    return mat[spec.region_slice, spec.region_slice]


def build_branch_masks() -> dict[str, np.ndarray]:
    """Create the three full-size branch masks used by trainprobe."""

    mouth_slice = slice(MOUTH_START, MOUTH_END)
    non_mouth_prefix = slice(0, MOUTH_START)
    non_mouth_suffix = slice(MOUTH_END, FULL_BASIS_SIZE)

    mouth_self = np.zeros((FULL_BASIS_SIZE, FULL_BASIS_SIZE), dtype=np.float32)
    mouth_self[mouth_slice, mouth_slice] = 1.0

    mouth_cross_other = np.zeros((FULL_BASIS_SIZE, FULL_BASIS_SIZE), dtype=np.float32)
    mouth_cross_other[mouth_slice, non_mouth_prefix] = 1.0
    mouth_cross_other[mouth_slice, non_mouth_suffix] = 1.0
    mouth_cross_other[non_mouth_prefix, mouth_slice] = 1.0
    mouth_cross_other[non_mouth_suffix, mouth_slice] = 1.0

    other_self = np.ones((FULL_BASIS_SIZE, FULL_BASIS_SIZE), dtype=np.float32)
    other_self[mouth_slice, :] = 0.0
    other_self[:, mouth_slice] = 0.0

    return {
        "mouth_self": mouth_self,
        "mouth_cross_other": mouth_cross_other,
        "other_self": other_self,
    }


def _scale_index(idx: int, feature_size: int) -> int:
    return int(math.floor(idx * feature_size / FULL_BASIS_SIZE))


def _scale_index_end(idx: int, feature_size: int) -> int:
    return int(math.ceil(idx * feature_size / FULL_BASIS_SIZE))


def project_slice_to_feature_map(src: slice, feature_size: int) -> slice:
    start = _scale_index(src.start, feature_size)
    end = _scale_index_end(src.stop, feature_size)
    if end <= start:
        end = min(feature_size, start + 1)
    return slice(start, end)


def project_region_half_to_feature_map(
    region: str,
    half: str,
    feature_size: int,
) -> slice:
    spec = get_region_spec(region)
    if half == "left":
        src = spec.left_slice
    elif half == "right":
        src = spec.right_slice
    else:
        raise ValueError(f"Unsupported half: {half}")
    return project_slice_to_feature_map(src, feature_size)
