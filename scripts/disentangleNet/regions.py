"""
Shared region definitions for the LQ prototype.

This file exists to prevent silent drift between:
1. dataset preprocessing
2. action-basis initialization
3. future visualization / analysis helpers

The most important convention for the current project is that the LQ "mouth"
crop is the square block [188:307, 188:307]. This is intentionally wider than
the pure anatomical "mouth" block in the original grouped landmarks, because
the current experiments treat the mouth-centered motion submatrix as the main
training region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


@dataclass(frozen=True)
class RegionSpec:
    """Simple immutable descriptor for a square landmark submatrix."""

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        """Number of landmarks included in this region."""
        return self.end - self.start

    @property
    def region_slice(self) -> slice:
        """Slice used on both row and column axes of an NxN distance matrix."""
        return slice(self.start, self.end)


REGION_SPECS = {
    "full": RegionSpec(name="full", start=0, end=REGION_BOUNDARIES[-1]),
    "mouth": RegionSpec(name="mouth", start=188, end=307),
}


def get_region_spec(region: str) -> RegionSpec:
    """Resolve a region name to its configured matrix crop specification."""

    if region not in REGION_SPECS:
        raise ValueError(f"Unsupported region: {region}")
    return REGION_SPECS[region]


def crop_region(mat: np.ndarray, region: str) -> np.ndarray:
    """Crop a square pairwise-distance matrix to the configured facial region."""

    spec = get_region_spec(region)
    return mat[spec.region_slice, spec.region_slice]
